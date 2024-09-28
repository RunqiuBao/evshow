import onnxruntime
import numpy
import torch

from .utils import CropParameters, EventPreprocessor, IntensityRescaler, ImageFilter, UnsharpMaskFilter, events_to_voxel_grid, events_to_voxel_grid_pytorch


class Event2VideoConverter:
    """
    Transform raw events into intensity frames by [rpg_e2vid][https://github.com/uzh-rpg/rpg_e2vid]
    """
    ort_session = None  # onnx model for video reconstruction
    model_params = None  # params of the onnx model

    def __init__(self, height: int, width: int, num_bins: int) -> None:
        self.device = torch.device('cuda:' + str(torch.cuda.current_device())) if torch.cuda.is_available() else torch.device('cpu')
        providers = [("CUDAExecutionProvider", {"device_id": torch.cuda.current_device()})]  # Note: if cuda not available, it will fall back to CPU.
        self.ort_session = onnxruntime.InferenceSession("src/evshow/event2frame/e2vid/e2vid_net.onnx", providers=providers)
        self.model_params = {
            "imageHeight": height,
            "imageWidth": width,
            "num_bins": num_bins,
            "num_encoders": 3,
            "no_normalize": False,  # disable normalization of input event tensors (saves a bit of time, but may produce slightly worse results)
            "flip": False,
            "auto_hdr": True,  # If True, will compute Imin and Imax automatically.
            "auto_hdr_median_filter_size": 10,  # Size of the median filter window used to smooth temporally Imin and Imax
            "Imin": 0.0,  # Min intensity for intensity rescaling (linear tone mapping).
            "Imax": 1.0,  # Max intensity value for intensity rescaling (linear tone mapping).
            "bilateral_filter_sigma": 0.0,
            "unsharp_mask_amount": 0.3,
            "unsharp_mask_sigma": 1.0
        }
        self.crop = CropParameters(width, height, self.model_params["num_encoders"])
        self.event_preprocessor = EventPreprocessor(self.model_params["no_normalize"], self.model_params["flip"])
        self.intensity_rescaler = IntensityRescaler(self.model_params["auto_hdr"], self.model_params["auto_hdr_median_filter_size"], self.model_params["Imin"], self.model_params["Imax"])
        self.image_filter = ImageFilter(self.model_params["bilateral_filter_sigma"])
        self.unsharp_mask_filter = UnsharpMaskFilter(self.model_params["unsharp_mask_amount"], self.model_params["unsharp_mask_sigma"], device=self.device)

        self.last_states_for_each_channel = {'grayscale': None}  # states for RNN

    @staticmethod
    def _get_zeros_states(B: int, C: int, H: int, W: int) -> list:
        return [
            numpy.zeros((B, C, H // 2, W // 2), dtype="float32"),
            numpy.zeros((B, 2 * C, H // 4, W // 4), dtype="float32"),
            numpy.zeros((B, 4 * C, H // 8, W // 8), dtype="float32"),
        ]

    def __getitem__(self, eventsDict: dict) -> numpy.array:
        """
        Args:
            eventsDict: dict with 1d arraies, keys are [ts, x, y, p].
        """
        numEvents = eventsDict['p'].shape[0]
        events = numpy.zeros((numEvents, 4), dtype="float")
        events[:, 0] = eventsDict["t"]
        events[:, 1] = eventsDict["x"]
        events[:, 2] = eventsDict["y"]
        events[:, 3] = eventsDict["p"]

        if not torch.cuda.is_available():
            event_tensor = events_to_voxel_grid(events,
                                                num_bins=self.model_params["num_bins"],
                                                width=self.model_params["imageWidth"],
                                                height=self.model_params["imageHeight"])
            event_tensor = torch.from_numpy(event_tensor)
        else:
            event_tensor = events_to_voxel_grid_pytorch(events,
                                                        num_bins=self.model_params["num_bins"],
                                                        width=self.model_params["imageWidth"],
                                                        height=self.model_params["imageHeight"],
                                                        device=self.device)

        event_tensor = self.event_preprocessor(event_tensor)

        # Resize tensor to [1 x C x crop_size x crop_size] by applying zero padding
        events_for_each_channel = {'grayscale': self.crop.pad(event_tensor).unsqueeze(dim=0)}
        reconstructions_for_each_channel = {}

        # Reconstruct new intensity image (grayscale)
        for channel in events_for_each_channel.keys():
            if self.last_states_for_each_channel[channel] is None:
                B, C, H, W = events_for_each_channel[channel].shape
                C = 64
                zeros_states = self._get_zeros_states(B, C, H, W)
                ort_inputs = {
                    self.ort_session.get_inputs()[0].name: events_for_each_channel[channel].cpu().numpy(),
                    self.ort_session.get_inputs()[1].name: zeros_states[0],
                    self.ort_session.get_inputs()[2].name: zeros_states[0],
                    self.ort_session.get_inputs()[3].name: zeros_states[1],
                    self.ort_session.get_inputs()[4].name: zeros_states[1],
                    self.ort_session.get_inputs()[5].name: zeros_states[2],
                    self.ort_session.get_inputs()[6].name: zeros_states[2]
                }
            else:
                ort_inputs = {
                    self.ort_session.get_inputs()[0].name: events_for_each_channel[channel].cpu().numpy(),
                    self.ort_session.get_inputs()[1].name: self.last_states_for_each_channel[channel][0][0].cpu().numpy(),
                    self.ort_session.get_inputs()[2].name: self.last_states_for_each_channel[channel][0][1].cpu().numpy(),
                    self.ort_session.get_inputs()[3].name: self.last_states_for_each_channel[channel][1][0].cpu().numpy(),
                    self.ort_session.get_inputs()[4].name: self.last_states_for_each_channel[channel][1][1].cpu().numpy(),
                    self.ort_session.get_inputs()[5].name: self.last_states_for_each_channel[channel][2][0].cpu().numpy(),
                    self.ort_session.get_inputs()[6].name: self.last_states_for_each_channel[channel][2][1].cpu().numpy()
                }
            ort_outs = self.ort_session.run(None, ort_inputs)
            new_predicted_frame = torch.from_numpy(ort_outs[0]).to(self.device)
            states =[
                (torch.from_numpy(ort_outs[1]).to(self.device), torch.from_numpy(ort_outs[2]).to(self.device)),
                (torch.from_numpy(ort_outs[3]).to(self.device), torch.from_numpy(ort_outs[4]).to(self.device)),
                (torch.from_numpy(ort_outs[5]).to(self.device), torch.from_numpy(ort_outs[6]).to(self.device))
            ]

            self.last_states_for_each_channel[channel] = states

            # Output reconstructed image
            crop = self.crop if channel == 'grayscale' else None

            # Unsharp mask (on GPU)
            new_predicted_frame = self.unsharp_mask_filter(new_predicted_frame)

            # Intensity rescaler (on GPU)
            new_predicted_frame = self.intensity_rescaler(new_predicted_frame)

            reconstructions_for_each_channel[channel] = new_predicted_frame[0, 0, crop.iy0:crop.iy1, crop.ix0:crop.ix1].cpu().numpy()

        out = reconstructions_for_each_channel['grayscale']

        # Post-processing, e.g bilateral filter (on CPU)
        out = self.image_filter(out)

        return out
