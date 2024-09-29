"""
How to save onnx model:

torch.onnx.export(
    self.concentration_net,
    event_stack['l'][0].unsqueeze(0),
    "/home/runqiu/code/se-cff/concentrate_events.onnx",
    export_params=True,
    opset_version=16,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
"""
import onnxruntime
import numpy

from ..stacking import MixedDensityEventStacking


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class EventFrameConcentrater:
    """
    Transform raw events into sharp frame based representations by (concentration_net)[https://github.com/yonseivnl/se-cff?tab=readme-ov-file].
    """
    stack_function = None
    ort_session = None

    def __init__(self,
            num_of_event,
            event_height,
            event_width,
            stack_size=10,
            **kwargs
        ) -> None:
        self.stack_function = MixedDensityEventStacking(stack_size, num_of_event,
                                                           event_height, event_width, **kwargs) 

        if onnxruntime.get_device() == 'GPU':
            print("Using GPU for onnx inference")
            providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
            self.ort_session = onnxruntime.InferenceSession("src/evshow/event2frame/concentration_net/concentrate_events.onnx", providers=providers)
        else:
            self.ort_session = onnxruntime.InferenceSession("src/evshow/event2frame/concentration_net/concentrate_events.onnx")

    def __getitem__(self, raw_events):
        event_data = self._pre_load_event_data(raw_events)
        event_data = self._post_load_event_data(event_data)
        event_data = numpy.transpose(event_data.squeeze(), (2, 0, 1))[numpy.newaxis, ...]
        event_frame_img = self._ConcentrateEventsByNeuralNetwork(event_data)
        return event_frame_img

    def _pre_load_event_data(self, raw_events):
        event_data = self.stack_function.pre_stack(raw_events, raw_events['t'][-1])
        return event_data

    def _post_load_event_data(self, event_data):
        event_data = self.stack_function.post_stack(event_data)
        return event_data

    def _ConcentrateEventsByNeuralNetwork(self, input_tensor):
        """
        Args:
            input_tensor: numpy ndarray. the shape needs to be (b, 10, h, w).
        """
        input_tensor = numpy.pad(input_tensor, ((0, 0), (0, 0), (0, 0), (0, 8)), mode='constant', constant_values=0)  # Note: padding to fit network input shape.
        ort_inputs = {self.ort_session.get_inputs()[0].name: input_tensor}      
        ort_outs = self.ort_session.run(None, ort_inputs)
        oneImg = numpy.squeeze(ort_outs)
        oneImg -= oneImg.min()
        oneImg *= 255 / oneImg.max()        
        return oneImg[:, :-8]  # Note: unpadding.
