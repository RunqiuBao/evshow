from torch.nn import ReflectionPad2d
import torch.nn.functional as F
import torch
from math import ceil, floor
import numpy as np
from collections import deque
import scipy.stats

from .timers import Timer, CudaTimer


# ------------------------------------- from rpg_e2vid -------------------------------------
def gkern(kernlen=5, nsig=1.0):
    """Returns a 2D Gaussian kernel array."""
    """https://stackoverflow.com/a/29731818"""
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(scipy.stats.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return torch.from_numpy(kernel).float()


def optimal_crop_size(max_size, max_subsample_factor):
    """ Find the optimal crop size for a given max_size and subsample_factor.
        The optimal crop size is the smallest integer which is greater or equal than max_size,
        while being divisible by 2^max_subsample_factor.
    """
    crop_size = int(pow(2, max_subsample_factor) * ceil(max_size / pow(2, max_subsample_factor)))
    return crop_size


class CropParameters:
    """ Helper class to compute and store useful parameters for pre-processing and post-processing
        of images in and out of E2VID.
        Pre-processing: finding the best image size for the network, and padding the input image with zeros
        Post-processing: Crop the output image back to the original image size
    """

    def __init__(self, width, height, num_encoders):

        self.height = height
        self.width = width
        self.num_encoders = num_encoders
        self.width_crop_size = optimal_crop_size(self.width, num_encoders)
        self.height_crop_size = optimal_crop_size(self.height, num_encoders)

        self.padding_top = ceil(0.5 * (self.height_crop_size - self.height))
        self.padding_bottom = floor(0.5 * (self.height_crop_size - self.height))
        self.padding_left = ceil(0.5 * (self.width_crop_size - self.width))
        self.padding_right = floor(0.5 * (self.width_crop_size - self.width))
        self.pad = ReflectionPad2d((self.padding_left, self.padding_right, self.padding_top, self.padding_bottom))

        self.cx = floor(self.width_crop_size / 2)
        self.cy = floor(self.height_crop_size / 2)

        self.ix0 = self.cx - floor(self.width / 2)
        self.ix1 = self.cx + ceil(self.width / 2)
        self.iy0 = self.cy - floor(self.height / 2)
        self.iy1 = self.cy + ceil(self.height / 2)


class EventPreprocessor:
    """
    Utility class to preprocess event tensors.
    Can perform operations such as hot pixel removing, event tensor normalization,
    or flipping the event tensor.
    """

    def __init__(self, no_normalize: bool, flip: bool):

        print('== Event preprocessing ==')
        self.no_normalize = no_normalize
        if self.no_normalize:
            print('!!Will not normalize event tensors!!')
        else:
            print('Will normalize event tensors.')

        self.hot_pixel_locations = []

        self.flip = flip
        if self.flip:
            print('Will flip event tensors.')

    def __call__(self, events):

        # Remove (i.e. zero out) the hot pixels
        for x, y in self.hot_pixel_locations:
            events[:, :, y, x] = 0

        # Flip tensor vertically and horizontally
        if self.flip:
            events = torch.flip(events, dims=[2, 3])

        # Normalize the event tensor (voxel grid) so that
        # the mean and stddev of the nonzero values in the tensor are equal to (0.0, 1.0)
        if not self.no_normalize:
            with CudaTimer('Normalization'):
                nonzero_ev = (events != 0)
                num_nonzeros = nonzero_ev.sum()
                if num_nonzeros > 0:
                    # compute mean and stddev of the **nonzero** elements of the event tensor
                    # we do not use PyTorch's default mean() and std() functions since it's faster
                    # to compute it by hand than applying those funcs to a masked array
                    mean = events.sum() / num_nonzeros
                    stddev = torch.sqrt((events ** 2).sum() / num_nonzeros - mean ** 2)
                    mask = nonzero_ev.float()
                    events = mask * (events - mean) / stddev

        return events


class IntensityRescaler:
    """
    Utility class to rescale image intensities to the range [0, 1],
    using (robust) min/max normalization.
    Optionally, the min/max bounds can be smoothed over a sliding window to avoid jitter.
    """

    def __init__(self, auto_hdr: bool, auto_hdr_median_filter_size: int, Imin: float, Imax: float):
        self.auto_hdr = auto_hdr
        self.intensity_bounds = deque()
        self.auto_hdr_median_filter_size = auto_hdr_median_filter_size
        self.Imin = Imin
        self.Imax = Imax

    def __call__(self, img):
        """
        param img: [1 x 1 x H x W] Tensor taking values in [0, 1]
        """
        if self.auto_hdr:
            with CudaTimer('Compute Imin/Imax (auto HDR)'):
                Imin = torch.min(img).item()
                Imax = torch.max(img).item()

                # ensure that the range is at least 0.1
                Imin = np.clip(Imin, 0.0, 0.45)
                Imax = np.clip(Imax, 0.55, 1.0)

                # adjust image dynamic range (i.e. its contrast)
                if len(self.intensity_bounds) > self.auto_hdr_median_filter_size:
                    self.intensity_bounds.popleft()

                self.intensity_bounds.append((Imin, Imax))
                self.Imin = np.median([rmin for rmin, rmax in self.intensity_bounds])
                self.Imax = np.median([rmax for rmin, rmax in self.intensity_bounds])

        with CudaTimer('Intensity rescaling'):
            img = 255.0 * (img - self.Imin) / (self.Imax - self.Imin)
            img.clamp_(0.0, 255.0)
            img = img.byte()  # convert to 8-bit tensor

        return img


class ImageFilter:
    """
    Utility class to perform some basic filtering on reconstructed images.
    """

    def __init__(self, bilateral_filter_sigma: float):
        self.bilateral_filter_sigma = bilateral_filter_sigma

    def __call__(self, img):

        if self.bilateral_filter_sigma:
            with Timer('Bilateral filter (sigma={:.2f})'.format(self.bilateral_filter_sigma)):
                filtered_img = np.zeros_like(img)
                filtered_img = cv2.bilateralFilter(
                    img, 5, 25.0 * self.bilateral_filter_sigma, 25.0 * self.bilateral_filter_sigma)
                img = filtered_img

        return img


class UnsharpMaskFilter:
    """
    Utility class to perform unsharp mask filtering on reconstructed images.
    """

    def __init__(self, unsharp_mask_amount: float, unsharp_mask_sigma: float, device: torch.device):
        self.unsharp_mask_amount = unsharp_mask_amount
        self.unsharp_mask_sigma = unsharp_mask_sigma
        self.gaussian_kernel_size = 5
        self.gaussian_kernel = gkern(self.gaussian_kernel_size,
                                     self.unsharp_mask_sigma).unsqueeze(0).unsqueeze(0).to(device)

    def __call__(self, img):
        if self.unsharp_mask_amount > 0:
            with CudaTimer('Unsharp mask'):
                blurred = F.conv2d(img, self.gaussian_kernel,
                                   padding=self.gaussian_kernel_size // 2)
                img = (1 + self.unsharp_mask_amount) * img - self.unsharp_mask_amount * blurred
        return img


def events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), dtype='float32').ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype('int')
    ys = events[:, 2].astype('int')
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype('int')
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid


def events_to_voxel_grid_pytorch(events, num_bins, width, height, device):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    :param device: device to use to perform computations
    :return voxel_grid: PyTorch event tensor (on the device specified)
    """

    DeviceTimer = CudaTimer if device.type == 'cuda' else Timer

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    with torch.no_grad():

        events_torch = torch.from_numpy(events)
        with DeviceTimer('Events -> Device (voxel grid)'):
            events_torch = events_torch.to(device)

        with DeviceTimer('Voxel grid voting'):
            voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32, device=device).flatten()

            # normalize the event timestamps so that they lie between 0 and num_bins
            last_stamp = events_torch[-1, 0]
            first_stamp = events_torch[0, 0]
            deltaT = last_stamp - first_stamp

            if deltaT == 0:
                deltaT = 1.0

            events_torch[:, 0] = (num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
            ts = events_torch[:, 0]
            xs = events_torch[:, 1].long()
            ys = events_torch[:, 2].long()
            pols = events_torch[:, 3].float()
            pols[pols == 0] = -1  # polarity should be +1 / -1

            tis = torch.floor(ts)
            tis_long = tis.long()
            dts = ts - tis
            vals_left = pols * (1.0 - dts.float())
            vals_right = pols * dts.float()

            valid_indices = tis < num_bins
            valid_indices &= tis >= 0
            voxel_grid.index_add_(dim=0,
                                  index=xs[valid_indices] + ys[valid_indices]
                                  * width + tis_long[valid_indices] * width * height,
                                  source=vals_left[valid_indices])

            valid_indices = (tis + 1) < num_bins
            valid_indices &= tis >= 0

            voxel_grid.index_add_(dim=0,
                                  index=xs[valid_indices] + ys[valid_indices] * width
                                  + (tis_long[valid_indices] + 1) * width * height,
                                  source=vals_right[valid_indices])

        voxel_grid = voxel_grid.view(num_bins, height, width)

    return voxel_grid
