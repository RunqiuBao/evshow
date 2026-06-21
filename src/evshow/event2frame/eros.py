import numpy
import cv2


class EROS:
    """
    Exponential Reduced Ordinal Surface (EROS) representation.

    Ported from event-driven/algs/surface.h::EROS. EROS keeps a persistent
    surface that decays exponentially. For each incoming event at (x, y) a
    kernel_size x kernel_size neighborhood is multiplicatively decayed by
    odecay = parameter ** (1 / kernel_size) and then the center pixel is set
    to 1.0. The surface is sampled at the end of each event slice to produce a
    frame.

    Defaults (kernel_size=7, parameter=0.3) match the original event-driven
    EROS drawer (block_size / alpha), as does the cosmetic post-processing
    applied when sampling a frame.
    """

    def __init__(self, width, height, kernel_size=7, parameter=0.3):
        assert kernel_size % 2 == 1, "kernel_size must be odd."
        self.width = width
        self.height = height
        self.kernel_size = kernel_size
        self.half_kernel = kernel_size // 2
        self.parameter = parameter
        self.odecay = parameter ** (1.0 / kernel_size)
        # Pad the surface by half_kernel on every side so that the kernel window
        # around border events stays in-bounds (mirrors the padding used in the
        # C++ implementation where the center sits at (y+half_kernel, x+half_kernel)).
        self.surf = numpy.zeros(
            (height + 2 * self.half_kernel, width + 2 * self.half_kernel),
            dtype=numpy.float64,
        )

    def update(self, x, y):
        """
        Update the surface for a single event at raw image coordinate (x, y).
        """
        k = self.kernel_size
        hk = self.half_kernel
        # The kernel window's top-left corner in padded coordinates is (y, x),
        # so the event center lands at (y + half_kernel, x + half_kernel).
        self.surf[y:y + k, x:x + k] *= self.odecay
        self.surf[y + hk, x + hk] = 1.0

    def GetSurface(self):
        """
        Return the (unpadded) surface as a (height, width) float array.
        """
        hk = self.half_kernel
        return self.surf[hk:hk + self.height, hk:hk + self.width]

    def __getitem__(self, events):
        """
        Update the surface with one slice of events and return the current
        surface as a uint8 image. EROS is order dependent, so events are
        processed sequentially in the order they are received.
        """
        xs = events['x'].astype(numpy.int64)
        ys = events['y'].astype(numpy.int64)
        mask = (xs >= 0) & (xs < self.width) & (ys >= 0) & (ys < self.height)
        xs, ys = xs[mask], ys[mask]

        for x, y in zip(xs.tolist(), ys.tolist()):
            self.update(x, y)

        surface = self.GetSurface()
        img = (surface * 255.0).astype(numpy.uint8)

        # Cosmetic post-processing matching event-driven vFramer's erosDrawer:
        # median blur -> Gaussian blur -> min-max normalize to [0, 512] (the
        # uint8 output saturates above 255, which brightens the surface).
        img = cv2.medianBlur(img, 3)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.normalize(img, None, 0, 512, cv2.NORM_MINMAX)
        return img
