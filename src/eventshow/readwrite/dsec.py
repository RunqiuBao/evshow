from pathlib import Path
import weakref
import h5py
import numpy

from ._eventslicer import EventSlicer
from .base import BaseEventReader, BaseEventFrameWriter


class EventReader(BaseEventReader):
    _finalizer = None
    _rectifier = None  # used for event undistortion and rectification.
    h5f = None  # handler to the h5f file
    frameShape = None

    def __init__(self, filepath: Path, dt_ms: int = None, numevents_perslice: int = None):
        super().__init__(filepath, dt_ms, numevents_perslice)
        assert filepath.is_file()
        assert filepath.name.endswith('.h5')
        self.h5f = h5py.File(str(filepath), 'r')
        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

        self.frameShape = (640, 480)
        self.event_slicer = EventSlicer(self.h5f)
        self.SetRectifyMap(filepath.parent / "rectify_map.h5")  # Note: assume the rectify map is at the same folder with the data file
        if dt_ms is not None:
            self.dt_us = int(dt_ms * 1000)
            self.t_start_us = self.event_slicer.get_start_time_us()
            self.t_end_us = self.event_slicer.get_final_time_us()
            self._length = (self.t_end_us - self.t_start_us) // self.dt_us
        elif numevents_perslice is not None:
            self.numevents_perslice = numevents_perslice
            self.num_start = 0
            self.num_end = self.event_slicer.get_number_of_events()
            self._length = (self.num_end - self.num_start) // self.numevents_perslice

    def SetRectifyMap(self, rectify_filepath: Path):
        with h5py.File(str(rectify_filepath), 'r') as h5_rect:
            self._rectifier = h5_rect['rectify_map'][()]

    @staticmethod
    def close_callback(h5f: h5py.File):
        h5f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._finalizer()

    def __iter__(self):
        return self
    
    @staticmethod
    def _RectifyEvents(events, rectifier):
        # event rectification
        x = events['x']
        y = events['y']
        
        xy_rect = rectifier[y, x]
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]
        events['x'] = x_rect
        events['y'] = y_rect
        return events

    def _GetNextSliceSbt(self):
        t_end_us = self.t_start_us + self.dt_us
        if t_end_us > self.t_end_us:
            raise StopIteration
        events = self.event_slicer.get_events(self.t_start_us, t_end_us)
        if events is None:
            raise StopIteration

        events = self._RectifyEvents(events, self._rectifier)
        self.t_start_us = t_end_us
        return events
    
    def _GetNextSliceSbn(self):
        num_end = self.num_start + self.numevents_perslice
        if num_end > self.num_end:
            raise StopIteration
        events = self.event_slicer.get_events_byNumber(self.num_start, num_end)
        if events is None:
            raise StopIteration
        events = self._RectifyEvents(events, self._rectifier)
        self.num_start = num_end
        return events

    def __next__(self):
        if self.dt_us is not None:
            events = self._GetNextSliceSbt()
        elif self.numevents_perslice is not None:
            events = self._GetNextSliceSbn()
        else:
            raise NotImplementedError
        valid_mask = numpy.logical_and(numpy.where(events['x'] < self.frameShape[0], True, False), numpy.where(events['y'] < self.frameShape[1], True, False))
        events['x'] = events['x'][valid_mask]
        events['y'] = events['y'][valid_mask]
        events['t'] = events['t'][valid_mask]
        events['p'] = events['p'][valid_mask]
        return events


class EventFrameWriter(BaseEventFrameWriter):
    def __init__(self, output_path: Path, is_save_png: bool = True, is_save_lmdb: bool = False):
        super().__init__(output_path, is_save_png, is_save_lmdb)
        pass
