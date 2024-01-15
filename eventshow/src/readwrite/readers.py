#!/usr/bin/python3
# event data readers supporting different datasets
from pathlib import Path
import weakref
import h5py
from eventslicer import EventSlicer

class BaseEventReader:
    h5f = None
    _finalizer = None
    event_slicer = None
    dt_us = None
    t_start_us = None
    t_end_us = None
    _length = None
    
    def __init__(self):
        pass

    def __next__(self):
        raise NotImplementedError


class DSECEventReaderAbstract(BaseEventReader):
    _rectifier = None  # used for event undistortion and rectification.

    def __init__(self, filepath: Path, rectify_filepath: Path, dt_milliseconds: int):
        super().__init__()
        assert filepath.is_file()
        assert filepath.name.endswith('.h5')
        self.h5f = h5py.File(str(filepath), 'r')
        self._rectifier = h5py.File(str(rectify_filepath), 'r')
        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

        self.event_slicer = EventSlicer(self.h5f)
        self.dt_us = int(dt_milliseconds * 1000)
        self.t_start_us = self.event_slicer.get_start_time_us()
        self.t_end_us = self.event_slicer.get_final_time_us()
        self._length = (self.t_end_us - self.t_start_us)//self.dt_us

    @staticmethod
    def close_callback(h5f: h5py.File):
        h5f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._finalizer()

    def __iter__(self):
        return self

    def __len__(self):
        return self._length

    def __next__(self):
        t_end_us = self.t_start_us + self.dt_us
        if t_end_us > self.t_end_us:
            raise StopIteration
        events = self.event_slicer.get_events(self.t_start_us, t_end_us)
        if events is None:
            raise StopIteration

        self.t_start_us = t_end_us

        # event rectification
        x = events['x']
        y = events['y']
        xy_rect = self._rectifier[x, y]
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]
        events['x'] = x_rect
        events['y'] = y_rect
        return events
