from .base import BaseEventReader


class EventReader(BaseEventReader):
    _rectifier = None  # used for event undistortion and rectification.
    h5f = None  # handler to the h5f file

    def __init__(self, filepath: Path, dt_ms: int = None, numevents_perslice: int = None):
        super().__init__(filepath, dt_ms, numevents_perslice)
        assert filepath.is_file()
        assert filepath.name.endswith('.h5')
        self.h5f = h5py.File(str(filepath), 'r')
        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

        self.event_slicer = EventSlicer(self.h5f)
        if dt_ms is not None:
	    self.dt_us = int(dt_ms * 1000)
	    self.t_start_us = self.event_slicer.get_start_time_us()
	    self.t_end_us = self.event_slicer.get_final_time_us()
	    self._length = (self.t_end_us - self.t_start_us) // self.dt_us
        elif numevents_perslice is not None:
            self.numevents_perslice = numevents_perslice

    def SetRectifyMap(self, rectify_filepath: Path):
        self._rectifier = h5py.File(str(rectify_filepath), 'r')

    @staticmethod
    def close_callback(h5f: h5py.File):
        h5f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._finalizer()

    def __iter__(self):
        return self

    def _GetNextSliceSbt(self):
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

    def __next__(self):
        if self.dt_us is not None:
            events = self._GetNextSliceSbt()
        return events
