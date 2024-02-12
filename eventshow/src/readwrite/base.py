# event data readers supporting different datasets
from pathlib import Path
import weakref
import h5py
from _eventslicer import EventSlicer

class BaseEventReader:
    _finalizer = None
    event_slicer = None
    dt_us = None
    t_start_us = None
    t_end_us = None
    numevents_perslice = None
    _length = None
    
    def __init__(self, filepath: Path, dt_ms: int = None, numevents_perslice: int = None):
        pass

    def __next__(self):
        raise NotImplementedError

    def __len__(self):
        return self._length

