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

