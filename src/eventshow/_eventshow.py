import sys
import importlib.machinery
from pathlib import Path

from .event2frame.stacking import AccumulateEventsIntoFrame
from .event2frame.ai import EventFrameConcentrater


def eventshow(
    event_rw_module: str,
    input_file_path: Path,
    output_path: Path,
    dt_ms: int,
    numevents_perslice: int,
    is_use_concentrate: bool,
    num_frames_exit: int,
    is_save_lmdb: bool
) -> None:
    """
    Main logic in this func. Read raw events, transform to frame based representations and either
    visualize it or save to disk.
    """    
    # import rw module
    if (Path("readwrite") / event_rw_module).exists():  # Note: a 3rdparty module in arbitrary path
        readwrite_module = importlib.machinery.SourceFileLoader(
            " readwrite", event_rw_module
        ).load_module()
    else:
        try:
            readwrite_module = importlib.import_module(f"eventshow.readwrite.{event_rw_module}")
        except ModuleNotFoundError:
            print(f"Error: module {event_rw_module!r} not found.\n", file=sys.stderr)
            sys.exit(1)

    eventReader = readwrite_module.EventReader(input_file_path, dt_ms=dt_ms, numevents_perslice=numevents_perslice)
    eventFrameWriter = readwrite_module.EventFrameWriter(output_path, is_save_lmdb=is_save_lmdb)

    if is_use_concentrate:
        event_concentrater = EventFrameConcentrater(numevents_perslice, eventReader.frameShape[1], eventReader.frameShape[0], stack_size=10)  # Note: stack_size is fixed to 10 due to network structure.

    for indexBatch, events in enumerate(eventReader):
        if num_frames_exit and indexBatch >= num_frames_exit:
            break

        # TODO: visualize here
        if is_use_concentrate:
            eventFrameImg = event_concentrater[events]
        else:
            eventFrameImg, eventFrame = AccumulateEventsIntoFrame(events, eventReader.frameShape)
        
        # write to disk
        eventFrameWriter.WriteOneFrame(indexBatch, eventFrameImg)

    print("Done!")
