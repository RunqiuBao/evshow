import sys
import importlib.machinery
from pathlib import Path

from .event2frame.stacking import AccumulateEventsIntoFrame


def eventshow(
    event_rw_module: str,
    input_file_path: Path,
    output_path: Path,
    seq_index: int,
    dt_ms: int,
    numevents_perslice: int,
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

    for indexBatch, events in enumerate(eventReader):
        if num_frames_exit and indexBatch >= num_frames_exit:
            break
        # TODO: visualize here
        # write to disk
        eventFrameImg, eventFrame = AccumulateEventsIntoFrame(events, eventReader.frameShape)
        eventFrameWriter.WriteOneFrame(seq_index if seq_index else 0, indexBatch, eventFrameImg)


    print("Done!")
