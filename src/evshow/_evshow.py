import sys
import importlib.machinery
from pathlib import Path
import json
import tqdm

from .event2frame.stacking import AccumulateEventsIntoFrame
from .event2frame.concentration_net.concentration_net import EventFrameConcentrater
from .event2frame.e2vid.e2vid_net import Event2VideoConverter


def evshow(
    event_rw_module: str,
    input_file_path: Path,
    output_path: Path,
    dt_ms: int,
    numevents_perslice: int,
    is_use_concentrate: bool,
    is_use_e2vid: bool,
    num_frames_exit: int,
    is_save_lmdb: bool,
    existing_tsfile_path: Path
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
            readwrite_module = importlib.import_module(f"evshow.readwrite.{event_rw_module}")
        except ModuleNotFoundError:
            print(f"Error: module {event_rw_module!r} not found.\n", file=sys.stderr)
            sys.exit(1)

    eventReader = readwrite_module.EventReader(input_file_path, dt_ms=dt_ms, numevents_perslice=numevents_perslice)
    eventFrameWriter = readwrite_module.EventFrameWriter(output_path, is_save_lmdb=is_save_lmdb)

    if is_use_concentrate:
        event_concentrater = EventFrameConcentrater(numevents_perslice, eventReader.frameShape[1], eventReader.frameShape[0], stack_size=10)  # Note: stack_size is fixed to 10 due to network structure.
    elif is_use_e2vid:
        num_bins = 5
        e2vid_converter = Event2VideoConverter(eventReader.frameShape[1], eventReader.frameShape[0], num_bins)  # Note: rpg_e2vid pre-trained model uses 5 channels.

    if existing_tsfile_path is None:
        list_frameEnd_ts = []
    else:
        with open(existing_tsfile_path, 'r') as file:
            list_frameEnd_ts = json.load(file)
            list_frameEnd_ts = [int(ts) for ts in list_frameEnd_ts]            

    lengthBatch = len(eventReader) if existing_tsfile_path is None else len(list_frameEnd_ts)
    lengthBatch = num_frames_exit if num_frames_exit is not None else lengthBatch
    for indexBatch in tqdm.tqdm(range(lengthBatch)):
        if existing_tsfile_path is None:
            events = next(eventReader)
        else:
            ts_end = list_frameEnd_ts[indexBatch]
            events = eventReader.GetSliceSbnByTimestamp(ts_end)        

        # TODO: visualize here
        if is_use_concentrate:
            eventFrameImg = event_concentrater[events]
        elif is_use_e2vid:
            eventFrameImg = e2vid_converter[events]
        else:
            eventFrameImg, eventFrame = AccumulateEventsIntoFrame(events, eventReader.frameShape)
        
        # write to disk
        eventFrameWriter.WriteOneFrame(indexBatch, eventFrameImg)

        if existing_tsfile_path is None:
            list_frameEnd_ts.append(str(events['t'][-1] + eventReader.GetTimeOffsetUs()))

    if existing_tsfile_path is None:
        # write timestamp file:
        with open(str(output_path / "timestamps.json"), 'w') as file:
            json.dump(list_frameEnd_ts, file)

    print("Done!")
