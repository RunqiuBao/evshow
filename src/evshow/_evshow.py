import sys
import importlib.machinery
from pathlib import Path
import json
import tqdm

# Note: representation backends are imported lazily inside evshow() so that each
# path only pulls in its own (sometimes heavy / optional) dependencies, e.g.
# onnxruntime for concentrate or torch for e2vid.
from .event2frame.eros import EROS


def evshow(
    event_rw_module: str,
    input_file_path: Path,
    output_path: Path,
    dt_ms: int,
    numevents_perslice: int,
    is_use_concentrate: bool,
    is_use_e2vid: bool,
    is_use_eros: bool,
    num_frames_exit: int,
    is_save_lmdb: bool,
    existing_tsfile_path: Path,
    is_show: bool = False
) -> None:
    """
    Main logic in this func. Read raw events, transform to frame based representations and either
    visualize it or save to disk.
    """    
    # import rw module
    if (Path("readwrite") / event_rw_module).exists():  # Note: a 3rdparty module in arbitrary path
        readwrite_module = importlib.machinery.SourceFileLoader(
            "readwrite", event_rw_module
        ).load_module()
    else:
        try:
            readwrite_module = importlib.import_module(f"evshow.readwrite.{event_rw_module}")
        except ModuleNotFoundError:
            print(f"Error: module {event_rw_module!r} not found.\n", file=sys.stderr)
            sys.exit(1)

    show_supported = False
    if is_show:
        # --show only supports representations that produce a directly viewable
        # reconstruction. The default 'accumulate' path is raw event data, not a
        # visible frame, so it is not supported.
        show_supported = is_use_concentrate or is_use_e2vid or is_use_eros
        if not show_supported:
            print("Warning: --show is not supported for the 'accumulate' representation; skipping display (frames are still saved).")
        else:
            import matplotlib.pyplot as plt
            plt.ion()
            show_fig, show_ax = plt.subplots(num="evshow")
            show_ax.set_axis_off()
            show_im = None

    eventReader = readwrite_module.EventReader(input_file_path, dt_ms=dt_ms, numevents_perslice=numevents_perslice)
    eventFrameWriter = readwrite_module.EventFrameWriter(output_path, is_save_lmdb=is_save_lmdb)

    if is_use_concentrate:
        from .event2frame.concentration_net.concentration_net import EventFrameConcentrater
        event_concentrater = EventFrameConcentrater(numevents_perslice, eventReader.frameShape[1], eventReader.frameShape[0], stack_size=10)  # Note: stack_size is fixed to 10 due to network structure.
    elif is_use_e2vid:
        from .event2frame.e2vid.e2vid_net import Event2VideoConverter
        num_bins = 5
        e2vid_converter = Event2VideoConverter(eventReader.frameShape[1], eventReader.frameShape[0], num_bins)  # Note: rpg_e2vid pre-trained model uses 5 channels.
    elif is_use_eros:
        from .event2frame.eros import EROS
        eros_converter = EROS(eventReader.frameShape[0], eventReader.frameShape[1])
    else:
        from .event2frame.stacking import AccumulateEventsIntoFrame

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
        elif is_use_eros:
            eventFrameImg = eros_converter[events]
        else:
            eventFrameImg, eventFrame = AccumulateEventsIntoFrame(events, eventReader.frameShape)

        # write to disk
        eventFrameWriter.WriteOneFrame(indexBatch, eventFrameImg)

        # live display
        if is_show and show_supported and eventFrameImg is not None:
            if not plt.fignum_exists(show_fig.number):
                print("Display window closed, stopping.")
                break
            if show_im is None:
                show_im = show_ax.imshow(eventFrameImg, cmap='gray', vmin=0, vmax=255)
            else:
                show_im.set_data(eventFrameImg)
            show_ax.set_title(f"frame {indexBatch}")
            show_fig.canvas.draw_idle()
            plt.pause(0.001)

        if existing_tsfile_path is None:
            if len(events['t']) > 0:
                frame_end_ts = int(events['t'][-1] + eventReader.GetTimeOffsetUs())
            elif eventReader.t_start_us is not None:
                # empty time window (can happen with a small --dtms): the reader has
                # already advanced t_start_us to the end of this window, so use it.
                frame_end_ts = int(eventReader.t_start_us + eventReader.GetTimeOffsetUs())
            else:
                frame_end_ts = int(eventReader.GetTimeOffsetUs())
            list_frameEnd_ts.append(str(frame_end_ts))

    if existing_tsfile_path is None:
        # write timestamp file:
        with open(str(output_path / "timestamps.json"), 'w') as file:
            json.dump(list_frameEnd_ts, file)

    if is_show and show_supported:
        plt.ioff()
        if plt.fignum_exists(show_fig.number):
            print("Done! Close the display window to exit.")
            plt.show()

    print("Done!")
