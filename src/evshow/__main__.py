#!/usr/bin/python3
import argparse
from pathlib import Path

import evshow
import evshow.readwrite


def main():
    parser = argparse.ArgumentParser(description = "A commandline tool for transforming or visualizing  event raw file into frame based representations")
    parser.add_argument("input_file", help="Path to the input event raw data file.")
    parser.add_argument("-o", "--output_path", help="Path to save the output data.")
    parser.add_argument("-vo", "--viszonly", help="Only visualize, do not save to disk.")
    parser.add_argument("--dtms", help="Time interval in ms for each frame.")
    parser.add_argument("--numevents", help="Number of events per frame.")
    parser.add_argument("--tsfile", help="Read in a list of timestamps as reference (from the same folder as input_file). Only support timestamps + number_of_events currently.")
    parser.add_argument("--concentrate", action="store_true", help="use concentrate network to generate sharp frames. Need to set --numevents.")
    parser.add_argument("--e2vid", action="store_true", help="use e2vid network to predict grayscale image from events. Need to set --numevents.")
    parser.add_argument("--numframes", help="Number of event frames for early quit.")
    parser.add_argument("--savelmdb", action="store_true", help="whether save output to lmdb format.")

    official_loaders = [
        loader for loader in dir(evshow.readwrite) if not loader.startswith("_")
    ]
    parser.add_argument("-m", "--rw_module", default="base", help="official loaders: "f"{official_loaders}")
    args = parser.parse_args()

    if not args.output_path:
        print("Error: output_path is required.")
        return

    if args.dtms and args.numevents:
        print("Error: only one of --dtms and --numevents can be set.")
        return
    
    if args.concentrate and not args.numevents:
        print("Error: --concentrate need to set --numevents.")
        return

    if args.e2vid and not args.numevents:
        print("Error: --e2vid need to set --numevents.")
        return

    if args.tsfile and not args.numevents:
        print("Error: --tsfile need to set --numevents.")
        return

    evshow.evshow(
        args.rw_module,
        Path(args.input_file),
        Path(args.output_path),
        dt_ms=int(args.dtms) if args.dtms else None,
        numevents_perslice=int(args.numevents) if args.numevents else None,
        is_use_concentrate=args.concentrate if args.concentrate else False,
        is_use_e2vid=args.e2vid if args.e2vid else False,
        num_frames_exit=int(args.numframes) if args.numframes else None,
        is_save_lmdb=args.savelmdb if args.savelmdb else False,
        existing_tsfile_path=Path(args.tsfile) if args.tsfile else None
    )


if __name__ == "__main__":
    main()
