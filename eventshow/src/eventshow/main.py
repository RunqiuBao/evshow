#!/usr/bAin/python3
import argparse
from pathlib import Path

import ..readwrite
import eventshow


def main():
    parser = argparse.ArgumentParser(description = "A commandline tool for transforming or visualizing  event raw file into frame based representations")
    parser.add_argument("input_file", help="Path to the input event raw data file.")
    parser.add_argument("-o", "--output_path", help="Path to save the output data.")
    parser.add_argument("-vo", "--viszonly", help="Only visualize, do not save to disk.")

    official_loaders = [
        loader for loader in dir(eventshow.readwrite) if not loader.startwith("_")
    ]
    parser.add_argument("-m", "--rw_module", default="base", help="official loaders: "f"{official_loaders}")
    args = parser.parse_args()

    eventshow.eventshow(
        args.rw_module,
        Path(args.input_file),
        Path(args.output_path)
    )


if __name__ == "__main__":
    main()
