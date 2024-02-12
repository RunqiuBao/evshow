import os


def eventshow(
    event_rw_module: str
    input_file_path: Path,
    output_path: Path,
    dt_ms: int = None,
    numevents_perslice: int = None    
) -> None:
    """
    Main logic in this func. Read raw events, transform to frame based representations and either
    visualize it or save to disk.
    """
    # import rw module
    if os.path.exists(event_rw_module):
        readwrite_module = importlib.machinery.SourceFileLoader(
            "..readwrite", args.rw_module
        ).load_module()
    else:
        print("Error: module %s not found" % event_rw_module)
        return

    eventReader = readwrite_module.EventReader(input_file_path, dt_ms)
