import pytest
from pathlib import Path
import os

import evshow


@pytest.fixture
def inputs_e2vid():
    return {
        "event_rw_module": "dsec",
        "input_file_path": (Path(os.path.dirname(__file__)) / "../data/sample_dsecformat.h5"),
        "output_path": Path("../data/seq_sample_e2vid/"),
        "dt_ms": None,
        "numevents_perslice": 200000,
        "is_use_concentrate": False,
        "is_use_e2vid": True,
        "num_frames_exit": 10,
        "is_save_lmdb": False,
        "existing_tsfile_path": None
    }

@pytest.fixture
def inputs_concentrate():
    return {
        "event_rw_module": "dsec",
        "input_file_path": (Path(os.path.dirname(__file__)) / "../data/sample_dsecformat.h5"),
        "output_path": Path("../data/seq_sample_concentrate/"),
        "dt_ms": None,
        "numevents_perslice": 200000,
        "is_use_concentrate": True,
        "is_use_e2vid": False,
        "num_frames_exit": 10,
        "is_save_lmdb": False,
        "existing_tsfile_path": None
    }

def test_e2vid(inputs_e2vid):    
    evshow.evshow(**inputs_e2vid)
    for i in range(10):
        assert (inputs_e2vid["output_path"] / "png" / f"{i:06d}.png").exists()

def test_concentrate(inputs_concentrate):
    evshow.evshow(**inputs_concentrate)
    for i in range(10):
        assert (inputs_concentrate["output_path"] / "png" / f"{i:06d}.png").exists()
