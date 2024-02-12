# event data readers supporting different datasets
from pathlib import Path
import lmdb
import shutil
import cv2
import weakref


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


class LmdbWriter():  # lmdb is multi read single write
    def __init__(self, write_path, map_size=1099511627776, isDummyMode=False):
        self.write_path = write_path
        self.map_size = map_size
        self.isDummyMode = isDummyMode
        if not isDummyMode:
            self.env = lmdb.open(write_path, map_size)
            self.txn = self.env.begin(write=True)

    def write(self, key, dataunit):
        if not self.isDummyMode:
            self.txn.put(key=key, value=dataunit)
    
    def commitchange(self):
        # commit change before ram is full
        if not self.isDummyMode:
            self.txn.commit()

    def endwriting(self):
        if not self.isDummyMode:
            self.env.close()


class BaseEventFrameWriter:
    _finalizer = None
    _lmdbWriter = None
    _pngSavePath = None

    def __init__(self, output_path: Path, is_save_png: bool, is_save_lmdb: bool):
        if is_save_lmdb:
            lmdb_path = output_path / "lmdb"
            if lmdb_path.exists():
                shutil.rmtree(lmdb_path)
            lmdb_path.mkdir(parents=True, exist_ok=True)
            self._lmdbWriter = LmdbWriter(lmdb_path)
            self._finalizer = weakref.finalize(self, self.CallbackAtEnd, self._lmdbWriter)
        if is_save_png:
            png_path = output_path / "png"
            if png_path.exists():
                shutil.rmtree(png_path)
            png_path.mkdir(parents=True, exist_ok=True)
            self._pngSavePath = png_path

    def WriteOneFrame(self, seq_index, frame_index, oneFrame):
        if self._lmdbWriter is not None:
            self._lmdbWriter.write(frame_index, oneFrame)
            code = '%03d_%06d' % (seq_index, frame_index)
            code = code.encode()
            self._lmdbWriter.write(code, oneFrame)

        if self._pngSavePath is not None:            
            png_path = self._pngSavePath / ('%03d_%06d.png' % (seq_index, frame_index))
            cv2.imwrite(str(png_path), oneFrame)

    @staticmethod
    def CallbackAtEnd(lmdb_writer):
        if lmdb_writer is not None:
            lmdb_writer.commitchange()
            lmdb_writer.endwriting()
            print("Commit to lmdb done!")
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._finalizer()

                
