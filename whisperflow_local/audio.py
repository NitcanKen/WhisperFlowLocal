"""Microphone capture at 16 kHz mono with a live input-level meter."""
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000


class Recorder:
    """Captures microphone audio between start() and stop().

    `level` is a 0..1 RMS estimate updated from the audio callback so the UI
    can render a live meter while recording.
    """

    def __init__(self):
        self._frames = []
        self._stream = None
        self._lock = threading.Lock()
        self.level = 0.0
        self.recording = False

    def _callback(self, indata, frames, time_info, status):
        with self._lock:
            self._frames.append(indata.copy())
        rms = float(np.sqrt(np.mean(np.square(indata))))
        # Map typical speech RMS (~0.01-0.3) onto 0..1 for display.
        self.level = min(1.0, rms * 8.0)

    def start(self) -> None:
        if self.recording:
            return
        with self._lock:
            self._frames = []
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        self.recording = True

    def stop(self) -> np.ndarray:
        if not self.recording:
            return np.zeros(0, dtype="float32")
        self.recording = False
        self.level = 0.0
        self._stream.stop()
        self._stream.close()
        self._stream = None
        with self._lock:
            if not self._frames:
                return np.zeros(0, dtype="float32")
            audio = np.concatenate(self._frames, axis=0).flatten()
            self._frames = []
        return audio


def save_wav(audio: np.ndarray, path: str) -> str:
    sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
    return path


def duration_seconds(audio: np.ndarray) -> float:
    return len(audio) / float(SAMPLE_RATE)
