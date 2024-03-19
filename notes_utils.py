import argparse
import numpy as np
import librosa
from scipy import signal

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pulse", help="Pulse Width", type=float, default=0.5)
args = parser.parse_known_args()

# def enveloping_square(note: str, seconds: float, sr: float, envelope_duration=4):
#     # t = np.linspace(0, seconds, int(round(sr * seconds)), endpoint=False)
#     t = np.arange(0, seconds, 1/sr)
#     freq = librosa.note_to_hz(note)
#     square_signal = signal.square(2 * np.pi * freq * t)
#     square_signal = square_signal * (1 - np.exp(-t * envelope_duration))
#     return square_signal


def square_signal(note: str, seconds: float, sr: float):
    # t = np.linspace(0, seconds, int(round(sr * seconds)), endpoint=False)
    t = np.arange(0, seconds, 1/sr)
    freq = librosa.note_to_hz(note)
    return signal.square(2 * np.pi * freq * t, duty=args[0].pulse)


def find_stopping_index(array: np.ndarray, index: int):
    value = array[index]
    look_ahead = index + 1
    for look_ahead in range(index + 1, len(array)):
        if array[look_ahead] == value:
            continue
        else:
            break
    return look_ahead


def parse_notes_per_frame(f0):
    f0_parsed = np.array(["---" for x in range(len(f0))])
    f0_parsed[~np.isnan(f0)] = librosa.hz_to_note(f0[~np.isnan(f0)], octave=True)
    return f0_parsed