import librosa
import numpy as np
from scipy.signal import medfilt

import notes_utils


def piptrack(audio, sr):
    duration = len(audio) / sr
    S = np.abs(librosa.stft(audio))
    pitches, magnitudes = librosa.piptrack(S=S, sr=sr, hop_length=512, win_length=1024)

    output_audio = []
    pitch_frames = len(pitches[0])
    seconds_per_pitch = duration / pitch_frames
    t_per_pitch = np.arange(0, seconds_per_pitch, 1/sr)

    pitches_parsed = np.ndarray(shape=pitches.shape, dtype=np.dtype('U4'))
    pitches_parsed[:][:] = "NaN"
    pitches_parsed[pitches != 0.0] = librosa.hz_to_note(pitches[pitches != 0])
    for t in range(len(pitches_parsed[0])):
        expressed_pitches = pitches_parsed[:][t]
        expressed_pitches = expressed_pitches[expressed_pitches != "NaN"]

        if len(expressed_pitches) == 0:
            output_audio = np.concatenate([output_audio, t_per_pitch * 0])
            continue

        notes, counts = np.unique(expressed_pitches[:], return_counts=True)
        most_frequent_note = notes[counts == max(counts)]

        if type(most_frequent_note) is np.ndarray:
            most_frequent_note = most_frequent_note[0]

        signal = notes_utils.square_signal(str(most_frequent_note), seconds_per_pitch, sr)
        output_audio = np.concatenate([output_audio, signal])

    output_audio = output_audio[0:len(audio)]  # trim any additional frames
    kernel_size = int((512 / 2) + 1)  # Adjust the kernel size as needed, currently half of hop size
    smoothed_signal = medfilt(output_audio, kernel_size)
    return smoothed_signal