import librosa
from scipy.signal import medfilt

import notes_utils
from tqdm import tqdm
import numpy as np

def pyin(audio, sr):
    duration = len(audio) / sr
    f0, v_flag, v_prob = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr,
                                      frame_length=2048, win_length=1024, hop_length=512, n_thresholds=100,
                                      beta_parameters=(2, 18), boltzmann_parameter=2, resolution=0.1,
                                      max_transition_rate=35.92,
                                      switch_prob=0.01, no_trough_prob=0.01)

    f0_parsed = notes_utils.parse_notes_per_frame(f0)

    sec_per_pyin_frame = duration / len(f0)
    # print("Sec per frame: ", sec_per_pyin_frame)
    # print("Samples per frame: ", sec_per_pyin_frame * sr)

    output_audio = []
    t_per_frame = np.arange(0, sec_per_pyin_frame, 1 / sr)
    frame_index = 0

    pbar = tqdm(total=len(f0_parsed) + 1)
    while frame_index < len(f0_parsed):
        if (not v_flag[frame_index]):
            # print("Skipping not-voiced note: ", frame_index)
            output_audio = np.concatenate([output_audio, t_per_frame])
            frame_index += 1
            pbar.update(1)
            continue

        # Find note duration
        note_stopping_index = notes_utils.find_stopping_index(f0_parsed, frame_index)
        note_length = note_stopping_index - frame_index
        note_seconds = note_length * sec_per_pyin_frame

        # print(
        #     "Note {} lasts {} seconds from frame {} up to {}".format(f0_parsed[frame_index], note_seconds, frame_index,
        #                                                              note_stopping_index))

        square_wave = notes_utils.square_signal(f0_parsed[frame_index], note_seconds, sr)
        square_wave /= np.max(np.abs(square_wave))
        output_audio = np.concatenate([output_audio, square_wave])

        frame_index = note_stopping_index
        pbar.update(note_length)
    pbar.close()

    output_audio = output_audio[0:len(audio)]  # trim any additional frames
    kernel_size = int((512 / 2) + 1)  # Adjust the kernel size as needed, currently half of hop size
    smoothed_signal = medfilt(output_audio, kernel_size)
    return smoothed_signal