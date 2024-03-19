import argparse
import librosa
import numpy as np
from scipy.io import wavfile
from pyin import pyin
from piptrack import piptrack

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Single File", required=False)
parser.add_argument("-v", "--vocals", help="Vocals File", required=False)
parser.add_argument("-a", "--accomp", help="Accompaniment File", required=False)
parser.add_argument("-b", "--bass", help="Bass File", required=False)
parser.add_argument("-d", "--drums", help="Drums File", required=False)
parser.add_argument("-o", "--output", help="Output Folder", required=False)
parser.add_argument("-m", "--method", help="Method (pyin,pip)", default="pyin")
parser.add_argument("-p", "--pulse", help="Pulse Width", default="0.5")
args = parser.parse_known_args()

def main(args):
    sr = 0.0
    length = 0.0
    output = []
    if args.file is not None:
        sr, length, output = call_processor(args.file, args.method)
    else:
        stem_audios = [None, None, None, None]
        for index, audio_arg in enumerate([args.vocals, args.accomp, args.bass, args.drums]):
            if audio_arg is not None:
                sr, length, stem_output = call_processor(audio_arg, args.method)
                stem_audios[index] = stem_output

        output = np.zeros(length)
        if stem_audios[0] is not None:
            output += 0.4 * stem_audios[0]

        backing_tracks = np.zeros(length)
        for index in range(1,4):
            if stem_audios[index] is not None:
                backing_tracks += stem_audios[index]

        backing_max = max(backing_tracks)
        if backing_max != 0:
            backing_tracks = backing_tracks / backing_max

        output += 0.3 * backing_tracks

    output_folder = "output/" if args.output is None else args.output
    wavfile.write("{}/output.wav".format(output_folder), int(sr), output.astype(np.float32))


def call_processor(audio_file, method):
    print("Processing File: {}".format(audio_file))
    audio, sr = librosa.load(audio_file, sr=None, mono=True)
    length = len(audio)
    output = None
    if method == "pyin":
        output = pyin(audio, sr)
    elif method == "pip":
        output = piptrack(audio, sr)

    return sr, length, output


main(args[0])
