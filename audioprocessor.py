from pydub import AudioSegment
import warnings
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")
import pydub
from pydub.silence import split_on_silence
import os
import speech_recognition as sr
import pandas as pd
import librosa
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
pd.set_option('display.max_colwidth', None)

class AudioProcessor:
    def __init__(self, audio_file_path, output_folder_path, summarize=False, persons=False):
        self.audio_file_path = audio_file_path
        self.output_folder_path = output_folder_path
        self.summarize = summarize
        self.persons = persons
        self.audio_file = AudioSegment.from_wav(self.audio_file_path)
        self.min_silence_len = 750
        self.silence_thresh = self.audio_file.dBFS - 16
        self.sentences = split_on_silence(self.audio_file, min_silence_len=self.min_silence_len, silence_thresh=self.silence_thresh, keep_silence=500)
        self.df = pd.DataFrame(columns=["id", "text", "spectral_centroid", "spectral_spread", "spectral_entropy", "spectral_flux", "spectral_rolloff", "zero_crossing_rate", "energy", "entropy_of_energy", "pitch"])

    def process(self):
        for i, sentence in enumerate(self.sentences):
            sentence.export(os.path.join(self.output_folder_path, "{}.wav".format(i)), format="wav")

        for file in os.listdir(self.output_folder_path):
            file_path = os.path.join(self.output_folder_path, file)
            print("File path:", file_path)
            features = self.extract_features(file_path)
            text = self.transcribe_audio(file_path)
            # if self.summarize:
            #     summary = self.summarize_text(text)
            #     summary_file_path = os.path.join(self.output_folder_path, "summary_{}.txt".format(os.path.splitext(file)[0]))
            #     with open(summary_file_path, 'w') as f:
            #         f.write(summary)
            # if self.persons:
            #     persons = self.extract_persons(text)
            #     persons_file_path = os.path.join(self.output_folder_path, "persons_{}.txt".format(os.path.splitext(file)[0]))
            #     with open(persons_file_path, 'w') as f:
            #         f.write('\n'.join(persons))
            self.df = self.df.append({"id": os.path.splitext(file)[0], "text": text, "spectral_centroid": features[0], "spectral_spread": features[1], "spectral_entropy": features[2], "spectral_flux": features[3], "spectral_rolloff": features[4], "zero_crossing_rate": features[5], "energy": features[6], "entropy_of_energy": features[7], "pitch": features[8]}, ignore_index=True)

        self.df.to_csv(os.path.join(self.output_folder_path, "features.csv"), index=False)



    @staticmethod
    def extract_features(file_path):
        print(file_path)
        audio, sr = librosa.load(file_path, sr=None)

        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_spread = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        spectral_entropy = librosa.feature.spectral_flatness(y=audio)[0]
        spectral_flux = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=512)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)[0]
        energy = np.array([sum(abs(audio[i:i + 512] ** 2)) for i in range(0, len(audio), 512)])
        entropy_of_energy = -sum([(i / sum(energy)) * np.log(i / sum(energy)) for i in energy])
        pitch, _ = librosa.core.piptrack(y=audio, sr=sr)
        pitch = pitch.mean(axis=1)
        pitch = pitch[~np.isnan(pitch)]
        return [spectral_centroid.mean(), spectral_spread.mean(), spectral_entropy.mean(), spectral_flux.mean(), spectral_rolloff.mean(), zero_crossing_rate.mean(), energy.mean(), entropy_of_energy, pitch.mean()]

    def transcribe_audio(self, file_path):
        r = sr.Recognizer()
        audio_file = sr.AudioFile(file_path)
        with audio_file as source:
            audio = r.record(source)
        try:
            text = r.recognize_google(audio)
        except sr.UnknownValueError:
            text = "Error"
        return text