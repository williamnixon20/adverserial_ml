
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # must be before any torch import

from pydub import AudioSegment
import os
from gtts import gTTS
from audioseal import AudioSeal
import soundfile as sf
import torch
import numpy as np
from datasets import load_dataset, Audio
import io
import librosa
import soundfile as sf
import torchaudio

import AudioSealChecker


os.add_dll_directory("C:\\ffmpeg\\bin") #fix the torchaudio issue
os.makedirs("elevenlab_speed_no_pitch_compressed_audio", exist_ok=True)
####generate text-to-speech####




audio_list = os.listdir('elevenlab_generated_original')

for a in audio_list:


    
    audio_name = a.replace('.mp3','')

    speed_factor = 0.9

    y, sr = librosa.load(f'elevenlab_generated_original/{a}', sr=None)
    y_stretched = librosa.effects.time_stretch(y, rate=speed_factor)
    sf.write('temp.wav', y_stretched, sr)
    AudioSegment.from_wav('temp.wav').export(
        f'elevenlab_speed_no_pitch_compressed_audio/{audio_name}_watermarked_speed_{int(speed_factor*100)}.mp3',
        format='mp3',
        bitrate='128k'
    )

    os.remove('temp.wav')
