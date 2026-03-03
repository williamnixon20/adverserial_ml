
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
import json

import torchaudio

import AudioSealChecker


os.add_dll_directory("C:\\ffmpeg\\bin") #fix the torchaudio issue
os.makedirs("elevenlab_sample_rate_compressed_audio", exist_ok=True)
####generate text-to-speech####




audio_list = os.listdir('elevenlab_generated_original')

for a in audio_list:


    
    audio_name = a.replace('.mp3','')

    compressed_sample_rate = 12000

    AudioSegment.from_mp3(f'elevenlab_generated_original/{a}').set_frame_rate(compressed_sample_rate).export(
        f'elevenlab_sample_rate_compressed_audio/{audio_name}_watermarked_compressed_{compressed_sample_rate}.mp3 ', 
        format='mp3', 
        bitrate='128k'
    )

