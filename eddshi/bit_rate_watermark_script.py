
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
os.makedirs("watermarked_audio", exist_ok=True)
os.makedirs("watermarked_sample_rate_compressed_audio", exist_ok=True)
####generate text-to-speech####




ds = load_dataset("MLCommons/peoples_speech", "clean_sa", split="train", streaming=True).take(100)
ds = ds.cast_column("audio", Audio(sampling_rate=16000)) #resample

for d in ds:


    
    audio_segment_id = d['id'].replace('.flac','')
    audio_segment = d['audio']
    sampling_rate = audio_segment["sampling_rate"]
    audio_array = audio_segment["array"]
   

    #####AudioSeal######

    model = AudioSeal.load_generator("audioseal_wm_16bits")
    model.eval()

    waveform = torch.from_numpy(audio_array).float().unsqueeze(0)
    wav = waveform.unsqueeze(0)


    msg = torch.randint(0, 2, (wav.shape[0], model.msg_processor.nbits), device=wav.device)
    watermark = model.get_watermark(wav, message = msg)


    # watermark = model.get_watermark(wav)
    watermarked_audio = wav + watermark

    watermarked_audio = watermarked_audio.squeeze(0)

    torchaudio.save(f'watermarked_audio/{audio_segment_id}_watermarked.wav',  watermarked_audio, 16000)

    compressed_bit_rate = "64k"

    AudioSegment.from_wav(f'watermarked_audio/{audio_segment_id}_watermarked.wav').export(
        f'watermarked_bit_rate_compressed_audio/{audio_segment_id}_watermarked_compressed_{compressed_bit_rate}.mp3 ', 
        format='mp3', 
        bitrate=f'{compressed_bit_rate}'
    )

    AudioSealChecker.audioseal_checker(audio_segment_id, None, compressed_bit_rate, None,"bit_rate")
