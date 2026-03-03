
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # must be before any torch import

from pydub import AudioSegment
import os
from gtts import gTTS



os.add_dll_directory("C:\\ffmpeg\\bin") #fix the torchaudio issue
os.makedirs("elevenlab_bit_rate_compressed_audio", exist_ok=True)
####generate text-to-speech####



audio_list = os.listdir('elevenlab_generated_original')

for a in audio_list:


    
    audio_name = a.replace('.mp3','')
   
    compressed_bit_rate = "64k"

    AudioSegment.from_mp3(f'elevenlab_generated_original/{a}').export(
        f'elevenlab_bit_rate_compressed_audio/{audio_name}_watermarked_compressed_{compressed_bit_rate}.mp3 ', 
        format='mp3', 
        bitrate=f'{compressed_bit_rate}'
    )
