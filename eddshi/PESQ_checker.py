from pesq import pesq
import soundfile as sf
import os
import torch
import json
import sys

def compute_pesq(og_folder_name, adv_folder_name, perb_mode, jason_filename, sample_rate=16000):
    """
    x_ref, x_adv: [1,1,T] torch tensors
    returns float PESQ score
    """
    adv_audio_list = os.listdir(adv_folder_name)
    og_audio_list = os.listdir(og_folder_name)

    for i in range(len(adv_audio_list)):
        adv_audio, _ = sf.read(f"{adv_folder_name}/{adv_audio_list[i]}", frames = 40* sf.info(f"{adv_folder_name}/{adv_audio_list[i]}").samplerate) #only first 40 seconds due to limitation of pesq
        og_audio, _ = sf.read(f"{og_folder_name}/{og_audio_list[i]}", frames= 40* sf.info(f"{og_folder_name}/{og_audio_list[i]}").samplerate )

        mode = "wb" if sample_rate >= 16000 else "nb"
        
        data = {"audio": f"{adv_folder_name}/{adv_audio_list[i]}", "PESQ Score": pesq(sample_rate, og_audio, adv_audio, mode), "mode": perb_mode}

        with open(f"{jason_filename}", "a") as file:
            file.write(json.dumps(data) + ',\n') 

if __name__ == "__main__":

    sys.argv
    compute_pesq(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

