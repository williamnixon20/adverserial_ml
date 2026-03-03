from audioseal import AudioSeal
import soundfile as sf
import torch
import json




def audioseal_checker(audio_segment_id: str, compressed_sample_rate: str|None, compressed_bit_rate: int|None, playback_speed: float|None, perturbation_mode: str):

    model = AudioSeal.load_generator("audioseal_wm_16bits")
    model.eval()


    detector = AudioSeal.load_detector("audioseal_detector_16bits")

    if perturbation_mode == "sample_rate":
        compressed_audio, compressed_sr = sf.read(f'watermarked_sample_rate_compressed_audio/{audio_segment_id}_watermarked_compressed_{compressed_sample_rate}.mp3')
        compressed_waveform = torch.from_numpy(compressed_audio).float()


        compressed_waveform = compressed_waveform.unsqueeze(0).unsqueeze(0)


        result, message = detector.detect_watermark(compressed_waveform)
        
        data = {"id": audio_segment_id+"_"+perturbation_mode, "perturbation_mode": perturbation_mode, "detection_confidence": result.item()}
        with open("checker_result.json", "a") as file:
            file.write(json.dumps(data) + '\n')

    elif perturbation_mode == "bit_rate":
        compressed_audio, compressed_sr = sf.read(f'watermarked_bit_rate_compressed_audio/{audio_segment_id}_watermarked_compressed_{compressed_bit_rate}.mp3')
        compressed_waveform = torch.from_numpy(compressed_audio).float()


        compressed_waveform = compressed_waveform.unsqueeze(0).unsqueeze(0)


        result, message = detector.detect_watermark(compressed_waveform)
        
        data = {"id": audio_segment_id+"_"+perturbation_mode, "perturbation_mode": perturbation_mode, "detection_confidence": result.item()}
        with open("checker_result.json", "a") as file:
            file.write(json.dumps(data) + '\n')       

    elif perturbation_mode == "speed_no_pitch":
        compressed_audio, compressed_sr = sf.read(f'watermarked_speed_no_pitch_compressed_audio/{audio_segment_id}_watermarked_speed_{int(playback_speed*100)}.mp3')
        compressed_waveform = torch.from_numpy(compressed_audio).float()


        compressed_waveform = compressed_waveform.unsqueeze(0).unsqueeze(0)


        result, message = detector.detect_watermark(compressed_waveform)
        
        data = {"id": audio_segment_id+"_"+perturbation_mode+"_"+str(int(playback_speed*100)), "perturbation_mode": perturbation_mode, "detection_confidence": result.item()}
        with open("checker_result.json", "a") as file:
            file.write(json.dumps(data) + '\n')     

    elif perturbation_mode == "speed_with_pitch":
        compressed_audio, compressed_sr = sf.read(f'watermarked_speed_compressed_audio/{audio_segment_id}_watermarked_speed_{int(playback_speed*100)}.mp3')
        compressed_waveform = torch.from_numpy(compressed_audio).float()


        compressed_waveform = compressed_waveform.unsqueeze(0).unsqueeze(0)


        result, message = detector.detect_watermark(compressed_waveform)
        
        data = {"id": audio_segment_id+"_"+perturbation_mode+"_"+str(int(playback_speed*100)), "perturbation_mode": perturbation_mode, "detection_confidence": result.item()}
        with open("checker_result.json", "a") as file:
            file.write(json.dumps(data) + '\n')   