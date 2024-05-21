import os
import einops
import torch.nn.functional as F
import random
import torch
import soundfile as sf


def save_audio(preds, save_path, string, sample_rate):
    file_name=save_path / (string+".wav")
    #save the audio
    preds=einops.rearrange(preds,'b c t -> (b c t)')
    sf.write(file_name, preds.squeeze().cpu().numpy(), sample_rate)



def crop_or_extend( audio, segment_length, pad_zeros=False, deterministic=False, start=None):
        if start is None:
            start=0
        if audio.size(0) >= segment_length:
            max_audio_start = audio.size(0) - segment_length
            if deterministic:
                if start is not None:
                    #minimum of max_audio_start and start
                    audio_start = min(max_audio_start, start) 
                else:
                    audio_start = 0
            else:
                audio_start = random.randint(0, max_audio_start)

            audio = audio[audio_start : audio_start + segment_length]
        elif audio.size(0) < segment_length:
            if pad_zeros:
                audio = F.pad(
                    audio, (0, segment_length - audio.size(0)), "constant"
                ).data
                assert audio.size(0) == segment_length, f"audio.size(0)={audio.size(0)} != segment_length={segment_length}"
            else:
                num_repeats = segment_length // audio.size(0) + 1
                audio = audio.repeat(num_repeats)
                #roll the audio to get a random start
                if not deterministic:
                    audio=torch.roll(audio, random.randint(0, audio.size(0)), 0)
                else:
                    audio=torch.roll(audio, 0, 0)

                audio = audio[:segment_length]

                assert audio.size(0) == segment_length, f"audio.size(0)={audio.size(0)} != segment_length={segment_length}"

        return audio


def apply_RIR_delay( x, H, delay):
        """
        Simply convolves the audio with the RIR, and corrects the delay
        """

        xinpshape=x.shape

        H_conv=H.unsqueeze(0).unsqueeze(0)
        #reverse IR
        H_conv=H_conv.flip(2)
        #convolve audio with RIR
        x_conv=F.pad(x, (H_conv.shape[2]-1-delay, delay), mode="constant", value=0)
        Y=torch.conv1d(x_conv, H_conv, padding=0) #zero padding is probably not the best option
        assert Y.shape == xinpshape, f"Y.shape={Y.shape} != x.shape={xinpshape}"

        return Y.float()


def calculate_curvature(trajectory):
    base=trajectory[0]-trajectory[-1]
    N=len(trajectory)
    dt=1.0/N
    mse=[]
    for i in range(1,N):
        v=(trajectory[i-1]-trajectory[i])/dt
        mse.append(torch.mean((v-base)**2).cpu())
    return torch.mean(torch.stack(mse)), mse


def get_checkpoint_file(checkpoint_arg, save_path):
    if checkpoint_arg!=None and checkpoint_arg!="None" and checkpoint_arg!="none" and checkpoint_arg!="":
        #trying checpoint_arg as absolute path
        print(checkpoint_arg)
        if os.path.exists(checkpoint_arg):
            return checkpoint_arg
        elif (save_path / checkpoint_arg).exists():
            return save_path / checkpoint_arg
        elif os.path.exists("../"+checkpoint_arg):
            return  ".." / checkpoint_arg 
        else:
            print("Could not find the checkpoint file", checkpoint_arg)

    print("Trying to load a checkpoint from the save path")
    checkpoint_files = list(save_path.glob("*.pt"))
    checkpoint_files = sorted(checkpoint_files, key=lambda x: x.stat().st_mtime)
    checkpoint_file = checkpoint_files[-1] if checkpoint_files else None
    return checkpoint_file
