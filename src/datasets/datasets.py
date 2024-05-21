import torch
import os
import torch.utils.data
import torch.nn.functional as F
import soundfile as sf
import numpy as np
import pandas as pd
import random
from glob import glob
from utils.utils import crop_or_extend, apply_RIR_delay


class AudioDatasetInfinite_VCTK_Reverb(torch.utils.data.IterableDataset):
    """
    Infinite random audio data sampler.
    """

    def __init__(self,
                 path=None,
                 segment_length=65536,
                 fs=16000,
                 sigma_data=0.05,
                 path_RIR=None,
                 RIR_file=None,
                 normalize_RIR=True,
                 T60_margin=[0, 1],
                 T60_uniform=True,
                 T60_divisions=100,
                 h_length=16000,
                 apply_random_gain=True,
                 random_gain=[-2, 2],
                 percentage_anechoic=10,
                 ):

        self.sampling_rate = fs
        self.segment_length = segment_length

        self.audio_files = glob(os.path.join(path, "*.flac"))
        self.audio_files.extend(glob(os.path.join(path, "*.wav")))
        self.path_RIR = path_RIR
        self.RIR = pd.read_csv(RIR_file, sep=",")
        self.RIR_train = self.RIR[self.RIR["split"] == "train"]

        self.num_files = len(self.audio_files)
        # what to do when the sample is shorter than the segment length
        self.sigma_data = sigma_data
        self.normalize_RIR = normalize_RIR
        self.h_length = h_length
        self.T60_uniform = T60_uniform

        if T60_uniform:

            assert len(
                T60_margin) == 2, "T60_margin must be a list of two elements"
            assert T60_margin[0] < T60_margin[1], "T60_margin must be a list of two elements"
            assert T60_divisions > 0, "T60_divisions must be a positive integer"
            self.T60_margin = T60_margin
            self.T60_divisions = T60_divisions

            self.ranges = [self.T60_margin[0]+i*(self.T60_margin[1]-self.T60_margin[0]) /
                           self.T60_divisions for i in range(self.T60_divisions+1)]

            self.RIR_divs = [self.RIR_train[(self.RIR_train["T60_WB"] >= self.ranges[i]) & (
                self.RIR_train["T60_WB"] < self.ranges[i+1])] for i in range(self.T60_divisions)]

        # assert that all self.RIR_divs have at least one element
        for i in range(self.T60_divisions):
            assert len(
                self.RIR_divs[i]) > 0, f"no RIRs found for T60 in range {self.ranges[i]}-{self.ranges[i+1]}"

        self.apply_random_gain = apply_random_gain
        if self.apply_random_gain:
            assert random_gain[0] < random_gain[1], "random_gain must be a list of two elements"
            self.random_gain = random_gain

        self.percentage_anechoic = percentage_anechoic

    def get_IR(self):
        # randomly choose anechoic or reverberant
        if random.randint(0, 100) < self.percentage_anechoic:
            # anechoic
            T60 = 0
            C50 = 50
            h = torch.zeros((self.h_length,))
            h[0] = 1
            h = h.float()
            delay = 0
        else:
            i = random.randint(0, self.T60_divisions-1)

            try:
                rir = self.RIR_divs[i].sample(1)
            except:
                print("A", i, len(self.RIR_divs), len(self.ranges))

            T60 = rir["T60_WB"].values[0]
            C50 = rir["C50_WB"].values[0]
            delay = rir["delay_in_samples"].values[0]
            try:
                assert T60 >= self.ranges[i] and T60 < self.ranges[i +1], f"T60={T60} not in range {self.ranges[i]}-{self.ranges[i+1]}"
            except:
                print("B", i)

            h, fs = sf.read(os.path.join(
                self.path_RIR, rir["relative_path"].values[0]))
            h = torch.from_numpy(h).float()
            h /= torch.sqrt(torch.sum(h**2))

            assert fs == self.sampling_rate, f"fs={fs} != self.sampling_rate={self.sampling_rate}"
            if h.shape[-1] < self.h_length:
                # pad with zeros
                h = F.pad(
                    h, (0, self.h_length - h.size(-1)), "constant"
                ).data
            elif h.shape[-1] > self.h_length:
                # cut
                h = h[:self.h_length]
        return h, T60, C50, delay

    def __iter__(self):
        while True:
            index = random.randint(0, self.num_files-1)
            # Read audio
            filename = self.audio_files[index]
            audio, sampling_rate = load_wav_to_torch(filename)
            # we assume that the dataset is already at the correct sampling rate, please preprocess it accordingly
            assert sampling_rate == self.sampling_rate, f"sampling_rate={sampling_rate} != self.sampling_rate={self.sampling_rate}"

            # Take segment
            audio = crop_or_extend(audio, self.segment_length)

            h, T60, C50, delay = self.get_IR()

            audio = audio / self.sigma_data
            if self.apply_random_gain:
                audio = audio * \
                    10**(np.random.uniform(
                        low=self.random_gain[0], high=self.random_gain[1])/20)

            audio = audio.float()
            # convert numpy arrays to torch tensors
            h = h.float()
            T60 = torch.Tensor([T60]).float()
            C50 = torch.from_numpy(np.array([C50])).float()
            delay = torch.from_numpy(np.array([delay])).float()

            x = apply_RIR_delay(audio.unsqueeze(0), h, int(delay))

            yield x, T60, C50


class AudioDatasetInfinite_VCTK_clip(torch.utils.data.IterableDataset):
    """
    Infinite random audio data sampler.
    """

    def __init__(self,
                 path="/home/t-ejuanpere/blob/datasets/VCTK_16kHz",
                 segment_length=65536,
                 fs=16000,
                 sigma_data=0.05,
                 apply_random_gain=True,
                 random_gain=[-2, 2],
                 percentage_clean=50,
                 num_resampling_methods=5,
                 gain_range=[5, 30]  # defined in dB arbitrarily
                 ):

        self.sampling_rate = fs
        self.segment_length = segment_length
        self.audio_files = glob(os.path.join(path, "*.flac"))
        self.audio_files.extend(glob(os.path.join(path, "*.wav")))

        self.num_files = len(self.audio_files)
        # what to do when the sample is shorter than the segment length
        self.sigma_data = sigma_data

        self.apply_random_gain = apply_random_gain
        if self.apply_random_gain:
            assert random_gain[0] < random_gain[1], "random_gain must be a list of two elements"
            self.random_gain = random_gain

        self.percentage_clean = percentage_clean
        self.num_resampling_methods = num_resampling_methods

        self.gain_range = gain_range

    def hard_clip_audio(self, x):
        # randomly choose anechoic or reverberant
        if random.randint(0, 100) < self.percentage_clean:
            y = x
            SDR = 50
            SDR = torch.tensor(SDR)
        else:
            gain_db = random.uniform(self.gain_range[0], self.gain_range[1])
            gain_lin = 10**(gain_db/20)

            y = torch.clamp(x*gain_lin, min=-1, max=1)
            y /= gain_lin

            SDR = 10*torch.log10(torch.sum(x**2)/torch.sum((x-y)**2))
            SDR = torch.clamp(SDR, min=0, max=50)

        return y, SDR

    def __iter__(self):
        while True:
            index = random.randint(0, self.num_files-1)
            # Read audio
            filename = self.audio_files[index]
            audio, sampling_rate = load_wav_to_torch(filename)
            assert sampling_rate == self.sampling_rate, f"sampling_rate={sampling_rate} != self.sampling_rate={self.sampling_rate}"

            # Take segment
            audio = crop_or_extend(audio, self.segment_length)

            audio, SDR = self.hard_clip_audio(audio)

            audio = audio/audio.std()  # applying standardization here

            if self.apply_random_gain:
                audio = audio * \
                    10**(np.random.uniform(
                        low=self.random_gain[0], high=self.random_gain[1])/20)
            audio = audio.float()

            yield audio.unsqueeze(0), SDR.float()


def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    data, sampling_rate = sf.read(full_path)

    return torch.from_numpy(data).float(), sampling_rate
