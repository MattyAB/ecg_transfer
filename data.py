import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import pywt
import numpy as np
import random
from utils import *


class WindowDataset(Dataset):
    def __init__(self, data, labelmap, threshold_length=1000, device='cpu', eval=False, trim=None, trim_samples=None):
        def normalise_1d(x):
            assert x.ndim == 1

            if x.std() == 0:
                return x

            return (x - x.mean()) / x.std()

        self.device=device
        self.eval = eval
        self.threshold_length = threshold_length

        if trim:
            assert False

        self.lengths = []

        self.data = []

        self.rpeaks = []

        for i, (waveform, label) in tqdm(enumerate(data)):
            ## Z-Score Normalisation https://pdf.sciencedirectassets.com/273545/1-s2.0-S1746809419X00099/1-s2.0-S1746809419304008/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIBU0QAQMTYgfyUZnLwfd%2FvB%2F4nxRCm3RuLZtKZ7fWyjcAiAKU%2FRDCwHN4XPuvRsXqcPZzUlACrRZM82gpaiyEee74yqyBQhIEAUaDDA1OTAwMzU0Njg2NSIMx5ynckvJ3tNuWl%2FlKo8FhqrYIJYO1JDS6KYXvVVFHfcov1wUhIa4p5Aa0ULGmrhMhFOKB%2Bn3I6TYKt9UVSxDucTQM7%2FCxwh904JgSmuL1nPCVN1xxMQnCUdR1UIxVJLQQ9FwJoM%2FfTS2efo3QTIh%2BnCaqVrYICqL7Bx1pGYSl975A%2BZVrSGi2tzROPTS%2BmQx9ORlf6Tv24X1hElEpdVy%2BgbHkNrZDILTwkbfs%2FbYXIh9fO0a2jyIwpAtrUxF%2Bd%2F6%2FulVUUsY6ZR6dIObMGwqlH22Agv9nOcgMTINELGC%2BqsWUPw%2FuC1cebaTT6bI32wKSLx2w6dq%2BiHT88sYW8V74MH1GaswjpeE%2FoxEEq6WiJjJuYI9qV3SOcbRMaYJyu69EM5zkoMaKknA7NCmXPIrURD6HEX7wgQXlHrBOieXJqF%2FnsFzp2Lp19oSqcozuNIIcAEPnpKDEHppA%2BMrc2BYfiGEwVropc1Cb3mc444uFzJH7I47q39vgzzQ%2BpGSc5iYyOfGXDWbPay4qWvewT2cSI%2FF0hN7kTrehuwjQHW%2Fidfv4TO27hFTMKLA7ZDQ%2BCoSkk0f8%2F3%2FQcMSbgp50q2BdiVjUJZzgkZvDFNsrB3RzpcjNRGFV3Qb7M8hKw1w4BoUoATLLu3awMFAhDAAmNfAnGiTb301svzkZUSZ5OyQhRL1igD4pNXQPmqsbV7%2FbpdaC5TeBYz54DsEIGqO%2Fig1NA6cHql4rQyI05oHk0szh0wXaaZOOfihuUqooxdJjUfmDRZiUUochgrvVC0EGQOCe9JbyAMSbqVPqNfq1csg16UOhF%2F3RO6faYCt08EK6pi4uFDmkakv42RR09DQvDNS%2FA0vNQX%2B%2F2z2Dr%2B4bgVOoJnHAkcngZeHYwNf3aZFJDD4i7qtBjqyAW3kY3S%2BP%2BkrLVg%2FPNf4t4YTDK9EE2RkbVaIQxTcoUMqpEZEXg7ny8GL3B9MIblfUE%2Bvx3PFd6rwdB%2F6uuWTgyW82XsWvsKH2LB44WiID%2B1tHF2W7f3lf182%2Bn5AVg8qa8mRCjG6MvcKiIxD8lOsH%2BlyHsLN52k0%2BUbp1XceM15R0FHbd7lv8WjWrJbBYrUNMuTg%2B005K8siLNk8k8Oftn3aKmYPWWseHQXjhOrVFaL78%2Bk%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240122T164545Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYZIIXIYUV%2F20240122%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=4fbb80e266fb73d5a1b1e5faf281b2b7e46e5ec6c4427eecddee0780a75135de&hash=7c3b44aa650c599596e1a6205d0816322ae68a2826439052209fe4e9f926e465&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1746809419304008&tid=spdf-a7d33034-bd26-4c4a-a4d7-bcd82730d377&sid=6bb7a7de95a0754e720b928565e5b1a3a850gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=1d025a565f5e50550556&rr=849944e7ad486340&cc=gb
            
            if waveform.ndim == 1:
                waveform = normalise_1d(waveform)
                
                waveform,b = pywt.dwt(waveform, 'db6')

                if trim_samples:
                    waveform = waveform[:trim_samples]

            elif waveform.ndim == 2:
                waves = []

                for j in range(waveform.shape[1]):
                    waveform[:,j] = normalise_1d(waveform[:,j])

                    waves.append(pywt.dwt(waveform[:,j], 'db6')[0])

                waveform = np.stack(waves, axis=1)


                if trim_samples:
                    waveform = waveform[:trim_samples,:]
            
            self.lengths.append(waveform.shape[0])

            self.rpeaks.append(read_rpeaks(waveform, str(i), fs=150))
            self.data.append((torch.tensor(waveform, dtype=torch.float32, device=device), torch.tensor(labelmap[label], device=device)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, return_rpeaks=False):
        waveform, label = self.data[idx]

        if waveform.shape[0] < self.threshold_length:
            if waveform.ndim == 1:
                blank_waveform = torch.zeros((self.threshold_length - waveform.shape[0]), device=self.device)
            elif waveform.ndim == 2:
                blank_waveform = torch.zeros((self.threshold_length - waveform.shape[0], 12), device=self.device)
            waveform = torch.cat((blank_waveform, waveform))
        else:
            startidx = random.randint(0, len(waveform) - self.threshold_length)
            waveform = waveform[startidx:startidx+self.threshold_length]


        if return_rpeaks:
            rpeaks = self.rpeaks[idx]

            if startidx:
                rpeaks = [x for x in rpeaks if x >= startidx and x < startidx + self.threshold_length]
                rpeaks = [x - startidx for x in rpeaks] 
            
            return waveform, label, rpeaks
        else:
            return waveform, label
    
    def get_value_counts(self, count=4):
        counts = [0] * count

        for _,label in self.data:
            counts[label.item()] += 1

        return counts
    
class RandomAugment():
    def __init__(self, device):
        if random.uniform(0,1) > 1.0: # I don't think doing flipx is correct at the moment.
            self.flipx = True
        else:
            self.flipx = False
        if random.uniform(0,1) > 0.5:
            self.flipy = True
        else:
            self.flipy = False

        ## Todo: Replace with a perlin map?
        gm = [0]
        for i in range(79):
            gm.append(random.normalvariate(gm[-1]*0.8, 0.5))

        self.gaussian_map = torch.tensor(gm, dtype=torch.float32, device=device)

    def __call__(self, waveform, rpeaks):
        waveform = waveform.clone()

        if self.flipx:
            waveform = torch.flip(waveform, (0,))
        if self.flipy:
            waveform = -1 * waveform

        for x in rpeaks:
            if x >= waveform.shape[0]:
                continue
            start_index = max(x - 40, 0)  # Adjusting to avoid negative index
            end_index = min(x + 40, waveform.size(0))  # Adjusting to avoid index out of bounds

            # Adjusting smaller tensor if necessary
            if start_index == 0:
                # If the operation starts from the very beginning of larger_tensor
                adjusted_smaller_tensor = self.gaussian_map[40-x:]
            elif end_index == waveform.size(0):
                # If the operation ends at the very end of larger_tensor
                adjusted_smaller_tensor = self.gaussian_map[:40 + waveform.size(0) - x]
            else:
                adjusted_smaller_tensor = self.gaussian_map

            # Perform the addition
            waveform[start_index:end_index] += adjusted_smaller_tensor[:end_index-start_index]

        return waveform


class AugmentDataset(WindowDataset):
    def __init__(self, data, labelmap, threshold_length=1000, device='cpu', eval=False, trim=None, trim_samples=None):
        assert len(data[0][0].shape) == 1 ## We don't want 2d data here!
        
        super().__init__(data, labelmap, threshold_length=threshold_length, device=device, eval=eval, trim=trim, trim_samples=trim_samples)

        self.augments = []
        for i in range(12):
            self.augments.append(RandomAugment(device))

    def __getitem__(self, idx):
        waveform, label, rpeaks = super().__getitem__(idx, return_rpeaks=True)
        
        waveforms = []

        for i in range(12):
            waveforms.append(self.augments[i](waveform, rpeaks))

        return torch.stack(waveforms, dim=1), label
    
    def get_value_counts(self, count=4):
        counts = [0] * 4

        for _,label in self.data:
            counts[label.item()] += 1

        return counts