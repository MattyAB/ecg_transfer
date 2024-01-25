import scipy.io
import csv

from tqdm import tqdm

import wfdb.processing
import wfdb
import torch
import os

### Data Import

afc_root = './af_challenge_2017/'
reference_file = 'REFERENCE-v3.csv'
data_path = 'training2017/'

def import_csv_2col_dict(filename):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        return {rows[0]: rows[1] for rows in reader}

def import_afc_data():
    labels_dict = import_csv_2col_dict(afc_root+reference_file)

    data = []

    for id,label in tqdm(labels_dict.items()):
        waveform = scipy.io.loadmat(afc_root+data_path+id+'.mat')['val']

        data.append((waveform, label, id))

    return data

mit_label_remap = {'/': 'N', ## Paced Beat - Normal
    'f': 'N', ## Fusion of paced an normal beat
    'Q': 'N', 
    'N': 'N', ## Normal Beat
    '+': 'N',
    '~': 'N',
    'A': 'P', ## APB
    'V': 'V', ## PVC
    'a': 'P',
    'F': 'N',
    '|': 'N',
    'j': 'N',
    'e': 'N',
    'R': 'R', ## RBBB
    'L': 'L', ## LBBB
    'J': 'P', ## APB
    'S': 'P',
    '!': 'N',
    '[': 'N',
    ']': 'N',
    'E': 'N',
    '"': 'N',
'x': 'P'}

def import_mit_data(lb=100, ub=150):
    ids = set()  # Using a set to store unique values
    dir = './mit-bih-arrhythmia-database-1.0.0/'

    # Loop through all files in the directory
    for filename in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, filename)):
            # Extract the first three characters
            prefix = filename[:3]
            try:
                int(prefix)
                ids.add(prefix)
            except:
                pass
    
    samples = []

    for id in ids:
        sample = wfdb.rdsamp(dir + str(id))
        annotation = wfdb.rdann(dir + str(id), 'atr')

        thissamp = []
        for idx,symbol in zip(annotation.sample, annotation.symbol):
            if len(thissamp) == 0:
                if idx > lb and idx < len(sample[0]) - ub:
                    thissamp.append((idx,mit_label_remap[symbol]))
            else:
                if thissamp[-1][1] == mit_label_remap[symbol]:
                    if idx > lb and idx < len(sample[0]) - ub:
                        thissamp.append((idx,mit_label_remap[symbol]))
                else:
                    samples.append((sample[0][thissamp[0][0]-lb:thissamp[-1][0]+ub,0], thissamp[0][1]))
                    thissamp = []

    return samples


### R Peak Detection

def preprocess_1d_signal(signal, name, device=torch.device('cpu'), lb=150, ub=200):
    filename = './af_challenge_2017/training2017/' + name + '.txt'

    if os.path.exists(filename):
        # Read rpeaks from the file
        with open(filename, 'r') as file:
            rpeaks = [int(line.strip()) for line in file]
    else:
        # Generate rpeaks using the provided function
        rpeaks = wfdb.processing.xqrs_detect(signal, fs=250, verbose=False)

        # Write rpeaks to the file
        with open(filename, 'w') as file:
            for rpeak in rpeaks:
                file.write(f"{rpeak}\n")

    datapoints = []

    for rpeak in rpeaks:
        if rpeak > lb and rpeak < signal.shape[0] - ub:
            segment = signal[rpeak-lb:rpeak+ub]
            segment = (segment - segment.mean()) / segment.std()
            datapoints.append(torch.tensor(segment, dtype=torch.float32, device=device))

    return datapoints