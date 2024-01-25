import scipy.io
import csv

from tqdm import tqdm

import wfdb.processing
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



### R Peak Detection

def preprocess_1d_signal(signal, name, device=torch.device('cpu')):
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
        if rpeak > 100 and rpeak < signal.shape[0] - 600:
            datapoints.append(torch.tensor(signal[rpeak-100:rpeak+600], dtype=torch.float32, device=device))

    return datapoints