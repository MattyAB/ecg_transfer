import scipy.io
import csv

from tqdm import tqdm

import wfdb.processing
import wfdb
import torch
import os
import pickle
import numpy as np
import random

import matplotlib.pyplot as plt

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
        waveform = scipy.io.loadmat(afc_root+data_path+id+'.mat')['val'].reshape((-1))

        data.append((waveform, label))#, id))

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
    
    returner = []

    for id in ids:
        samples = []

        sample = wfdb.rdsamp(dir + str(id))
        annotation = wfdb.rdann(dir + str(id), 'atr')

        # if 'N' not in annotation.aux_note[0]:
        #     print(annotation.aux_note)

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

        returner.append(samples)

    return returner

data_12lead_path = './af_challenge_2020/classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2/'

ecg_diagnoses = {
    "164889003": "A",  # atrial fibrillation
    "195080001": "A",  # atrial fibrillation and flutter
    "426749004": "A",  # chronic atrial fibrillation
    "282825002": "A",  # paroxysmal atrial fibrillation
    "314208002": "A",  # rapid atrial fibrillation
    "426783006": "N",  # sinus rhythm
}

def resample(xs, fs, fs_target):
    lx = []
    for chan in range(xs.shape[1]):
        resampled_x, _ = wfdb.processing.resample_sig(xs[:, chan], fs, fs_target)
        lx.append(resampled_x)

    return np.column_stack(lx)

## Level - 0: 1000 samples from georgia, 1: all samples from georgia, 2: all samples
def import_12lead_data(target_freq=300, level=1):
    # Dx_map = pd.read_csv(data_12lead_path + 'Dx_map.csv')
    # Dx_map = Dx_map.set_index('SNOMED CT Code')['Abbreviation'].to_dict()
    # print(Dx_map)

    samples = []

    with open(data_12lead_path + 'RECORDS', 'r') as f:
        for line in f.readlines():
            dir = data_12lead_path + line.rstrip('\n')

            if ('cpsc_2018' not in dir) and level <= 1:
                continue

            if not os.path.exists(dir):
                continue

            for filename in tqdm(os.listdir(dir)):
                if filename[-3:] == 'mat':
                    sample = wfdb.rdsamp(dir + '/' + filename[:-4])

                    diagnoses = [x[4:] for x in sample[1]['comments'] if 'Dx: ' in x]
                    assert len(diagnoses) == 1

                    D = ''
                    if diagnoses[0] in ecg_diagnoses:
                        D = ecg_diagnoses[diagnoses[0]]
                    else:
                        D = 'O'
                        

                    waveform = resample(sample[0], sample[1]['fs'], target_freq)

                    samples.append((waveform, D)) 
            
            if level == 0:
                break

    return samples

def import_balanced_12lead_data():
    pick_path = './af_challenge_2020/balanced.pk'

    if os.path.exists(pick_path):
        with open(pick_path, 'rb') as file:
            return pickle.load(file)
    else:
        data = import_12lead_data(level=2)

        unique_categories = list(set([x[1] for x in data]))

        dictmap = {cat:[x for x in data if x[1] == cat] for cat in unique_categories}

        trim_value = min([len(ls) for ls in dictmap.values()])

        output_data = []
        for ls in dictmap.values():
            # output_data += ls[:trim_value]
            output_data += random.sample(ls, trim_value)

        with open(pick_path, 'wb') as file:
            pickle.dump(output_data, file)

        return output_data
    


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


### PLOT

def plot_tt_graph(history, idx=0):
    fig, ax1 = plt.subplots()

    # Plotting 'loss' on the left y-axis
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot([x for x in history['train_loss'][idx]], label='Train Loss', color='tab:red')
    ax1.plot([x for x in history['test_loss'][idx]], label='Test Loss', color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.legend(loc='upper left')

    # Creating a second y-axis for 'accuracy'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:blue')
    ax2.plot([x for x in history['train_acc'][idx]], label='Train Accuracy', color='tab:blue')
    ax2.plot([x for x in history['test_acc'][idx]], label='Test Accuracy', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.legend(loc='upper right')

    # Show the plot
    plt.show()


def display_results(history, trainparams):
    def mean(x):
        return sum(x) / len(x)

    if trainparams.m:
        print(f'Overall results of {trainparams.k} fold cross-validation with leave-{trainparams.m}-out')
    else:
        print(f'Overall results of {trainparams.k} fold cross-validation')
    print(f'Train: Average loss {mean([x[-1] for x in history["train_loss"]])}, average accuracy {mean([max(x) for x in history["train_acc"]]) * 100}')
    print(f'Test: Average loss {mean([x[-1] for x in history["test_loss"]])}, average accuracy {mean([max(x) for x in history["test_acc"]]) * 100}')