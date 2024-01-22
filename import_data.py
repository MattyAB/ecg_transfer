import scipy.io
import sys
import csv

from tqdm import tqdm

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

        data.append((waveform, label))

    return data