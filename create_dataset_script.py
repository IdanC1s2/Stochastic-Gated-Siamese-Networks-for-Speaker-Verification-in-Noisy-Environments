import random
import shutil
import numpy as np
import soundfile as sf
import os
from torch.utils.data import Dataset
import scipy.signal as signal
from tqdm import tqdm


class LibriSpeechTripletLoader(Dataset):
    def __init__(self, base_path, transform=None):
        self.base_path = base_path
        self.speakers = [item for item in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, item))]
        self.transform = transform

    def __getitem__(self, item):
        anchor_waveform = 0

def load_noises(noises_folder):
    # noises_folder = '/home/dsi/idancohen/Unsupervised_Learning/Datasets/Noises/'
    noises = {'babble': list(sf.read(noises_folder + "Babble.wav")),
              'car': list(sf.read(noises_folder + "Car.wav")),
              'factory': list(sf.read(noises_folder + "Factory.wav")),
              'room': list(sf.read(noises_folder + "Room.wav"))}
    for noise_type in noises.keys():
        noise = noises[noise_type]
        if noise[1] != 16000:
            noise[0] = signal.resample(noise[0], int(len(noise[0]) * 16000 / noise[1]))
            noise[1] = 16000
    z = np.random.randn(1000 * 16000)
    noises['white'] = list((np.random.randn(1000 * 16000), 16000))
    noises['none'] = list((np.zeros(1000 * 16000, dtype='float64'), 16000))
    return noises


def add_noise(waveform, snr, noises_dict, noise_type='none'):
    try:
        noise = noises_dict[noise_type][0]
    except KeyError as err:
        raise KeyError('Noise type {0} is invalid '.format(err))


    waveform_length = len(waveform)
    noise_length = len(noise)
    # noise_length - waveform_length is the maximal index we can choose:
    m = np.random.randint(0,noise_length - waveform_length)
    noise = noise[m:m + waveform_length]

    snr = 10**(snr/10)
    E_s = np.sum(waveform**2)
    E_n = np.sum(noise**2)
    if noise_type == 'none':
        sigma = 0
    else:
        sigma = np.sqrt(E_s/(snr*E_n))
    return waveform + sigma * noise


def create_dataset(librispeech_path, dst_path, ids, n_samples, sample_length):
    """
    This function is used to create either the validation dataset or training dataset, given the ids of
    the training or validation speakers.
    We save the first [sample_length] seconds of [n_samples] recordings for each speaker.

    :param librispeech_path: Path to the 'train-clean-100' folder of the Librispeech dataset
    :param dst_path: Path to the folder of either 'training'  or 'val' that we created
    :param ids: ids of either the training or validation speakers
    :param n_samples: number of samples we wish to use for each speaker
    :param sample_length: Desired length (in seconds) of each sample
    :param noise_dict: Dictionary containing the noises. Use load_noises() to create it
    :param snr: SNR value for the dataset
    :param noise_type: Type of noise to be added - possibilites are: 'babble', 'car', 'factory', 'room', 'white', 'none'
    :return:
    """
    for i in tqdm(ids):
        dst_path_speaker = dst_path + '/' + i
        os.mkdir(dst_path_speaker)
        speaker_folder = librispeech_path + i
        chapters = os.listdir(speaker_folder)
        chapter = random.choice(chapters)
        recordings_dir = speaker_folder + '/' + chapter
        recordings = os.listdir(recordings_dir)
        recordings = [rec for rec in recordings if rec[-3:] != 'txt']
        samples = random.sample(recordings, np.min([n_samples, len(recordings)]))
        for sample in samples:
            data, fs = sf.read(recordings_dir + '/' + sample)
            if fs != 16000:
                raise ValueError('Sampling frequency should be 16kHz')  # make sure
            # If our new sample is shorter than what we want - we pad
            if data.shape[0] < sample_length * fs:
                data = np.pad(data, (0, sample_length * fs - data.shape[0]), 'wrap')
            else:
                data = data[:sample_length * fs]
            sf.write(dst_path_speaker + '/' + sample, data, fs)

    print(f'Dataset was created')
    return


def main():
    # TODO - Change Parameters According to Your Need
    # Path to Librispeech train-clean-100-folder
    Librispeech_path = './LibriSpeech/train-clean-100/'
    # Path to noises folder:
    noises_folder = './noise/'
    # Destination path for created dataset:
    dataset_dst_path = './dataset/'
    # Noise type:
    noise_type = 'none'  # possible types:  'white', 'room, 'factory', 'car', 'babble', 'none'
    # Audio file length:
    sample_length = 3  # seconds
    # Number of samples for each speaker:
    n_samples = 10

    # Create training and validation folders:
    for folder in ['train', 'val']:
        try:
            dst_folder = dataset_dst_path + '/' + folder
            os.mkdir(dst_folder)
            print(f'directory \'{folder}\' was created')
        except:
            # os.rmdir(train_dst_folder)
            shutil.rmtree(dst_folder)
            os.mkdir(dst_folder)
            print(f'directory \'{folder}\' was removed and created again.')

    train_dst_folder = dataset_dst_path + 'train'
    val_dst_folder = dataset_dst_path + 'val'

    speaker_id_list = os.listdir(Librispeech_path)
    N = len(speaker_id_list)
    indices = np.random.permutation(N)
    # Train-Val ratio: 0.9
    indices_train, indices_val = indices[:N * 9 // 10], indices[N * 9 // 10:]
    # A stupid workaround for list indexing that is impossible otherwise:
    training_ids = []
    for i in indices_train:
        training_ids.append(speaker_id_list[i])

    validation_ids = []
    for i in indices_val:
        validation_ids.append(speaker_id_list[i])

    # dst_path = train_dst_folder or val_dst_folder
    # ids = training_ids or validation_ids

    noises_dict = load_noises(noises_folder)

    create_dataset(Librispeech_path, dst_path=val_dst_folder, ids=validation_ids, n_samples=n_samples,
                   sample_length=sample_length)
    create_dataset(Librispeech_path, dst_path=train_dst_folder, ids=training_ids, n_samples=n_samples,
                   sample_length=sample_length)


if __name__ == '__main__':
    main()
