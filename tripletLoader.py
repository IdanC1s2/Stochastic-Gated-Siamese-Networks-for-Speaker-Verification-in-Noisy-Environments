from torch.utils.data.sampler import Sampler
import numpy as np
import os
from pathlib import Path

import torchaudio
from torch.utils.data import Dataset


#
# class LibriSpeechVerificationDataset(Dataset):
#     fs = 16000
#
#     def __init__(self, sample_time, subset, transform, root='data', p=8, k=4):
#         """
#         c'tor for LibriSpeech dataset for speaker verification
#         :param sample_time: audio sample time in seconds
#         :param subset: "dev-clean", "dev-other", "test-clean", "test-other",
#                        "train-clean-100", "train-clean-360", "train-other-500"
#         """
#         self.librispeech_dataset = datasets.librispeech.LIBRISPEECH(root=root, url=subset, download=True)
#         self.sample_time = sample_time
#         self.base_path = f'{root}/LibriSpeech/{subset}'
#         self.speakers = [item for item in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, item))]
#
#         self.label_to_samples = {}
#         speakers_vec = np.array(np.array(self.librispeech_dataset)[:, 3], dtype=int)
#         for speaker in self.speakers:
#             indices = np.where(speakers_vec == int(speaker))
#             self.label_to_samples[speaker] = indices
#
#         self.transform = transform
#         self.p = p
#         self.k = k
#
#     def __len__(self):
#         return len(self.librispeech_dataset)
#
#     def __getitem__(self, idx):
#         """
#         return sample_time length audio samples from the dataset
#         """
#         if self.__sample_length(idx) < self.sample_time * self.fs:  # If our new sample is shorter than what we want
#             padded_audio = F.pad(self.librispeech_dataset[idx][0],
#                                  (1, self.sample_time * self.fs - self.__sample_length(idx)),
#                                  "replicate").float()
#             if padded_audio.shape[1] == self.sample_time * self.fs + 1:
#                 padded_audio = padded_audio[:, :-1]
#             return self.transform(padded_audio.squeeze(0)), self.librispeech_dataset[idx][3]
#         else:
#             truncated_audio = self.librispeech_dataset[idx][0][:, :self.sample_time * self.fs].float()
#             return self.transform(truncated_audio.squeeze(0)), self.librispeech_dataset[idx][3]
#
#     def __sample_length(self, idx):
#         return self.librispeech_dataset[idx][0].shape[1]



def load_librispeech_item(fileid: str, path: str, ext_audio: str):
    speaker_id, chapter_id, utterance_id = fileid.split("-")

    # Load audio
    fileid_audio = f"{speaker_id}-{chapter_id}-{utterance_id}"
    file_audio = fileid_audio + ext_audio
    file_audio = os.path.join(path, speaker_id, file_audio)
    waveform, sample_rate = torchaudio.load(file_audio)

    return (
        waveform,
        # sample_rate,
        int(speaker_id),
    )


class LibriSpeechVerificationDataset(Dataset):
    _ext_txt = ".trans.txt"
    _ext_audio = ".flac"

    def __init__(self, path, transform):
        """
        :param path: Path to the directory that contains speakers folders
        :param transform: Spectrogram/ Mel-spectrogram transform
        """

        self._path = path
        self.transform = transform

        self.speakers = [item for item in os.listdir(self._path) if
                         os.path.isdir(os.path.join(self._path, item))]

        self._walker = sorted(str(p.stem) for p in Path(self._path).glob("*/*" + self._ext_audio))

        # Create mapping between each label and its samples indices
        self.label_to_samples = {}
        speakers_vec = np.array(self.get_classes(slice(0,len(self._walker))), dtype=int)
        for speaker in self.speakers:
            indices = np.where(speakers_vec == int(speaker))
            self.label_to_samples[speaker] = indices

    def __getitem__(self, n: int):
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int,):
            ``(waveform, speaker_id)``
        """
        fileid = self._walker[n]
        waveform, speaker_id = load_librispeech_item(fileid, self._path, self._ext_audio)
        spec = self.transform(waveform)
        return (spec.squeeze(0), speaker_id)

    def __len__(self) -> int:
        return len(self._walker)

    def get_classes(self, slice):
        fileids = self._walker[slice]
        classes = []
        for id in fileids:
            speaker_id, _, _ = id.split("-")
            classes.append(speaker_id)
        return classes


class PKSampler(Sampler):
    def __init__(self, data_source, p=8, k=4):
        super().__init__(data_source)
        self.p = p
        self.k = k
        self.data_source = data_source

    def __iter__(self):
        pk_count = len(self) // (self.p * self.k)
        for _ in range(pk_count):
            labels = np.random.choice([int(x) for x in self.data_source.label_to_samples.keys()], self.p, replace=False)
            for l in labels:
                indices = self.data_source.label_to_samples[str(l)][0]
                replace = True if len(indices) < self.k else False
                for i in np.random.choice(indices, self.k, replace=replace):
                    yield i

    def __len__(self):
        pk = self.p * self.k
        samples = ((len(self.data_source) - 1) // pk + 1) * pk
        return samples
