import torch
from torch import nn
import numpy as np
import soundfile as sf
import scipy.signal as signal


class TodB(nn.Module):
    def __init__(self):
        super(TodB, self).__init__()
        self.eps = 1e-5

    def forward(self, x):
        return 20 * torch.log10(x + self.eps)


class AddNoise(nn.Module):
    def __init__(self, noises_folder, snr, noise_type):
        """
        Add noise to audio recording.
        :param snr: Value in dB
        :param noise_type: 'None', 'White', 'Babble', 'Car', 'Factory', 'Room', 'Random'
        """
        super().__init__()
        self.noises = [list(sf.read(noises_folder + "Babble.wav")),
                       list(sf.read(noises_folder + "Car.wav")),
                       list(sf.read(noises_folder + "Factory.wav")),
                       list(sf.read(noises_folder + "Room.wav"))]
        self.noise_to_idx = {'Babble':  0,
                             'Car':     1,
                             'Factory': 2,
                             'Room':    3}
        for noise in self.noises:
            if noise[1] != 16000:
                noise[0] = signal.resample(noise[0], int(len(noise[0]) * 16000 / noise[1]))
                noise[1] = 16000
        self.snr = snr
        self.noise_type = noise_type

    def forward(self, waveform):
        sample_noise = False
        if self.noise_type == 'Random':
            noise = np.expand_dims(self.noises[np.random.randint(0, 3)][0], 0)
            sample_noise = True
        elif self.noise_type == 'None':
            return waveform
        elif self.noise_type == 'White':
            noise = torch.randn_like(waveform)
        else:
            noise = np.expand_dims(self.noises[self.noise_to_idx[self.noise_type]][0], 0)
            sample_noise = True

        if sample_noise:
            waveform_length = waveform.shape[1]
            noise_length = noise.shape[1]
            m = np.random.randint(0, noise_length - waveform_length)
            noise = torch.from_numpy(noise[:, m:m + waveform_length]).float()

        snr = 10 ** (self.snr / 10)
        E_s = torch.sum(waveform ** 2)
        E_n = torch.sum(noise ** 2)
        sigma = torch.sqrt(E_s / (snr * E_n))
        return waveform + sigma * noise


def pairwise_distance_torch(embeddings, device, dtype='float32'):
    """Computes the pairwise-distance-matrix
    """

    # pairwise distance matrix with precise embeddings
    if dtype == 'float32':
        precise_embeddings = embeddings.to(dtype=torch.float32)
    elif dtype == 'float64':
        precise_embeddings = embeddings.to(dtype=torch.float64)
    else:
        raise(ValueError)

    # Computing distance for every pair using
    # (a - b) ^ 2 = a^2 - 2ab + b^2
    ab = precise_embeddings.mm(precise_embeddings.t())
    a_squared = ab.diag().unsqueeze(1)
    b_squared = ab.diag().unsqueeze(0)
    pairwise_distances_sq = a_squared - 2 * ab + b_squared

    # Set small negatives to zero.
    pairwise_distances_sq = torch.max(pairwise_distances_sq, torch.tensor([0.]).to(device))
    # Get the mask where the zero distances are at.
    error_mask = pairwise_distances_sq.clone()
    error_mask[error_mask > 0.0] = 1.
    error_mask[error_mask <= 0.0] = 0.

    pairwise_distance_mat = torch.mul(pairwise_distances_sq, error_mask)

    # Set elements on the main diagonal to zero manually:
    mask_offdiagonals = torch.ones((pairwise_distance_mat.shape[0], pairwise_distance_mat.shape[1])) - torch.diag(torch.ones(pairwise_distance_mat.shape[0]))
    pairwise_distance_mat = torch.mul(pairwise_distance_mat.to(device), mask_offdiagonals.to(device))
    return pairwise_distance_mat

def TripletSemiHardLoss(labels, embeddings, device, margin=1.0):
    """Computes the triplet loss_functions with semi-hard negative mining.
    Given a minibatch, picks the hardest valid semi-hard triplets.
    Valid semi-hard triplets are triplets that satisfy: |D(a,p)| < |D(a,n)| < margin
    The hardest triplets are those where the negative is the closest to the anchor, and the positive
    is the farthest from the anchore.
    If no such negative exists, uses the largest negative distance instead.
    """
    # Reshape label tensor to [batch_size, 1].
    lshape = labels.shape
    labels = torch.reshape(labels, [lshape[0], 1])

    pdist_matrix = pairwise_distance_torch(embeddings, device)
    # Build pairwise binary adjacency matrix.
    adjacency = torch.eq(labels, labels.transpose(0, 1))
    # Invert so we can select negatives only.
    adjacency_not = adjacency.logical_not()

    batch_size = labels.shape[0]

    # Compute the mask.
    pdist_matrix_tile = pdist_matrix.repeat(batch_size, 1)
    adjacency_not_tile = adjacency_not.repeat(batch_size, 1)

    transpose_reshape = pdist_matrix.transpose(0, 1).reshape(-1, 1)
    greater = pdist_matrix_tile > transpose_reshape

    mask = adjacency_not_tile & greater

    # final mask
    mask_step = mask.to(dtype=torch.float32)
    mask_step = mask_step.sum(axis=1)
    mask_step = mask_step > 0.0
    mask_final = mask_step.reshape(batch_size, batch_size)
    mask_final = mask_final.transpose(0, 1)

    adjacency_not = adjacency_not.to(dtype=torch.float32)
    mask = mask.to(dtype=torch.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    axis_maximums = torch.max(pdist_matrix_tile, dim=1, keepdim=True)
    masked_minimums = torch.min(torch.mul(pdist_matrix_tile - axis_maximums[0], mask), dim=1, keepdim=True)[0] + axis_maximums[0]
    negatives_outside = masked_minimums.reshape([batch_size, batch_size])
    negatives_outside = negatives_outside.transpose(0, 1)

    # negatives_inside: largest D_an.
    axis_minimums = torch.min(pdist_matrix, dim=1, keepdim=True)
    masked_maximums = torch.max(torch.mul(pdist_matrix - axis_minimums[0], adjacency_not), dim=1, keepdim=True)[0] + axis_minimums[0]
    negatives_inside = masked_maximums.repeat(1, batch_size)

    semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = margin + pdist_matrix - semi_hard_negatives

    mask_positives = adjacency.to(dtype=torch.float32) - torch.diag(torch.ones(batch_size)).to(device)
    num_positives = mask_positives.sum()

    triplet_loss = (torch.max(torch.mul(loss_mat, mask_positives), torch.tensor([0.]).to(device))).sum() / num_positives
    triplet_loss = triplet_loss.to(dtype=embeddings.dtype)
    return triplet_loss


class TripletLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, input, target):
        return TripletSemiHardLoss(target, input, self.device)


