from tripletLoader import LibriSpeechVerificationDataset, PKSampler
from utils import TripletLoss, AddNoise, pairwise_distance_torch, TodB
from siameseNet import SiameseNetwork
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchaudio import transforms

def train_epoch(net, train_loader, device, optimizer_func, criterion, optimizer_gates=None, gates_flag=True):
    net.train()
    running_loss = 0.0

    for i, (specs, labels) in enumerate(train_loader, 1):
        specs, labels = specs.unsqueeze(1), labels.unsqueeze(1)
        specs, labels = specs.to(device), labels.to(device)

        optimizer_func.zero_grad()
        if optimizer_gates != None:
            optimizer_gates.zero_grad()

        output = net(specs)
        output = torch.nn.functional.normalize(output)  # required for the triplet loss
        loss = criterion(output, labels)
        running_loss += loss.item()
        if gates_flag:
            loss += net.gates.get_reg()
        loss.backward()

        optimizer_func.step()
        if gates_flag:
            optimizer_gates.step()
    net.epoch = net.epoch + 1
    return running_loss / i


def train_single_optim(net, train_loader, device, optimizer, criterion):
    net.train()
    running_loss = 0.0
    for i, (specs, labels) in enumerate(train_loader, 1):
        specs, labels = specs.unsqueeze(1), labels.unsqueeze(1)
        specs, labels = specs.to(device), labels.to(device)

        optimizer.zero_grad()

        output = net(specs)
        output = torch.nn.functional.normalize(output)
        loss = criterion(output, labels)
        running_loss += loss.item()
        loss += net.gates.get_reg()
        loss.backward()

        optimizer.step()
    return running_loss / i



def prepare_training(params):
    """
    Set the model, optimizers, and dataloaders

    params['noises_folder']: noises_folder
    params['p']: PK-Sampler value of P - number of labels per batch
    params['k']:PK-Sampler value of K - number of samples per label
    params['backbone_type']: backbone type - either 'VGG' or 'Linear'
    params['emb_dim']: emb_dim
    params['lambda']: lambda value for Stochastic Gates
    params['device']: device
    params['gates']: gates_flag
    params['noise_type']: noise_type
    params['data_path']: path to training dataset
    params['val_path']: path to validation dataset
    params['snr']: SNR value

    """
    batch_size = params['p'] * params['k']

    transform = nn.Sequential(
        AddNoise(params['noises_folder'], params['snr'], params['noise_type']),
        transforms.MelSpectrogram(sample_rate=16000, n_fft=512, n_mels=128),
        TodB()
    )

    dataset = LibriSpeechVerificationDataset(params['data_path'], transform)
    pk_sampler = PKSampler(dataset, p=params['p'], k=params['k'])
    dataloader_train = DataLoader(dataset, batch_size=batch_size, sampler=pk_sampler, shuffle=False, num_workers=2)
    H = dataset[0][0].shape[0]  # Input height
    W = dataset[0][0].shape[1]  # Input width
    params['H'] = H
    params['W'] = W

    # Set Model
    net = SiameseNetwork(lam=params['lambda'], input_shape=[H, W], emb_dim=params['emb_dim'],
                         backbone_type=params['backbone_type'], gates_flag=params['gates']).to(params['device'])
    # print(sum(p.numel() for p in net.parameters() if p.requires_grad))  # print number of parameters

    # Set Criterion:
    criterion = TripletLoss(params['device'])

    # Set Optimizers:
    optimizer_func = optim.Adam(net.get_func_params(), lr=0.0005)
    if params['gates']:
        optimizer_gates = optim.Adam(net.get_gates_parameters(), lr=0.01)
    else:
        optimizer_gates = None

    # Set Evaluation Dataloader:
    val_dataset = LibriSpeechVerificationDataset(params['val_path'], transform)
    dataloader_val_eval = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return net, dataloader_train, criterion, optimizer_func, optimizer_gates, dataloader_val_eval



