import numpy as np

from tripletLoader import LibriSpeechVerificationDataset, PKSampler
from utils import TripletLoss, AddNoise, pairwise_distance_torch, TodB
from siameseNet import SiameseNetwork
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchaudio import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_gates(net, title, H, W):
    net.eval()
    gates = net.gates.get_gates().detach().cpu().numpy()
    if len(gates.shape) == 1:
        gates = gates.reshape((H, W))
    plt.imshow(gates, cmap='hot')
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.clim(0.0, 1.0)
    plt.colorbar()
    plt.show()  # Idan - maybe change to savefig since you run using Moba


def plot_loss(loss_list, title):
    plt.plot(loss_list)
    plt.title(title)
    plt.grid(which='major')
    plt.show()  # Idan - maybe change to savefig since you run using Moba


def eval(net, loader, threshold, device):
    net.eval()
    n_samples = loader.dataset.__len__()
    emb_tensor = torch.zeros(n_samples, net.emb_dim).to(device)
    label_tensor = torch.zeros(n_samples, 1).to(device)
    batch_size = loader.batch_size

    # Create Pairwise distance matrix for the data, together with masks for different and same labels:
    id = 0
    for i, (specs, labels) in enumerate(loader, 0):
        specs, labels = specs.unsqueeze(1), labels.unsqueeze(1)
        specs, labels = specs.to(device), labels.to(device)
        output = net(specs)
        output = torch.nn.functional.normalize(output)
        emb_tensor[id:id+batch_size,:] = output.detach_()
        label_tensor[id:id+batch_size] = labels
        id = id + batch_size

    label_same_mask = pairwise_distance_torch(label_tensor, device=device, dtype='float64')  # keep float64! o.w. will return errors.
    label_same_mask = torch.where(label_same_mask == 0.0, 1, 0)
    pdist = pairwise_distance_torch(emb_tensor, device=device)
    threshold_mask = torch.where(pdist <= threshold, 1, 0)

    # Calculate TA (True Accepts), and FA (False Accepts):
    P_same_size = torch.tril(label_same_mask).sum()
    TA_tensor = torch.logical_and(torch.tril(threshold_mask), torch.tril(label_same_mask))

    label_diff_mask = torch.logical_not(label_same_mask.bool()).int()
    P_diff_size = torch.tril(label_diff_mask).sum()
    FA_tensor = torch.logical_and(torch.tril(threshold_mask), torch.tril(label_diff_mask))

    # Calculate VAL(d) (Validation Rate), and FAR (False Accepts Rate) as a function of 'd' (threshold):
    VAL_d = TA_tensor.int().sum() / P_same_size
    FAR_d = FA_tensor.int().sum() / P_diff_size

    return VAL_d, FAR_d


def train(net, train_loader, device, optimizer_func, criterion, optimizer_gates=None, gates_flag=True):
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


def main():
    cuda_idx = 3
    device = f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu"
    # p = 16
    p = 32
    # k = 4
    k = 8
    backbone_type = 'VGG'
    batch_size = p * k
    epochs = 100
    emb_dim = 128
    lam = 1e-5
    gates_flag = False  # True: with stochastic gates, False - without stochastic gates
    # data_path = './dataset/train/'
    data_path = '/home/dsi/idancohen/Unsupervised_Learning/Datasets/Speech/Clean/train/'
    # val_path = './dataset/val/'
    val_path = '/home/dsi/idancohen/Unsupervised_Learning/Datasets/Speech/Clean/val/'

    noises_folder = '/home/dsi/idancohen/Unsupervised_Learning/Datasets/Noises/'
    # noises_folder = './noise/'

    transform = nn.Sequential(
        AddNoise(noises_folder, 0, 'White'),
        transforms.MelSpectrogram(sample_rate=16000, n_fft=512, n_mels=64),
        # transforms.MelSpectrogram(sample_rate=16000, n_fft=512, n_mels=128),
        # transforms.MelSpectrogram(n_fft=256),
        TodB()
    )

    dataset = LibriSpeechVerificationDataset(data_path, transform)
    pk_sampler = PKSampler(dataset, p=p, k=k)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=pk_sampler, shuffle=False, num_workers=2)
    H = dataset[0][0].shape[0]
    W = dataset[0][0].shape[1]
    net = SiameseNetwork(lam=lam, input_shape=[H, W], emb_dim=emb_dim, backbone_type=backbone_type, gates_flag=gates_flag).to(device)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))  # print number of parameters

    criterion = TripletLoss(device)
    optimizer_func = optim.Adam(net.get_func_params(), lr=0.0005)
    if gates_flag:
        optimizer_gates = optim.Adam(net.get_gates_parameters(), lr=0.01)
    else:
        optimizer_gates = None
    # optimizer = optim.Adam(net.parameters(), lr=0.0001)

    # loss_list = []
    for epoch in range(epochs):
        net.loss_list.append(train(net, dataloader, device, optimizer_func,criterion, optimizer_gates, gates_flag=gates_flag))
        # loss_list.append(train_single_optim(net, dataloader, device, optimizer, criterion))
        if gates_flag:
            print(f'Epoch[{epoch+1}/{epochs}]:\t Loss: {net.loss_list[-1]}\tReg: {net.gates.get_reg()}')
            if (epoch + 1) % 10 == 0:
                plot_gates(net, f'Epoch: [{epoch+1}/{epochs}], Loss: {net.loss_list[-1]}', H, W)
        else:
            print(f'Epoch[{epoch+1}/{epochs}]:\t Loss: {net.loss_list[-1]}\t')

    plot_loss(net.loss_list, f'Loss')

    # We now calculate the validation rate and false accept rate as a function of threshold 'd',
    # we can do that both for training dataset and validation dataset, but we use validation set to determine the right threshold.
    FA_lim = 0.05  # We allow 15% False-Accept Rate

    def eval_VAL_FAR_vs_threshold(net, threshold_list, dataloader):
        VAL_d = []
        FAR_d = []
        for threshold in tqdm(threshold_list):
            val, far = eval(net, dataloader, threshold, device)
            VAL_d.append(val.item())
            FAR_d.append(far.item())
        return VAL_d, FAR_d

    def calc_VAL_for_FA_lim(net, FA_lim, dataloader_val_eval):
        threshold_list_exp = np.logspace(-2, 2, 40)

        VAL_d_valset, FAR_d_valset = eval_VAL_FAR_vs_threshold(net, threshold_list_exp, dataloader_val_eval)
        id = np.argmax(np.array(FAR_d_valset) > FA_lim)
        thresh1 = threshold_list_exp[id - 1]
        thresh2 = threshold_list_exp[id]
        threshold_list_lin = np.linspace(thresh1, thresh2, 40)
        VAL_d_valset, FAR_d_valset = eval_VAL_FAR_vs_threshold(net, threshold_list_lin, dataloader_val_eval)

        id_FA_lim = np.abs(np.array(FAR_d_valset) - FA_lim).argmin()
        VAL = VAL_d_valset[id_FA_lim]
        optimal_threshold = threshold_list_lin[id_FA_lim]

        return VAL, optimal_threshold

    # # To estimate and print the VAL and FAR values of the train set:
    threshold_list_exp = np.logspace(-2, 2, 40)
    # dataloader_eval = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # VAL_d_trainset, FAR_d_trainset = eval_VAL_FAR_vs_threshold(net, threshold_list_exp, dataloader_eval)
    # print(np.around(np.array(VAL_d_trainset), 3).tolist())
    # print(np.around(np.array(FAR_d_trainset), 3).tolist())

    # To estimate and print the VAL value that suits FA_lim:

    val_dataset = LibriSpeechVerificationDataset(val_path, transform)
    dataloader_val_eval = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    VAL_for_FA_lim, optimal_threshold = calc_VAL_for_FA_lim(net, FA_lim, dataloader_val_eval)
    print(VAL_for_FA_lim)
    print(optimal_threshold)



if __name__ == '__main__':
    main()
