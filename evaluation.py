import numpy as np
import torch
from tqdm import tqdm
from utils import pairwise_distance_torch

def eval_VAL_FAR_vs_threshold(net, threshold_list, dataloader, device):
    """
    A helper function that is used to calculate the VAL and FAR values of the model for a given threshold list.
    """
    VAL_d = []
    FAR_d = []
    for threshold in tqdm(threshold_list):
        val, far = eval(net, dataloader, threshold, device)
        VAL_d.append(val.item())
        FAR_d.append(far.item())
    return VAL_d, FAR_d

@torch.no_grad()
def calc_VAL_for_FA_lim(net, FA_lim, dataloader_val_eval,device):
    """
        This function calculates the VAL value that suits the acceptable false alarm rate (FA_lim).
    We first use an exponential 1D-grid search, and upon finding the position of FA_lim, we use a finer linear 1D-grid
    search.

    :param net: Our neural-network-model
    :param FA_lim: A value for an acceptable False-Alarm-Rate.
    :param dataloader_val_eval: A dataloader that sequentially goes over the validation dataset.
    :param device: GPU device
    Returns:
    VAl: Validation Rate value for the given FA_lim.
    optimal_threshold: Spherical threshold (in the embedding space) that generates VAL.
    VAL_d_valset_exp: Values of validation rate in an exponential scale
    FAR_d_valset_exp: Values of false-accept rate in an exponential scale
    """
    threshold_list_exp = np.logspace(-2, 2, 40)

    VAL_d_valset_exp, FAR_d_valset_exp = eval_VAL_FAR_vs_threshold(net, threshold_list_exp, dataloader_val_eval, device)
    id = np.argmax(np.array(FAR_d_valset_exp) > FA_lim)
    thresh1 = threshold_list_exp[id - 1]
    thresh2 = threshold_list_exp[id]
    threshold_list_lin = np.linspace(thresh1, thresh2, 40)
    VAL_d_valset, FAR_d_valset = eval_VAL_FAR_vs_threshold(net, threshold_list_lin, dataloader_val_eval, device)

    id_FA_lim = np.abs(np.array(FAR_d_valset) - FA_lim).argmin()
    VAL = VAL_d_valset[id_FA_lim]
    optimal_threshold = threshold_list_lin[id_FA_lim]

    return VAL, optimal_threshold, VAL_d_valset_exp, FAR_d_valset_exp


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
        emb_tensor[id:id+batch_size,:] = output
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

