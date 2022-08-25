import shutil
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from evaluation import calc_VAL_for_FA_lim
from training import train_epoch, prepare_training


def create_folder(dst_path):
    try:
        os.mkdir(dst_path)
        print(f'The following directory was created:\n\'{dst_path}\'')
    except:
        shutil.rmtree(dst_path)
        os.mkdir(dst_path)
        print(f'The following directory was removed and created again:\n\'{dst_path}\'')
    return


def plot_gates(net, title, H, W):
    if net.__class__ == np.ndarray:  # We allow net to be the gates themselves:
        gates = net
    else:
        net.eval()
        gates = net.gates.get_gates().detach().cpu().numpy()

    if len(gates.shape) == 1:
        gates = gates.reshape((H, W))
    fig = plt.figure()
    plt.imshow(gates, cmap='hot')
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.clim(0.0, 1.0)
    plt.colorbar()
    return fig


def plot_loss(loss_list, title):
    fig = plt.figure()
    plt.plot(loss_list)
    plt.title(title)
    plt.grid(which='major')
    return fig


def train_model(net, dataloader_train, criterion, optimizer_func, optimizer_gates, dataloader_val_eval, params):
    """
    params['max_epochs']: maximal amount of epochs to train
    params['FA_lim'] = 0.15  # Our acceptable value for False-Accept-Rate.
    params['eval_VAL_FAR_every']: every () epochs we evaluate the VAL value for our acceptable False-Accept-Rate.
    params['save_folder']: folder in which we save our current run results.
    We stop training once we see decline in the VAL value.
    :return:
    """
    epochs = params['max_epochs']
    best_VAL = 0
    best_threshold = 0
    early_stopping_flag = False
    # Set a file for saving the VAL-FAR lists in every evaluation.
    with open(params['save_folder'] + '/VAL-FAR_vs_Epoch.txt', 'w') as f:
        pass
    for epoch in range(epochs):
        net.loss_list.append(train_epoch(net, dataloader_train, params['device'], optimizer_func, criterion,
                                   optimizer_gates, gates_flag=params['gates']))

        # Evaluate every few epochs:
        if (epoch + 1) % params['eval_VAL_FAR_every'] == 0:
            # Evaluate pereformance over validation set:
            VAL, optimal_threshold, VAL_d_valset_exp, FAR_d_valset_exp = \
                calc_VAL_for_FA_lim(net, params['FA_lim'], dataloader_val_eval, params['device'])

            # Update the file with the new VAL-FAR lists of the current evaluation:
            with open(params['save_folder'] + '/VAL-FAR_vs_Epoch.txt', 'a') as f:
                VAL_d_valset_exp_str = '[' + ', '.join(f'{x:.3f}' for x in VAL_d_valset_exp) + ']'  # list to string
                FAR_d_valset_exp_str = '[' + ', '.join(f'{x:.3f}' for x in FAR_d_valset_exp) + ']'  # list to string
                f.write(f'Epoch {epoch+1}:\n')
                f.write('Validation-Rate (VAL) list:   ' + VAL_d_valset_exp_str + '\n')
                f.write('False-Accept-Rate (FAR) list: ' + FAR_d_valset_exp_str + '\n')
                f.write(f'current VAL is: {VAL:.3f}:\n')

            if FAR_d_valset_exp[0] < 0.1:
                # The above FAR condition is needed since in the early beginning of training we have
                # both FAR and VAL values around 1, as the embedding is very dense.
                if best_VAL < VAL:
                    best_VAL = VAL
                    best_threshold = optimal_threshold
                    counter_early_stop = 0

                    if params['gates']:
                        best_gates = net.gates.get_gates().detach().cpu().numpy()
                else:
                    counter_early_stop = counter_early_stop + 1
                    if counter_early_stop == 10:  # Stop after 10 evaluations without improvement)
                        early_stopping_flag = True

            # Save current gates:
            if params['gates']:
                fig = plot_gates(net, f'Epoch: [{epoch+1}/{epochs}], Loss: {net.loss_list[-1]:.3f}', params['H'], params['W'])
                figname = f'Gates - Epoch {epoch + 1}, Loss {net.loss_list[-1]:.3f}, cur_VAL={VAL:.3f}'
                plt.savefig(params['save_folder'] + '/' + figname + '.jpg')
                plt.close()


        # Print loss every epoch:
        if params['gates']:
            print(f'Epoch[{epoch + 1}/{epochs}]:\t Loss: {net.loss_list[-1]:.3f}\tReg: {net.gates.get_reg():.3f}')
        else:
            print(f'Epoch[{epoch + 1}/{epochs}]:\t Loss: {net.loss_list[-1]:.3f}\t')

        #### End of training: ####
        # If we stopped early or if we're on the last epoch:
        if early_stopping_flag or epoch + 1 == params['max_epochs']:
            # Save best gates:
            if params['gates']:
                fig = plot_gates(best_gates, f'Epoch: [{epoch + 1}/{epochs}], Loss: {net.loss_list[-1]:.3f}', params['H'], params['W'])
                figname = f'Best Gates - Epoch {epoch + 1}, Loss {net.loss_list[-1]:.3f}, best_VAL={best_VAL:.3f}'
                plt.savefig(params['save_folder'] + '/' + figname + '.jpg')
                plt.close()

            # Save best VAL value and its threshold
            with open(params['save_folder'] + '/VAL-FAR_vs_Epoch.txt', 'a') as f:
                f.write(f'Best VAL is: {best_VAL:.3f}:\n')
                f.write(f'Threshold of best VAL is: {best_threshold:.3f}:\n')

            # Save loss curve:
            fig = plot_loss(net.loss_list, f'Loss')
            figname = f'Loss Curve - Epoch {epoch + 1}, Final_Loss {net.loss_list[-1]:.3f}'
            plt.savefig(params['save_folder'] + '/' + figname + '.jpg')
            plt.close()

            if epoch == params['max_epochs']:
                with open(params['save_folder'] + '/VAL-FAR_vs_Epoch.txt', 'a') as f:
                    f.write(f'Training ended after {epochs} epochs and is not complete.\n')

            break

    return


if __name__ == '__main__':

    noises_folder = './noise/'
    data_path = './dataset/train/'
    val_path = './dataset/val/'
    results_dir = './results/'

    cuda_idx = 2
    device = f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu"

    params = {}
    params['device'] = device
    params['p'] = 16
    params['k'] = 8
    params['backbone_type'] = 'VGG'
    params['emb_dim'] = 129
    # params['lambda'] = 0.3*1e-4
    params['lambda'] = 1e-5
    params['noises_folder'] = noises_folder
    params['max_epochs'] = 2000
    params['data_path'] = data_path
    params['val_path'] = val_path
    params['eval_VAL_FAR_every'] = 10  # eval VAL and FAR values of the models every () epochs.
    params['FA_lim'] = 0.05  # Our acceptable value for false alarm


    snr_list = [ -5, 0, 5, 10, 15, 20]
    noise_type_list = ['None', 'White']
    # noise_type_list = ['Room']

    for noise_type in noise_type_list:
        params['noise_type'] = noise_type
        noise_folder = results_dir + noise_type
        create_folder(noise_folder)

        if noise_type == 'None':  # Without SNR
            params['snr'] = 300  # Need to pass a value, but it won't be used
            # Train without gates:
            params['gates'] = False
            no_gates_folder = noise_folder + '/' + 'No_Gates'
            params['save_folder'] = no_gates_folder
            create_folder(no_gates_folder)
            net, dataloader_train, criterion, optimizer_func, optimizer_gates, dataloader_val_eval = prepare_training(
                params)
            print(f'Starting training - noise type: {noise_type}, Gates: False')
            train_model(net, dataloader_train, criterion, optimizer_func, optimizer_gates, dataloader_val_eval, params)

            # Train with gates:
            params['gates'] = True
            gates_folder = noise_folder + '/' + 'Gates'
            params['save_folder'] = gates_folder
            create_folder(gates_folder)
            net, dataloader_train, criterion, optimizer_func, optimizer_gates, dataloader_val_eval = prepare_training(
                params)
            print(f'Starting training - noise type: {noise_type}, Gates: True')
            train_model(net, dataloader_train, criterion, optimizer_func, optimizer_gates, dataloader_val_eval, params)



        else:
            for snr in snr_list:
                params['snr'] = snr
                snr_folder = noise_folder + '/' + 'SNR=' + str(snr)
                create_folder(snr_folder)

                # Train without gates:
                params['gates'] = False
                no_gates_folder = snr_folder + '/' + 'No_Gates'
                params['save_folder'] = no_gates_folder
                create_folder(no_gates_folder)
                net, dataloader_train, criterion, optimizer_func, optimizer_gates, dataloader_val_eval = prepare_training(params)
                print(f'Starting training - noise type: {noise_type}, Gates: False, SNR: {snr}dB')
                train_model(net, dataloader_train, criterion, optimizer_func, optimizer_gates, dataloader_val_eval, params)

                # Train with gates:
                params['gates'] = True
                gates_folder = snr_folder + '/' + 'Gates'
                params['save_folder'] = gates_folder
                create_folder(gates_folder)
                net, dataloader_train, criterion, optimizer_func, optimizer_gates, dataloader_val_eval = prepare_training(params)
                print(f'Starting training - noise type: {noise_type}, Gates: True, SNR: {snr}dB')
                train_model(net, dataloader_train, criterion, optimizer_func, optimizer_gates, dataloader_val_eval, params)
