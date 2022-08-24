import torch.nn as nn
from stg import StochasticGates
import torch


class FullyConnectedBackbone(nn.Module):
    def __init__(self, in_size, emb_dim):
        super(FullyConnectedBackbone, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, emb_dim)
        )

    def forward(self, x):
        return self.net(x)


class VGG(nn.Module):
    """
    VGG-Style Network, few layers had been dropped dew to number of features
    """
    def __init__(self, input_shape, emb_dim):
        super(VGG, self).__init__()
        self.mp = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * self.calc_final_dim(input_shape), 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, emb_dim))

    def forward(self, x):
        x = self.conv1(x)
        x = self.mp(x)
        x = self.conv2(x)
        x = self.mp(x)
        x = self.conv3(x)
        x = self.mp(x)
        # x = self.conv4(x)
        # x = self.mp(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    @staticmethod
    def calc_final_dim(input_shape):
        def reduce_according_VGG(dimension_size):
            final_size = (dimension_size - 4) // 2
            final_size = (final_size - 4) // 2
            final_size = (final_size - 6) // 2
            # final_size = (final_size - 6) // 2
            return final_size

        final_H = reduce_according_VGG(input_shape[0])
        final_W = reduce_according_VGG(input_shape[1])
        return final_H*final_W


# create the Siamese Neural Network
class SiameseNetwork(nn.Module):
    def __init__(self, lam, input_shape, emb_dim, backbone_type, gates_flag=True):
        super(SiameseNetwork, self).__init__()
        self.emb_dim = emb_dim
        self.input_shape = input_shape
        self.gates_flag = gates_flag
        self.epoch = 1
        self.loss_list = []
        self.backbone_type = backbone_type

        if backbone_type == 'VGG':
            self.backbone = VGG(self.input_shape, self.emb_dim)
            if gates_flag:
                self.gates = StochasticGates(size=input_shape, sigma=1, lam=lam)
        elif backbone_type == 'Linear':
            self.backbone = FullyConnectedBackbone(self.input_shape[0]*self.input_shape[1], self.emb_dim)
            if gates_flag:
                self.gates = StochasticGates(size=self.input_shape[0]*self.input_shape[1], sigma=1, lam=lam)

    def get_func_params(self):
        params = list()
        params += list(self.backbone.parameters())
        return params

    def get_gates_parameters(self):
        params = list()
        params += list(self.gates.parameters())
        return params

    def forward(self, x):
        if self.backbone_type == 'Linear':
            x = torch.flatten(x, 1)
        if self.gates_flag:
            x = self.gates(x)
        x = self.backbone(x)
        return x

    def save_model(self, model_save_dir, model_name, optimizer_func, optimizer_gates=None):
        if self.gates_flag:
            torch.save({'epoch': self.epoch, 'model_state_dict': self.state_dict(),
                        'optimizer_func_state_dict': optimizer_func.state_dict(), 'losses': self.loss_list,
                        'optimizer_gates_state_dict': optimizer_gates.state_dict(), },
                       model_save_dir + model_name + '.pt')
        else:
            # save only optimizer_func
            torch.save({'epoch': self.epoch, 'model_state_dict': self.state_dict(),
                        'optimizer_func_state_dict': optimizer_func.state_dict(), 'losses': self.loss_list},
                       model_save_dir + model_name + '.pt')

    def load_model(self, model_loadpath, device, optimizer_func, optimizer_gates=None):
        state_dict = torch.load(model_loadpath, map_location=device)
        self.epoch = state_dict.get('epoch')
        self.load_state_dict(state_dict.get('model_state_dict'))
        self.loss_list = state_dict.get('losses')
        optimizer_func.load_state_dict(state_dict.get('optimizer_func_state_dict'))
        optimizer_func.param_groups[0]['capturable'] = True
        if self.gates_flag:
            optimizer_gates.load_state_dict(state_dict.get('optimizer_gates_state_dict'))
            optimizer_gates.param_groups[0]['capturable'] = True
