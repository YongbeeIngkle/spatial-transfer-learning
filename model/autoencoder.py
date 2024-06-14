import os
import numpy as np
import torch
from torch import nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader

class AutoencoderCnn(nn.Module):
    def __init__(self, **input_shapes):
        super(AutoencoderCnn, self).__init__()
        self.input_dim = input_shapes['input_dim']
        self.input_len = input_shapes['input_len']
        self._define_encoder()
        self._define_decoder()
        self._define_pm_estimator()

    def _define_encoder(self):
        self.enocde_layer1 = nn.Conv1d(in_channels=self.input_dim, out_channels=16, kernel_size=1)
        self.encode_layer2 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=1)
        self.encode_layer3 = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, padding="same")
        self.encode_layer4 = nn.Linear(4*self.input_len, 1)

    def _define_decoder(self):
        self.deocde_layer1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding="same")
        self.decode_layer2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding="same")
        self.decode_layer3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=1)
        self.decode_layer4 = nn.Conv1d(in_channels=16, out_channels=self.input_dim, kernel_size=1)

    def _define_pm_estimator(self):
        self.estimator_layer = nn.Linear(self.input_len, 1)

    def encode(self, x):
        x = torch.relu(self.enocde_layer1(x))
        x = torch.relu(self.encode_layer2(x))
        x = torch.relu(self.encode_layer3(x))
        x = x.view(-1, 4*self.input_len)
        x = self.encode_layer4(x)
        return x
    
    def decode(self, x):
        x = x.view(-1, 1, self.input_len)
        x = torch.relu(self.deocde_layer1(x))
        x = torch.relu(self.decode_layer2(x))
        x = torch.relu(self.decode_layer3(x))
        x = self.decode_layer4(x)
        return x
    
    def estimate(self, x):
        x = self.encode(x)
        x = self.estimator_layer(x)
        x = x.flatten()
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

class AutoencoderFeature(nn.Module):
    def __init__(self, latent_dim: int, target_num: int, **input_shapes):
        super(AutoencoderFeature, self).__init__()
        self.latent_dim = latent_dim
        self.target_num = target_num
        self.input_dim = input_shapes['input_dim']
        self.input_len = input_shapes['input_len']
        self._define_encoder()
        self._define_decoder()
        self._define_pm_estimator()

    def _define_encoder(self):
        self.enocde_layer1 = nn.Conv1d(in_channels=self.input_dim, out_channels=16, kernel_size=1)
        self.encode_layer2 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=1)
        self.encode_layer3 = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, padding="same")
        self.encode_layer4 = nn.Linear(4*self.input_len, self.latent_dim)

    def _define_decoder(self):
        self.deocde_layer1 = nn.Linear(self.latent_dim, self.input_len)
        self.decode_layer2 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding="same")
        self.decode_layer3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=1)
        self.decode_layer4 = nn.Conv1d(in_channels=16, out_channels=self.input_dim, kernel_size=1)

    def _define_pm_estimator(self):
        self.estimator_layer = nn.Linear(self.latent_dim, self.target_num)

    def encode(self, x):
        x = torch.relu(self.enocde_layer1(x))
        x = torch.relu(self.encode_layer2(x))
        x = torch.relu(self.encode_layer3(x))
        x = x.view(-1, 4*self.input_len)
        x = self.encode_layer4(x)
        return x
    
    def decode(self, x):
        x = torch.relu(self.deocde_layer1(x))
        x = x.view(-1, 1, self.input_len)
        x = torch.relu(self.decode_layer2(x))
        x = torch.relu(self.decode_layer3(x))
        x = self.decode_layer4(x)
        return x
    
    def estimate(self, x):
        x = self.encode(x)
        x = self.estimator_layer(x)
        # x = x.flatten()
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

class AutoencodeModel:
    def __init__(self, model):
        self.model = model
        self.loss_function = MSELoss()

    def fit(self, train_loader: DataLoader, epoch: int):
        optimizer = torch.optim.NAdam(self.model.parameters(), lr=0.001)
        
        for e in range(epoch):
            total_data_num = 0
            total_recon_loss, total_pm_loss = 0, 0
            for train_x, train_pm in train_loader:
                train_x = train_x.float()
                if len(train_pm.shape) < 2:
                    train_pm = train_pm.float().reshape(-1,1)
                recon = self.model(train_x).float()
                pred = self.model.estimate(train_x).float()
                recon_loss = self.loss_function(recon, train_x) 
                pm_loss = self.loss_function(pred, train_pm)
                total_recon_loss += (recon_loss*train_x.size(0)).item()
                total_pm_loss += (pm_loss*train_x.size(0)).item()
                total_data_num += train_x.size(0)

                optimizer.zero_grad()
                recon_loss.backward()
                pm_loss.backward()
                optimizer.step()
            mean_recon_loss = total_recon_loss/total_data_num
            mean_pm_loss = total_pm_loss/total_data_num
            print(f"Epoch{e+1} - Mean Reconstruction Loss: {mean_recon_loss}, Mean PM2.5 Loss: {mean_pm_loss}")

    def encode(self, encode_loader: DataLoader):
        all_encode_vals = []
        for encode_x, _ in encode_loader:
            encode_x = encode_x.float()
            encode_val = self.model.encode(encode_x).float()
            all_encode_vals.append(encode_val.detach().numpy())
        all_encode_vals = np.vstack(all_encode_vals)
        return all_encode_vals
    
    def freeze(self):
        layers_to_freeze = [
            'deocde_layer1', 'decode_layer2', 'decode_layer3', 'decode_layer4', 'estimator_layer'
        ]
        for name, module in self.model.named_modules():
            if name in layers_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False

class TrainLdfModel:
    def __init__(self, latent_dim: int, target_dim: int, input_shape: set):
        self.latent_dim = latent_dim
        self.target_dim = target_dim
        self.input_shape = input_shape

    def define_model(self):
        model = AutoencoderFeature(latent_dim=self.latent_dim, target_num=self.target_dim, 
                                   input_dim=self.input_shape[0], input_len=self.input_shape[1])
        return AutoencodeModel(model)

    def train(self, save_path: str, train_dataset: dict, epoch: int):
        model = self.define_model()
        model.fit(train_dataset, epoch)
        torch.save(model, save_path)
        self.model = model

    def load(self, save_path: str):
        model = torch.load(save_path)
        self.model = model
    
    def encode(self, save_path: str, test_dataset: DataLoader):
        self.load(save_path)
        encode_val = self.model.encode(test_dataset).flatten()
        return encode_val
