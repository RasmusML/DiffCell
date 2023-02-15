import os
import logging

from tqdm import tqdm
import torch
from torch import optim
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
from PIL import Image

from dataset import get_all_MOA_types, get_all_concentration_types
from utils import *

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def make_result_folders(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))


class PrintSize(nn.Module):
    """Utility module to print current shape of a Tensor in Sequential, only at the first pass."""
    
    first = True

    def forward(self, x):
        if self.first:
            print(f"Size: {x.size()}. ")
            self.first = False
        return x


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.channels = channels        
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        
        self.residual = residual
        
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256):
        super().__init__()
        
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)
        #self.bot1 = DoubleConv(256, 256)
        #self.bot3 = DoubleConv(256, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=one_param(self).device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def unet_forward(self, x, t):
        x1 = self.inc(x)
        
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        x4 = self.bot1(x4)
        
        # x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        
        return output
    
    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        return self.unet_forward(x, t)


class UNet_conditional(UNet):
    def __init__(self, n_classes, c_in=3, c_out=3, time_dim=256):
        super().__init__(c_in, c_out, time_dim)

        self.label_emb = nn.Embedding(n_classes, time_dim)
        self.linear = nn.Linear(1, 1)

    def forward(self, x, t, y_labels=None, y_regression=None):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)

        if y_labels is not None:
            t += self.label_emb(y_labels)

        if y_regression is not None:
            t += self.linear(y_regression)

        return self.unet_forward(x, t)

class Linear_variance_scheduler:
    r"""

    References
    ----------
        https://arxiv.org/pdf/2006.11239.pdf

    """

    def __init__(self, beta_start=1e-4, beta_end=0.02):
        self.beta_start = beta_start
        self.beta_end = beta_end

    def get(self, noise_steps):
        beta = torch.linspace(self.beta_start, self.beta_end, noise_steps)
        alpha = 1. - beta
        alpha_hat = torch.cumprod(alpha, dim=0)
        return beta, alpha, alpha_hat


class Cosine_variance_scheduler:
    r"""

    References
    ----------
        https://arxiv.org/pdf/2102.09672.pdf

    """
    def __init__(self, s_offset=0.008, singularities_clip=0.02):
        self.s_offset = s_offset
        self.singularities_clip = singularities_clip

    def get(self, noise_steps):
        f = lambda t: torch.cos((t/noise_steps + self.s_offset) / (1+self.s_offset) * torch.pi/2.)**2
        f_t = f(torch.arange(noise_steps))
        f_0 = f(torch.zeros(1))
        alpha_hat = f_t / f_0
        alpha_hat_left_shift = torch.tensor([1, *alpha_hat[:-1]])
        beta = torch.clip(1 - alpha_hat / alpha_hat_left_shift, max=self.singularities_clip)
        alpha = 1. - beta
        return beta, alpha, alpha_hat


class Diffusion:
    def __init__(self, variance_scheduler=None, noise_steps=1000, img_size=64, device="cpu"):
        self.noise_steps = noise_steps
        self.img_size = img_size
        self.device = device

        if not variance_scheduler:
            variance_scheduler = Cosine_variance_scheduler()

        beta, alpha, alpha_hat = variance_scheduler.get(noise_steps)
        self.beta = beta.to(device)
        self.alpha = alpha.to(device)
        self.alpha_hat = alpha_hat.to(device)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, N_images):
        logging.info(f"Sampling {N_images} new images....")

        model.eval()

        with torch.no_grad():
            x = torch.randn((N_images, 3, self.img_size, self.img_size)).to(self.device)

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(N_images) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2

        return x


class Diffusion_conditional:
    """
        Classifier-Free Guidance
    """
    def __init__(self, variance_scheduler=None, noise_steps=1000, img_size=64, device="cpu"):
        self.noise_steps = noise_steps
        self.img_size = img_size
        self.device = device

        if not variance_scheduler:
            variance_scheduler = Cosine_variance_scheduler()

        beta, alpha, alpha_hat = variance_scheduler.get(noise_steps)
        self.beta = beta.to(device)
        self.alpha = alpha.to(device)
        self.alpha_hat = alpha_hat.to(device)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, N_images, labels, y_regr, cfg_scale=3):
            logging.info(f"Sampling {N_images} new images....")
            model.eval()

            with torch.no_grad():
                x = torch.randn((N_images, 3, self.img_size, self.img_size)).to(self.device)

                for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                    t = (torch.ones(N_images) * i).long().to(self.device)
                    predicted_noise = model(x, t, labels, y_regr)
                    if cfg_scale > 0:
                        uncond_predicted_noise = model(x, t, None, None)
                        predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                    alpha = self.alpha[t][:, None, None, None]
                    alpha_hat = self.alpha_hat[t][:, None, None, None]
                    beta = self.beta[t][:, None, None, None]

                    if i > 1:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)

                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

            model.train()
            x = (x.clamp(-1, 1) + 1) / 2

            return x


def get_MOA_mappings():
    moa_types = get_all_MOA_types()
    
    moa_to_id = {}
    id_to_moa = {}
    for i, moa in enumerate(moa_types):
        moa_to_id[moa] = i
        id_to_moa[i] = moa
        
    return moa_to_id, id_to_moa


def indexify(concentrations):
    all_concentrations = get_all_concentration_types()
    mapping = {}
    
    for i, c in enumerate(all_concentrations):
        mapping[(c)] = i
        
    return np.array([mapping[(c)] for c in concentrations])


def train_diffusion_model(metadata, images, image_size=64, epochs=10, batch_size=2, lr=3e-4, epoch_sample_times=5):
    assert epoch_sample_times <= epochs, "can't sample more times than total epochs"

    run_name = "DDPM_Unconditional"
    make_result_folders(run_name)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    
    dataloader = DataLoader(images, batch_size=batch_size, shuffle=True)

    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=image_size, noise_steps=1000, device=device)

    k = 0
    epoch_sample_points = torch.linspace(1, epochs, epoch_sample_times, dtype=torch.int32)
    
    for epoch in range(1, epochs+1):
        logging.info(f"Starting epoch {epoch}:")

        pbar = tqdm(dataloader)
        for i, images in enumerate(pbar):
            images = images.to(device)

            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())

        if epoch == epoch_sample_points[k]:
            k += 1

            sampled_images = diffusion.sample(model, N_images=images.shape[0])

            epoch_save_dir = os.path.join("results", run_name)
            np.save(os.path.join(epoch_save_dir, f"{epoch}.npy"), sampled_images.cpu().numpy())
            #save_images(sampled_images, os.path.join(epoch_save_dir, f"{epoch}.jpg"))
    
            torch.save(model.state_dict(), os.path.join("models", run_name, f"ckpt{epoch}.pt"))


def log_transform(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x + 1.0)


def train_conditional_diffusion_model(metadata, images, image_size=64, epochs=10, batch_size=2, lr=3e-4, epoch_sample_times=5):
    assert epoch_sample_times <= epochs, "can't sample more times than total epochs"

    run_name = "DDPM_Conditional"
    make_result_folders(run_name)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    # prepare dataset
    concentration_transform = log_transform

    concentrations = torch.tensor(np.array(metadata["Image_Metadata_Concentration"], dtype=np.float32))
    concentrations = concentration_transform(concentrations)
    concentration_mean, concentration_std = concentrations.mean(), concentrations.std()
    concentrations = (concentrations - concentration_mean) / concentration_std

    moa_to_id, _ = get_MOA_mappings()
    moa = torch.from_numpy(np.array([moa_to_id[m] for m in metadata["moa"]]))

    dataset = TensorDataset(images, concentrations[:,None], moa)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # used for random sampling
    concentration_types = torch.tensor(get_all_concentration_types())
    concentration_types = concentration_transform(concentration_types)
    regression_labels = (concentration_types - concentration_mean) / concentration_std

    # setup model and parameters
    n_classes = len(get_all_MOA_types())
    model = UNet_conditional(n_classes).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    diffusion = Diffusion_conditional(img_size=image_size, noise_steps=1000, device=device)

    k = 0
    epoch_sample_points = torch.linspace(1, epochs, epoch_sample_times, dtype=torch.int32)
    
    for epoch in range(1, epochs+1):
        logging.info(f"Starting epoch {epoch}:")

        pbar = tqdm(dataloader)
        for i, (images, concentrations, moa) in enumerate(pbar):
            images = images.to(device)
            concentrations = concentrations.to(device)
            moa = moa.to(device)

            labels = moa
            y_regr = concentrations

            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)

            if np.random.random() < 0.1:
                labels = None

            predicted_noise = model(x_t, t, labels, y_regr)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())

        if epoch == epoch_sample_points[k]:
            k += 1

            labels = torch.arange(n_classes).long().to(device)

            random_concentrations = regression_labels[torch.randint(high=n_classes, size=(n_classes,1))]
            y_regr = random_concentrations.clone().float().to(device)

            sampled_images = diffusion.sample(model, N_images=len(labels), labels=labels, y_regr=y_regr)

            logging.info(f"saving results for epoch {epoch}")

            epoch_save_dir = os.path.join("results", run_name)
            np.save(os.path.join(epoch_save_dir, f"{epoch}.npy"), sampled_images.cpu().numpy())
            #save_images(sampled_images, os.path.join(epoch_save_dir, f"{epoch}.jpg"))
            torch.save(model.state_dict(), os.path.join("models", run_name, f"ckpt{epoch}.pt"))


#
# Treatment classification
#

class Treatment_classifier(nn.Module): 
    def __init__(self, c_in=3, N_moas=13):
        super().__init__()
        
        self.bulk = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=32, kernel_size=5, padding=2),
            # 64h * 64w * 32ch
            nn.MaxPool2d(2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),

            # 32h * 32w * 32ch
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=0),
            # 28h * 28w * 32ch
            nn.MaxPool2d(2),
            # 14h * 14w * 32ch
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=0),
            # 10h * 10w * 32ch
            nn.MaxPool2d(2),
            # 5h * 5w * 32ch
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),

            #  5h * 5w * 32ch
            nn.Conv2d(in_channels=32, out_channels=2*256, kernel_size=5, padding=0),
            # 1h * 1w * 512ch
            nn.BatchNorm2d(2*256),
            nn.Flatten()
        )
        
        bulk_outs = 2*256
        
        self.moa_out = nn.Linear(bulk_outs, N_moas)
        self.concentration_out = nn.Linear(bulk_outs, 1)
        
    def forward(self, images):
        x = self.bulk(images)
        y_moa = self.moa_out(x)
        y_concentration = self.concentration_out(x)
        return y_moa, y_concentration


def train_classifier(train_metadata, train_images, validation_metadata, validation_images, lr=0.01, epochs=50, batch_size=32, epoch_sample_times=10):
    run_name = "Classifier"
    make_result_folders(run_name)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    
    # prepare train dataset
    concentration_transform = log_transform

    train_concentrations = torch.tensor(np.array(train_metadata["Image_Metadata_Concentration"], dtype=np.float32))
    train_concentrations = concentration_transform(train_concentrations)
    train_concentration_mean, train_concentration_std = train_concentrations.mean(), train_concentrations.std()
    train_concentrations = (train_concentrations - train_concentration_mean) / train_concentration_std

    moa_to_id, _ = get_MOA_mappings()
    train_moa = torch.from_numpy(np.array([moa_to_id[m] for m in train_metadata["moa"]]))

    train_dataset = TensorDataset(train_images, train_concentrations[:,None], train_moa)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # prepare validation set
    validation_concentrations = torch.tensor(np.array(validation_metadata["Image_Metadata_Concentration"], dtype=np.float32))
    validation_concentrations = concentration_transform(validation_concentrations)
    validation_concentration_mean, validation_concentration_std = validation_concentrations.mean(), validation_concentrations.std()
    validation_concentrations = (validation_concentrations - validation_concentration_mean) / validation_concentration_std

    moa_to_id, _ = get_MOA_mappings()
    validation_moa = torch.from_numpy(np.array([moa_to_id[m] for m in validation_metadata["moa"]]))

    validation_dataset = TensorDataset(validation_images, validation_concentrations[:,None], validation_moa)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    training_result = {}
    
    moa_loss = nn.CrossEntropyLoss()
    concentration_loss = nn.MSELoss()
    penalty_concentration = .5
    
    model = Treatment_classifier()
    
    training_result["penalty_concentration"] = penalty_concentration
    training_result["train_loss"] = []      # (epoch, loss, moa loss, penalty * concentration loss)
    training_result["validation_loss"] = [] # (epoch, loss, moa loss, penalty * concentration loss)
    training_result["train_accuracy"] = []      # (epoch, accuracy)
    training_result["validation_accuracy"] = [] # (epoch, accuracy)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-5)
    
    k = 0
    epoch_sample_points = torch.linspace(1, epochs, epoch_sample_times, dtype=torch.int32)
    
    for epoch in range(1, epochs+1):
        logging.info(f"Starting epoch {epoch}:")
        
        train_batch_loss = []
        train_batch_accuracy = []

        pbar = tqdm(train_dataloader)
        for i, (images, target_concentrations, target_moa) in enumerate(pbar):
            images = images.to(device)
            target_concentrations = target_concentrations.to(device)
            target_moa = target_moa.to(device)

            pred_moa, pred_concentrations = model(images)
            
            moa_loss_value = moa_loss(pred_moa, target_moa)
            concentration_loss_value = penalty_concentration * concentration_loss(pred_concentrations, target_concentrations)
            loss = moa_loss_value + concentration_loss_value

            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1_000)
            optimizer.step()
            
            train_batch_loss.append((loss.detach().cpu(), moa_loss_value.detach().cpu(), concentration_loss_value.detach().cpu()))
            
            accuracy = (torch.sum(pred_moa.max(1)[1] == target_moa)).numpy() / len(images)
            train_batch_accuracy.append(accuracy)     
                
            pbar.set_postfix(loss=loss.item())

        # store training loss
        train_epoch_loss = np.array(train_batch_loss)
        training_result["train_loss"].append((epoch, train_epoch_loss[:,0].mean(), train_epoch_loss[:,1].mean(), train_epoch_loss[:,2].mean()))
        
        training_result["train_accuracy"].append((epoch, np.mean(np.array(train_batch_accuracy))))
        
        if epoch == epoch_sample_points[k]:
            k += 1
            
            with torch.no_grad():
                model.eval()
                
                validation_batch_loss = []
                validation_batch_accuracy = []
        
                for i, (images, target_concentrations, target_moa) in enumerate(validation_dataloader):
                    images = images.to(device)
                    target_concentrations = target_concentrations.to(device)
                    target_moa = target_moa.to(device)

                    pred_moa, pred_concentrations = model(images)

                    moa_loss_value = moa_loss(pred_moa, target_moa)
                    concentration_loss_value = penalty_concentration * concentration_loss(pred_concentrations, target_concentrations)
                    loss = moa_loss_value + concentration_loss_value

                    validation_batch_loss.append((loss.detach().cpu(), moa_loss_value.detach().cpu(), concentration_loss_value.detach().cpu()))
               
                    accuracy = (torch.sum(pred_moa.max(1)[1] == target_moa)).numpy() / len(images)
                    validation_batch_accuracy.append(accuracy)     
            
                validation_epoch_loss = np.array(validation_batch_loss)
                training_result["validation_loss"].append((epoch, validation_epoch_loss[:,0].mean(), validation_epoch_loss[:,1].mean(), validation_epoch_loss[:,2].mean()))
                    
                training_result["validation_accuracy"].append((epoch, np.mean(np.array(validation_batch_accuracy))))

                model.train()
            
            # store latest model and performance
            torch.save(model.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))
            save_dict(training_result, os.path.join("results", run_name, "train_results.pkl"))  


#
# Seperate predictor models
#

# MOA classifier
class MOA_classifier(nn.Module): 
    def __init__(self, c_in=3, N_moas=13):
        super().__init__()
        
        self.bulk = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=32, kernel_size=5, padding=2),
            # 64h * 64w * 32ch
            nn.MaxPool2d(2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),

            # 32h * 32w * 32ch
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=0),
            # 28h * 28w * 32ch
            nn.MaxPool2d(2),
            # 14h * 14w * 32ch
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=0),
            # 10h * 10w * 32ch
            nn.MaxPool2d(2),
            # 5h * 5w * 32ch
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),

            #  5h * 5w * 32ch
            nn.Conv2d(in_channels=32, out_channels=2*256, kernel_size=5, padding=0),
            # 1h * 1w * 512ch
            nn.BatchNorm2d(2*256),
            nn.Flatten()
        )
        
        bulk_outs = 2*256
        
        self.moa_out = nn.Sequential(
            nn.Linear(bulk_outs, 16),
            nn.Tanh(),
            nn.Linear(16, N_moas))
        
    def forward(self, images):
        x = self.bulk(images)
        y_moa = self.moa_out(x)
        return y_moa


def train_MOA_classifier(train_metadata, train_images, validation_metadata, validation_images, lr=0.01, epochs=50, batch_size=32, epoch_sample_times=10):
    run_name = "MOA_Classifier"
    make_result_folders(run_name)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    
    # prepare train dataset
    moa_to_id, _ = get_MOA_mappings()
    train_moa = torch.from_numpy(np.array([moa_to_id[m] for m in train_metadata["moa"]]))

    train_dataset = TensorDataset(train_images, train_moa)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # prepare validation set
    moa_to_id, _ = get_MOA_mappings()
    validation_moa = torch.from_numpy(np.array([moa_to_id[m] for m in validation_metadata["moa"]]))

    validation_dataset = TensorDataset(validation_images, validation_moa)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    training_result = {}
    
    moa_loss = nn.CrossEntropyLoss()
    
    model = MOA_classifier()
    
    training_result["train_loss"] = []      # (epoch, loss)
    training_result["validation_loss"] = [] # (epoch, loss)
    training_result["train_accuracy"] = []      # (epoch, accuracy)
    training_result["validation_accuracy"] = [] # (epoch, accuracy)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-5)
    
    k = 0
    epoch_sample_points = torch.linspace(1, epochs, epoch_sample_times, dtype=torch.int32)
    
    for epoch in range(1, epochs+1):
        logging.info(f"Starting epoch {epoch}:")
        
        train_batch_loss = []
        train_batch_accuracy = []

        pbar = tqdm(train_dataloader)
        for i, (images, target_moa) in enumerate(pbar):
            images = images.to(device)
            target_moa = target_moa.to(device)

            pred_moa = model(images)
            loss = moa_loss(pred_moa, target_moa)

            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1_000)
            optimizer.step()
            
            train_batch_loss.append(loss.detach().cpu())
            
            accuracy = (torch.sum(pred_moa.max(1)[1] == target_moa)).numpy() / len(images)
            train_batch_accuracy.append(accuracy)     
                
            pbar.set_postfix(loss=loss.item())

        # store training loss
        training_result["train_loss"].append((epoch, np.mean(np.array(train_batch_loss))))
        training_result["train_accuracy"].append((epoch, np.mean(np.array(train_batch_accuracy))))
        
        if epoch == epoch_sample_points[k]:
            k += 1
            
            with torch.no_grad():
                model.eval()
                
                validation_batch_loss = []
                validation_batch_accuracy = []
        
                for i, (images, target_moa) in enumerate(validation_dataloader):
                    images = images.to(device)
                    target_moa = target_moa.to(device)

                    pred_moa = model(images)
                    loss = moa_loss(pred_moa, target_moa)
                    
                    validation_batch_loss.append(loss.detach().cpu())
               
                    accuracy = (torch.sum(pred_moa.max(1)[1] == target_moa)).numpy() / len(images)
                    validation_batch_accuracy.append(accuracy)     
            
                training_result["validation_loss"].append((epoch, np.mean(np.array(validation_batch_loss))))
                training_result["validation_accuracy"].append((epoch, np.mean(np.array(validation_batch_accuracy))))

                model.train()
            
            # store latest model and performance
            torch.save(model.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))
            save_dict(training_result, os.path.join("results", run_name, "train_results.pkl"))  
