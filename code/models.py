import os
import logging

from collections import defaultdict
from typing import List, Set, Dict, Tuple, Optional, Any

from tqdm import tqdm
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from dataset import *
from utils import *

def make_training_folders(run_name):
    os.makedirs(os.path.join("results", run_name, "training"), exist_ok=True)
    os.makedirs(os.path.join("results", run_name, "weights"), exist_ok=True)

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
    def __init__(self, n_compounds, n_concentrations, c_in=3, c_out=3, time_dim=256):
        super().__init__(c_in, c_out, time_dim)

        self.compound_embedding = nn.Embedding(n_compounds, time_dim)
        self.concentration_embedding = nn.Embedding(n_concentrations, time_dim)

    def forward(self, x, t, y_compounds=None, y_concentrations=None):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)

        if y_compounds is not None:
            t += self.compound_embedding(y_compounds)

        if y_concentrations is not None:
            t += self.concentration_embedding(y_concentrations)

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

        x = (x.clamp(-1, 1) + 1) / 2
        model.train()

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

    def sample(self, model, N_images, y_compounds, y_concentrations, cfg_scale=3):
        logging.info(f"Sampling {N_images} new images....")
        model.eval()

        with torch.no_grad():
            x = torch.randn((N_images, 3, self.img_size, self.img_size)).to(self.device)

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(N_images) * i).long().to(self.device)
                predicted_noise = model(x, t, y_compounds, y_concentrations)
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

        x = (x.clamp(-1, 1) + 1) / 2
        model.train()

        return x

    def sample2(self, model, N_images, y_compounds, y_concentrations, cfg_scale=3, store_count=10):
        logging.info(f"Sampling {N_images} new images....")
        model.eval()

        space_index = 0
        spacing = np.linspace(self.noise_steps-1, 1, store_count, dtype=np.int32)
        xs = torch.empty((len(spacing), N_images, 3, self.img_size, self.img_size))

        with torch.no_grad():
            x = torch.randn((N_images, 3, self.img_size, self.img_size)).to(self.device)

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(N_images) * i).long().to(self.device)
                predicted_noise = model(x, t, y_compounds, y_concentrations)
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

                if i == spacing[space_index]:
                    xs[space_index] = (x.clone().clamp(-1, 1) + 1) / 2
                    space_index += 1

        return xs

def train_diffusion_model(metadata, images, image_size=64, epochs=10, batch_size=2, lr=3e-4, epoch_sample_times=5):
    assert epoch_sample_times <= epochs, "can't sample more times than total epochs"

    run_name = "DDPM_Unconditional"
    make_training_folders(run_name)
    
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

            np.save(os.path.join("results", run_name, "training", f"{epoch}.npy"), sampled_images.cpu().numpy())
            torch.save(model.state_dict(), os.path.join("results", run_name, "weights", f"ckpt{epoch}.pt"))


def train_conditional_diffusion_model(metadata, images, compound_types, concentration_types, image_size=64, epochs=10, batch_size=2, lr=3e-4, epoch_sample_times=5):
    assert epoch_sample_times <= epochs, "can't sample more times than total epochs"

    run_name = "DDPM_Conditional"
    make_training_folders(run_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    # prepare dataset
    compound_to_id, _ = get_label_mappings(compound_types)
    compounds = torch.from_numpy(np.array([compound_to_id[c] for c in metadata["Image_Metadata_Compound"]]))

    concentration_to_id, _ = get_label_mappings(concentration_types)
    concentrations = torch.from_numpy(np.array([concentration_to_id[c] for c in metadata["Image_Metadata_Concentration"]]))

    dataset = TensorDataset(images, compounds, concentrations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # setup model and parameters
    n_compounds = len(compound_types)
    n_concentrations = len(concentration_types)

    model = UNet_conditional(n_compounds, n_concentrations).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    diffusion = Diffusion_conditional(img_size=image_size, noise_steps=1000, device=device)

    results = {}
    results["loss"] = []

    k = 0
    epoch_sample_points = torch.linspace(1, epochs, epoch_sample_times, dtype=torch.int32)
    
    for epoch in range(1, epochs+1):
        logging.info(f"Starting epoch {epoch}:")

        pbar = tqdm(dataloader)
        for i, (images, compounds, concentrations) in enumerate(pbar):
            images = images.to(device)
            compounds = compounds.to(device)
            concentrations = concentrations.to(device)

            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)

            if np.random.random() < 0.1:
                compounds = None
                concentrations = None

            predicted_noise = model(x_t, t, compounds, concentrations)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            results["loss"].append(loss.detach().cpu())

            pbar.set_postfix(MSE=loss.item())

        if epoch == epoch_sample_points[k]:
            k += 1

            N = n_compounds

            y_compound = torch.arange(N).long().to(device)
            y_concentrations = (torch.ones(N) * concentration_to_id[1.]).long().to(device)

            sampled_images = diffusion.sample(model, N_images=len(y_compound), y_compounds=y_compound, y_concentrations=y_concentrations)

            logging.info(f"saving results for epoch {epoch}")

            np.save(os.path.join("results", run_name, "training", f"{epoch}.npy"), sampled_images.cpu().numpy())
            torch.save(model.state_dict(), os.path.join("results", run_name, "weights", f"ckpt{epoch}.pt"))

    torch.save(model.state_dict(), os.path.join("results", run_name, "weights", f"ckpt.pt"))
    save_dict(results, os.path.join("results", run_name, "training", "results.pkl"))

#
# predictor model
#
class Compound_classifier(nn.Module): 
    def __init__(self,  N_compounds, c_in=3):
        super().__init__()

        p=.02
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=32, kernel_size=3, padding=1),
            # 64h * 64w * 32ch
            nn.MaxPool2d(2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Dropout(p=p),

            # 32h * 32w * 32ch
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            # 16h * 16w * 32ch
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Dropout(p=p),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            # 16h * 16w * 64ch
            nn.MaxPool2d(2),
            # 8h * 8w * 64ch
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Dropout(p=p),

            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            # 4h * 4w * 96ch
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(96),
            nn.Dropout(p=p),

            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            # 2h * 2w * 128ch
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(128),
            nn.Dropout(p=p),


            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            # 1h * 1w * 256ch
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(256),
            nn.Dropout(p=p),

            nn.Flatten(),

            nn.Linear(256, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(16),
            nn.Dropout(p=p),
            nn.Linear(16, N_compounds))
        
    def forward(self, images):
        return self.net(images)


def train_compound_classifier(train_metadata, train_images, validation_metadata, validation_images, lr=0.001, epochs=200, batch_size=64, epoch_sample_times=50):
    run_name = "Compound_Classifier"
    make_training_folders(run_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    whitelist = get_treatment_whitelist()
    compound_types = extract_compound_types(whitelist)
    compound_to_id, _ = get_label_mappings(compound_types)
    
    # prepare train dataset
    train_compounds = torch.from_numpy(np.array([compound_to_id[c] for c in train_metadata["Image_Metadata_Compound"]]))
    train_dataset = TensorDataset(train_images, train_compounds)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # prepare validation dataset
    validation_compounds = torch.from_numpy(np.array([compound_to_id[c] for c in validation_metadata["Image_Metadata_Compound"]]))
    validation_dataset = TensorDataset(validation_images, validation_compounds)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    # setup model and parameters
    n_compounds = len(compound_types)
    
    loss_fn = nn.CrossEntropyLoss()
    model = Compound_classifier2(n_compounds).to(device)

    training_result = {}
    training_result["train_loss"] = []      # (epoch, loss)
    training_result["validation_loss"] = [] # (epoch, loss)
    training_result["train_accuracy"] = []      # (epoch, accuracy)
    training_result["validation_accuracy"] = [] # (epoch, accuracy)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    k = 0
    epoch_sample_points = torch.linspace(1, epochs, epoch_sample_times, dtype=torch.int32)

    l2 = 1e-3
    
    for epoch in range(1, epochs+1):
        logging.info(f"Starting epoch {epoch}:")
        
        train_batch_loss = []
        train_batch_accuracy = []

        pbar = tqdm(train_dataloader)
        for i, (images, target_compound) in enumerate(pbar):
            images = images.to(device)
            target_compound = target_compound.to(device)

            pred_compound = model(images)
            loss = loss_fn(pred_compound, target_compound)

            l2_penalty = l2 * sum([(p**2).sum() for p in model.parameters()])
            loss += l2_penalty

            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1_000)
            optimizer.step()
            
            train_batch_loss.append(loss.detach().cpu())
            
            accuracy = (torch.sum(pred_compound.max(1)[1] == target_compound)).cpu().numpy() / len(images)
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
                
                for i, (images, target_compound) in enumerate(validation_dataloader):
                    images = images.to(device)
                    target_compound = target_compound.to(device)

                    pred_compound = model(images)
                    loss = loss_fn(pred_compound, target_compound)
                    
                    validation_batch_loss.append(loss.detach().cpu())
                
                    accuracy = (torch.sum(pred_compound.max(1)[1] == target_compound)).cpu().numpy() / len(images)
                    validation_batch_accuracy.append(accuracy)     
            
                training_result["validation_loss"].append((epoch, np.mean(np.array(validation_batch_loss))))
                training_result["validation_accuracy"].append((epoch, np.mean(np.array(validation_batch_accuracy))))

                logging.info(f"train accuracy: {np.mean(np.array(train_batch_accuracy))}")
                logging.info(f"validation accuracy: {np.mean(np.array(validation_batch_accuracy))}")

                model.train()
            
            # store latest model and performance
            logging.info("saving")
            torch.save(model.state_dict(), os.path.join("results", run_name, "weights", f"ckpt{epoch}.pt"))
            save_dict(training_result, os.path.join("results", run_name, "training", "train_results.pkl"))  

    torch.save(model.state_dict(), os.path.join("results", run_name, "weights", f"ckpt.pt"))


class Concentration_classifier(nn.Module): 
    def __init__(self,  N_concentrations, c_in=3):
        super().__init__()

        p=.02
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=32, kernel_size=3, padding=1),
            # 64h * 64w * 32ch
            nn.MaxPool2d(2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Dropout(p=p),

            # 32h * 32w * 32ch
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            # 16h * 16w * 32ch
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Dropout(p=p),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            # 16h * 16w * 64ch
            nn.MaxPool2d(2),
            # 8h * 8w * 64ch
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Dropout(p=p),

            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            # 4h * 4w * 96ch
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(96),
            nn.Dropout(p=p),

            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            # 2h * 2w * 128ch
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(128),
            nn.Dropout(p=p),


            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            # 1h * 1w * 256ch
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(256),
            nn.Dropout(p=p),

            nn.Flatten(),

            nn.Linear(256, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(16),
            nn.Dropout(p=p),
            nn.Linear(16, N_concentrations))
        
    def forward(self, images):
        return self.net(images)


def train_concentration_classifier(train_metadata, train_images, validation_metadata, validation_images, lr=0.001, epochs=200, batch_size=64, epoch_sample_times=50):
    run_name = "Concentration_Classifier"
    make_training_folders(run_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    whitelist = get_treatment_whitelist()
    concentration_types = extract_concentration_types(whitelist)
    concentration_to_id, _ = get_label_mappings(concentration_types)
    
    # prepare train dataset
    train_concentrations = torch.from_numpy(np.array([concentration_to_id[c] for c in train_metadata["Image_Metadata_Concentration"]]))
    train_dataset = TensorDataset(train_images, train_concentrations)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # prepare validation dataset
    validation_compounds = torch.from_numpy(np.array([concentration_to_id[c] for c in validation_metadata["Image_Metadata_Concentration"]]))
    validation_dataset = TensorDataset(validation_images, validation_compounds)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    # setup model and parameters
    n_concentrations = len(concentration_types)
    
    loss_fn = nn.CrossEntropyLoss()
    model = Concentration_classifier(n_concentrations).to(device)

    training_result = {}
    training_result["train_loss"] = []      # (epoch, loss)
    training_result["validation_loss"] = [] # (epoch, loss)
    training_result["train_accuracy"] = []      # (epoch, accuracy)
    training_result["validation_accuracy"] = [] # (epoch, accuracy)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    k = 0
    epoch_sample_points = torch.linspace(1, epochs, epoch_sample_times, dtype=torch.int32)

    l2 = 1e-3
    
    for epoch in range(1, epochs+1):
        logging.info(f"Starting epoch {epoch}:")
        
        train_batch_loss = []
        train_batch_accuracy = []

        pbar = tqdm(train_dataloader)
        for i, (images, target_concentration) in enumerate(pbar):
            images = images.to(device)
            target_concentration = target_concentration.to(device)

            pred_concentration = model(images)
            loss = loss_fn(pred_concentration, target_concentration)

            l2_penalty = l2 * sum([(p**2).sum() for p in model.parameters()])
            loss += l2_penalty

            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1_000)
            optimizer.step()
            
            train_batch_loss.append(loss.detach().cpu())
            
            accuracy = (torch.sum(pred_concentration.max(1)[1] == target_concentration)).cpu().numpy() / len(images)
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
                
                for i, (images, target_concentration) in enumerate(validation_dataloader):
                    images = images.to(device)
                    target_concentration = target_concentration.to(device)

                    pred_concentration = model(images)
                    loss = loss_fn(pred_concentration, target_concentration)
                    
                    validation_batch_loss.append(loss.detach().cpu())
                
                    accuracy = (torch.sum(pred_concentration.max(1)[1] == target_concentration)).cpu().numpy() / len(images)
                    validation_batch_accuracy.append(accuracy)     
            
                training_result["validation_loss"].append((epoch, np.mean(np.array(validation_batch_loss))))
                training_result["validation_accuracy"].append((epoch, np.mean(np.array(validation_batch_accuracy))))

                logging.info(f"train accuracy: {np.mean(np.array(train_batch_accuracy))}")
                logging.info(f"validation accuracy: {np.mean(np.array(validation_batch_accuracy))}")

                model.train()
            
            # store latest model and performance
            logging.info("saving")
            torch.save(model.state_dict(), os.path.join("results", run_name, "weights", f"ckpt{epoch}.pt"))
            save_dict(training_result, os.path.join("results", run_name, "training", "train_results.pkl"))  

    torch.save(model.state_dict(), os.path.join("results", run_name, "weights", f"ckpt.pt"))


#
# Experimential
#

class Concentration_classifier2(nn.Module): 
    def __init__(self,  N_concentrations, N_compounds, c_in=3):
        super().__init__()

        p=.02
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=32, kernel_size=3, padding=1),
            # 64h * 64w * 32ch
            nn.MaxPool2d(2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Dropout(p=p),

            # 32h * 32w * 32ch
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            # 16h * 16w * 32ch
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Dropout(p=p),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            # 16h * 16w * 64ch
            nn.MaxPool2d(2),
            # 8h * 8w * 64ch
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Dropout(p=p),

            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            # 4h * 4w * 96ch
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(96),
            nn.Dropout(p=p),

            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            # 2h * 2w * 128ch
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(128),
            nn.Dropout(p=p),


            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            # 1h * 1w * 256ch
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(256),
            nn.Dropout(p=p),

            nn.Flatten())

        self.net2 = nn.Sequential(
            nn.Linear(256+N_compounds, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(16),
            nn.Dropout(p=p),
            nn.Linear(16, N_concentrations))
        
    def forward(self, images, compounds):
        img_embedding = self.net(images)
        concated = torch.concat([img_embedding, compounds], dim=1)
        return self.net2(concated)


def train_concentration_classifier2(train_metadata, train_images, validation_metadata, validation_images, lr=0.001, epochs=200, batch_size=64, epoch_sample_times=50):
    run_name = "Concentration_Classifier"
    make_training_folders(run_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    whitelist = get_treatment_whitelist()
    concentration_types = extract_concentration_types(whitelist)
    concentration_to_id, _ = get_label_mappings(concentration_types)
    
    compound_types = extract_compound_types(whitelist)
    compound_to_id, _ = get_label_mappings(compound_types)

    # prepare train dataset
    train_concentrations = torch.from_numpy(np.array([concentration_to_id[c] for c in train_metadata["Image_Metadata_Concentration"]]))
    train_compounds = torch.from_numpy(np.array([compound_to_id[c] for c in train_metadata["Image_Metadata_Compound"]]))
    train_dataset = TensorDataset(train_images, train_compounds, train_concentrations)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # prepare validation dataset
    validation_concentrations = torch.from_numpy(np.array([concentration_to_id[c] for c in validation_metadata["Image_Metadata_Concentration"]]))
    validation_compounds = torch.from_numpy(np.array([compound_to_id[c] for c in validation_metadata["Image_Metadata_Compound"]]))
    validation_dataset = TensorDataset(validation_images, validation_compounds, validation_concentrations)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    # setup model and parameters
    n_concentrations = len(concentration_types)
    n_compounds = len(compound_types)
    
    loss_fn = nn.CrossEntropyLoss()
    model = Concentration_classifier2(n_concentrations, n_compounds).to(device)

    training_result = {}
    training_result["train_loss"] = []      # (epoch, loss)
    training_result["validation_loss"] = [] # (epoch, loss)
    training_result["train_accuracy"] = []      # (epoch, accuracy)
    training_result["validation_accuracy"] = [] # (epoch, accuracy)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    k = 0
    epoch_sample_points = torch.linspace(1, epochs, epoch_sample_times, dtype=torch.int32)

    l2 = 1e-3
    
    for epoch in range(1, epochs+1):
        logging.info(f"Starting epoch {epoch}:")
        
        train_batch_loss = []
        train_batch_accuracy = []

        pbar = tqdm(train_dataloader)
        for i, (images, compound, target_concentration) in enumerate(pbar):
            images = images.to(device)

            compound = F.one_hot(compound, num_classes=n_compounds)
            compound = compound.to(device)

            target_concentration = target_concentration.to(device)

            pred_concentration = model(images, compound)
            loss = loss_fn(pred_concentration, target_concentration)

            l2_penalty = l2 * sum([(p**2).sum() for p in model.parameters()])
            loss += l2_penalty

            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1_000)
            optimizer.step()
            
            train_batch_loss.append(loss.detach().cpu())
            
            accuracy = (torch.sum(pred_concentration.max(1)[1] == target_concentration)).cpu().numpy() / len(images)
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
                
                for i, (images, compound, target_concentration) in enumerate(validation_dataloader):
                    images = images.to(device)

                    compound = F.one_hot(compound, num_classes=n_compounds)
                    compound = compound.to(device)

                    target_concentration = target_concentration.to(device)

                    pred_concentration = model(images, compound)
                    loss = loss_fn(pred_concentration, target_concentration)
                    
                    validation_batch_loss.append(loss.detach().cpu())
                
                    accuracy = (torch.sum(pred_concentration.max(1)[1] == target_concentration)).cpu().numpy() / len(images)
                    validation_batch_accuracy.append(accuracy)     
            
                training_result["validation_loss"].append((epoch, np.mean(np.array(validation_batch_loss))))
                training_result["validation_accuracy"].append((epoch, np.mean(np.array(validation_batch_accuracy))))

                logging.info(f"train accuracy: {np.mean(np.array(train_batch_accuracy))}")
                logging.info(f"validation accuracy: {np.mean(np.array(validation_batch_accuracy))}")

                model.train()
            
            # store latest model and performance
            logging.info("saving")
            torch.save(model.state_dict(), os.path.join("results", run_name, "weights", f"ckpt{epoch}.pt"))
            save_dict(training_result, os.path.join("results", run_name, "training", "train_results.pkl"))  

    torch.save(model.state_dict(), os.path.join("results", run_name, "weights", f"ckpt.pt"))


#
# VAE
#

class CytoVariationalAutoencoder(nn.Module):
 
    def __init__(self, input_shape, latent_features: int):
        super().__init__()
        
        self.input_shape = input_shape
        self.latent_features = latent_features
        self.observation_features = np.prod(input_shape)
        self.observation_shape = input_shape
        self.input_channels = input_shape[0]
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=0),
            nn.MaxPool2d(2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=0),
            nn.MaxPool2d(2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32, out_channels=2*self.latent_features, kernel_size=5, padding=0),
            nn.BatchNorm2d(2*self.latent_features),
            nn.Flatten()
        )

        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (self.latent_features,1,1)),
            nn.Conv2d(in_channels=self.latent_features, out_channels=32, kernel_size=5, padding=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            torch.nn.UpsamplingNearest2d(size=10),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            torch.nn.UpsamplingNearest2d(size=28),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            torch.nn.UpsamplingNearest2d(size=64),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, padding=0),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
    def observation(self, z:torch.Tensor) -> torch.Tensor:
        mu = self.decoder(z)
        mu = mu.view(-1, *self.input_shape)
        return mu

    def forward(self, x) -> Dict[str, Any]:
        h_z = self.encoder(x)
        qz_mu, qz_log_sigma =  h_z.chunk(2, dim=-1)        
        eps = torch.empty_like(qz_mu).normal_()
        z = qz_mu + qz_log_sigma.exp() * eps
        
        x_hat = self.observation(z)
        
        return {'x_hat': x_hat, 'qz_log_sigma': qz_log_sigma, 'qz_mu': qz_mu, 'z': z}    


class VariationalInference_VAE(nn.Module):
    
    def __init__(self, beta:float=1., p_norm = 2.):
        super().__init__()
        self.beta = beta
        self.p_norm = float(p_norm)

    def forward(self, model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        outputs = model(x)

        # Unpack values from VAE
        x_hat, qz_log_sigma, qz_mu, z = [outputs[k] for k in ["x_hat", "qz_log_sigma", "qz_mu", "z"]]
        qz_sigma = qz_log_sigma.exp()
        # Imagewise loss. Calculated as the p-norm distance in pixel-space between original and reconstructed image
        image_loss = ((x_hat - x).abs()**self.p_norm).sum(axis=[1,2,3])

        # KL-divergence calculated explicitly
        # Reference Kingma & Welling p. 5 bottom
        kl = - (.5 * (1 + (qz_sigma ** 2).log() - qz_mu ** 2 - qz_sigma**2)).sum(axis=[1])

        # Image-wise beta-elbo:
        beta_elbo = -image_loss - self.beta * kl

        # Loss is the mean of the imagewise losses, over the full batch of images
        loss = -beta_elbo.mean()

        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': beta_elbo, 'image_loss': image_loss, 'kl': kl}
            
        return loss, diagnostics, outputs


def train_VAE(training_data, validation_data, epochs=500, batch_size=32, lr=1e-3, weight_decay=1e-4, epoch_sample_times=10):
    run_name = "VAE_predictor"
    make_training_folders(run_name)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    
    vae = CytoVariationalAutoencoder(input_shape=np.array([3, 64, 64]), latent_features=256).to(device)
    vi = VariationalInference_VAE().to(device)
    
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=weight_decay)
    
    # prepare dataset loader
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

    training_result = {}
    training_result["train_loss"] = []      # (elbo, image loss, kl)
    training_result["validation_loss"] = [] # (elbo, image loss, kl)
    
    sample_index = 0
    epoch_sample_points = torch.linspace(1, epochs, epoch_sample_times, dtype=torch.int32)
    
    for epoch in range(1, epochs+1):
        logging.info(f"Starting epoch {epoch}:")
        
        training_epoch_data = defaultdict(list)
        vae.train()
        
        for images in train_dataloader:
            images = images.to(device)

            loss, diagnostics, outputs = vi(vae, images)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 10_000)
            optimizer.step()
            
            for k, v in diagnostics.items():
                training_epoch_data[k] += list(v.cpu().data.numpy())
        
        training_data = {}
        for k, v in training_epoch_data.items():
            training_data[k] = np.mean(training_epoch_data[k])
        
        training_result["train_loss"].append(training_data)
        
        with torch.no_grad():
            vae.eval()
            
            validation_epoch_data = defaultdict(list)
            
            for images in validation_dataloader:
                images = images.to(device)
                
                loss, diagnostics, outputs = vi(vae, images)
                
                for k, v in diagnostics.items():
                    validation_epoch_data[k] += list(v.cpu().data.numpy())
            
            validation_data = {}
            for k, v in diagnostics.items():
                validation_data[k] = np.mean(validation_epoch_data[k])
            
            training_result["validation_loss"].append(training_data)
            save_dict(training_result, os.path.join("results", run_name, "training", "train_results.pkl"))  

            if epoch == epoch_sample_points[sample_index]:
                sample_index += 1
                
                torch.save(vae.state_dict(), os.path.join("results", run_name, "weights", f"ckpt{epoch}.pt"))
            

    torch.save(vae.state_dict(), os.path.join("results", run_name, "weights", "ckpt.pt"))
    #return validation_data, training_data, params, vae


