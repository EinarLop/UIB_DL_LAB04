import logging
import math
import os
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

###########################
# Classes
###########################

class EMA:
    def __init__(self, beta):
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


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
            nn.Linear(
                emb_dim,
                out_channels
            ),
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
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


###########################
# Diffusion
###########################

class Diffusion:
    """Linear noise schedule diffusion model (DDPM).

    Implements the forward and reverse diffusion processes with a linear
    beta schedule, as described in Ho et al. (2020).
    """

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02,
                 img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        # Sample from [1, noise_steps-1]; t=0 represents the clean image
        # and is not used during training (DDPM convention)
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def denoise_step(self, x, t_index, predicted_noise):
        """Perform one reverse diffusion step (DDPM sampling formula).

        Args:
            x: Current noisy image tensor.
            t_index: Integer timestep index.
            predicted_noise: Model's noise prediction for this timestep.

        Returns:
            Denoised tensor after one step.
        """
        t = torch.tensor([t_index], device=x.device)
        alpha = self.alpha[t][:, None, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        beta = self.beta[t][:, None, None, None]
        if t_index > 1:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        return (1 / torch.sqrt(alpha)
                * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat))
                   * predicted_noise)
                + torch.sqrt(beta) * noise)

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                x = self.denoise_step(x, i, predicted_noise)
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


class CosineDiffusion(Diffusion):
    """Cosine noise schedule (Nichol & Dhariwal, 2021).

    Defines alpha_hat directly via a cosine function, producing betas that
    preserve more signal at early timesteps compared to the linear schedule.

    Note: beta_start and beta_end are accepted for API compatibility with
    Diffusion but are not used in the cosine schedule computation. Betas
    are clipped to [0, 0.999] as in the original paper.
    """

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02,
                 img_size=256, device="cuda", s=0.008):
        self.s = s
        super().__init__(noise_steps, beta_start, beta_end, img_size, device)

    def prepare_noise_schedule(self):
        steps = self.noise_steps
        t = torch.arange(steps + 1) / steps
        alphas_cumprod = torch.cos((t + self.s) / (1 + self.s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        # Clip to [0, 0.999] as in Nichol & Dhariwal (2021), not to
        # [beta_start, beta_end] which would destroy the cosine shape
        return torch.clip(betas, 0.0, 0.999)


###########################
# Dataset utilities
###########################

class CustomDataset(Dataset):
    """Dataset for flat folder structures (no ImageFolder subdirectories).

    Recursively finds all image files (.png, .jpg, .jpeg) in the given
    folder path.
    """

    _IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

    def __init__(self, folder_path, transform=None, limit=-1):
        super().__init__()
        self.folder_path = Path(folder_path)
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {folder_path}")
        self.image_paths = sorted(
            p for p in self.folder_path.rglob('*')
            if p.suffix.lower() in self._IMAGE_EXTENSIONS
        )
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No image files found in {folder_path}")
        if limit > 0:
            self.image_paths = self.image_paths[:limit]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0


def extract_dataset(zip_file_path, remove_zip=True):
    """Extract a zip file containing a dataset.

    Args:
        zip_file_path (str): Path to the zip file.
        remove_zip (bool): Whether to remove the zip after extraction.

    Returns:
        str: Path to the extracted dataset folder.
    """
    dataset_folder = os.path.splitext(zip_file_path)[0]
    if os.path.exists(dataset_folder):
        print(f"Dataset folder already exists: {dataset_folder}")
        return dataset_folder

    if not os.path.exists(zip_file_path):
        raise FileNotFoundError("Zip file not found.")

    os.makedirs(dataset_folder, exist_ok=True)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        print('Extracting files...')
        zip_ref.extractall(os.path.dirname(zip_file_path))
        print('Extraction finished.')

    if not os.listdir(dataset_folder):
        raise RuntimeError(
            f"Extraction completed but {dataset_folder} is empty. "
            "The zip file's internal structure may not match expectations.")

    if remove_zip:
        os.remove(zip_file_path)
        print('Zip file removed.')

    return dataset_folder


def get_data_flat(dataset_path, image_size=64, batch_size=4, limit=-1):
    """Create a DataLoader from a flat folder of images.

    Unlike ``get_data`` (which requires ImageFolder subdirectories), this
    function uses ``CustomDataset`` to load images from a flat directory.

    Args:
        dataset_path: Path to folder containing images.
        image_size: Target image size (default 64).
        batch_size: Batch size (default 4).
        limit: Max number of images to load (-1 for all).

    Returns:
        DataLoader yielding (image_tensor, label_string) tuples.
    """
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(int(image_size + image_size / 4)),
        torchvision.transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CustomDataset(dataset_path, transform=transforms, limit=limit)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


###########################
# Visualization utilities
###########################

def visualize_forward_diffusion(diffusion, dataloader, device, n_steps=10):
    """Visualize the forward diffusion process on a single image.

    Shows ``n_steps`` images from the original (t=0) to pure noise
    (t=noise_steps-1).

    Args:
        diffusion: A Diffusion instance.
        dataloader: DataLoader yielding image batches.
        device: Torch device.
        n_steps: Number of intermediate steps to display (default 10).
    """
    images, _ = next(iter(dataloader))
    x0 = images[0].unsqueeze(0).to(device)

    steps = torch.linspace(0, diffusion.noise_steps - 1, n_steps).long()

    noisy_images = []
    with torch.no_grad():
        for t in steps:
            x_t, _ = diffusion.noise_images(x0, torch.tensor([t], device=device))
            noisy_images.append(x_t.cpu())

    plt.figure(figsize=(2 * n_steps, 3))
    for i, img in enumerate(noisy_images):
        plt.subplot(1, n_steps, i + 1)
        plt.imshow((img.squeeze().permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1))
        plt.title(f"t={steps[i].item()}")
        plt.axis('off')
    plt.suptitle("Forward Diffusion Process", y=0.78)
    plt.tight_layout()
    plt.show()


def visualize_reverse_diffusion(model, diffusion, device, n_steps=10):
    """Visualize the reverse (denoising) diffusion process.

    Runs the full reverse loop from pure noise to a generated image,
    capturing ``n_steps`` intermediate results.

    Args:
        model: Trained UNet model.
        diffusion: A Diffusion instance.
        device: Torch device.
        n_steps: Number of intermediate steps to display (default 10).
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        x = torch.randn((1, 3, diffusion.img_size, diffusion.img_size)).to(device)

        # Determine which timesteps to capture
        capture_at = set(
            torch.linspace(diffusion.noise_steps - 1, 0, n_steps).long().tolist()
        )
        captured = {diffusion.noise_steps - 1: x.cpu().clone()}

        for i in tqdm(reversed(range(1, diffusion.noise_steps)),
                      desc="Denoising", position=0):
            t = torch.tensor([i], device=device)
            predicted_noise = model(x, t)
            x = diffusion.denoise_step(x, i, predicted_noise)
            if i in capture_at:
                captured[i] = x.cpu().clone()

        captured[0] = x.cpu().clone()

    # Sort by timestep descending and take exactly n_steps
    sorted_steps = sorted(captured.keys(), reverse=True)[:n_steps]

    plt.figure(figsize=(2 * len(sorted_steps), 3))
    for idx, ts in enumerate(sorted_steps):
        plt.subplot(1, len(sorted_steps), idx + 1)
        img_np = (captured[ts].squeeze().permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
        plt.imshow(img_np)
        plt.title(f"t={ts}")
        plt.axis('off')
    plt.suptitle("Reverse Diffusion Process", y=0.78)
    plt.tight_layout()
    plt.show()
    if was_training:
        model.train()


#####################
# Helper functions
#####################

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(int(args.image_size + args.image_size / 4)),
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def test_sigmoid_diffusion(student_diffusion_class, plot=True):
    """
    Tests if a SigmoidDiffusion class is implemented correctly.

    Args:
        student_diffusion_class: The student's SigmoidDiffusion class
        plot: Whether to plot the noise schedule for visualization

    Returns:
        dict: Test results with pass/fail status and error messages
    """
    results = {
        "passed": True,
        "errors": []
    }

    # Test parameters
    noise_steps = 1000
    beta_start = 1e-4
    beta_end = 0.02
    sigmoid_scale = 8.0
    img_size = 64
    device = "cpu"  # Use CPU for testing

    try:
        # Instantiate the student's class
        diffusion = student_diffusion_class(
            noise_steps=noise_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            sigmoid_scale=sigmoid_scale,
            img_size=img_size,
            device=device
        )

        # Check if beta attribute exists and has correct shape
        if not hasattr(diffusion, 'beta'):
            results["passed"] = False
            results["errors"].append("Missing 'beta' attribute in diffusion object")
        elif diffusion.beta.shape != torch.Size([noise_steps]):
            results["passed"] = False
            results["errors"].append(f"Beta has wrong shape: {diffusion.beta.shape}, expected: {torch.Size([noise_steps])}")

        # Check if alpha and alpha_hat attributes exist
        if not hasattr(diffusion, 'alpha'):
            results["passed"] = False
            results["errors"].append("Missing 'alpha' attribute in diffusion object")
        if not hasattr(diffusion, 'alpha_hat'):
            results["passed"] = False
            results["errors"].append("Missing 'alpha_hat' attribute in diffusion object")

        # Check if beta values are within expected range
        if hasattr(diffusion, 'beta'):
            if torch.min(diffusion.beta) < beta_start * 0.9:  # Allow small numerical errors
                results["passed"] = False
                results["errors"].append(f"Minimum beta value {torch.min(diffusion.beta).item()} is less than beta_start {beta_start}")
            if torch.max(diffusion.beta) > beta_end * 1.1:  # Allow small numerical errors
                results["passed"] = False
                results["errors"].append(f"Maximum beta value {torch.max(diffusion.beta).item()} is greater than beta_end {beta_end}")

        # Check sigmoid shape properties
        if hasattr(diffusion, 'beta'):
            # Calculate first and second derivatives
            first_diff = diffusion.beta[1:] - diffusion.beta[:-1]
            second_diff = first_diff[1:] - first_diff[:-1]

            # Check if first half of second derivatives are mostly positive (concave up)
            # and second half are mostly negative (concave down) - characteristic of sigmoid
            mid_point = len(second_diff) // 2
            first_half_positive = torch.sum(second_diff[:mid_point] > 0).item()
            second_half_negative = torch.sum(second_diff[mid_point:] < 0).item()

            if first_half_positive < mid_point * 0.7:  # At least 70% should be positive
                results["passed"] = False
                results["errors"].append("First half of beta schedule doesn't show sigmoid's concave up characteristic")

            if second_half_negative < (len(second_diff) - mid_point) * 0.7:  # At least 70% should be negative
                results["passed"] = False
                results["errors"].append("Second half of beta schedule doesn't show sigmoid's concave down characteristic")

        # Test noise_images method with a sample image
        try:
            x = torch.randn(4, 3, img_size, img_size).to(device)
            t = torch.tensor([100, 200, 300, 400]).to(device)
            noised_x, noise = diffusion.noise_images(x, t)

            if noised_x.shape != x.shape:
                results["passed"] = False
                results["errors"].append(f"noise_images returned wrong shape: {noised_x.shape}, expected: {x.shape}")
        except Exception as e:
            results["passed"] = False
            results["errors"].append(f"Error in noise_images method: {str(e)}")

        # Plot the noise schedule if requested
        if plot and hasattr(diffusion, 'beta'):
            plt.figure(figsize=(12, 8))

            # Plot beta values
            plt.subplot(2, 2, 1)
            plt.plot(diffusion.beta.cpu().numpy())
            plt.title('Beta Schedule')
            plt.xlabel('Timestep')
            plt.ylabel('Beta Value')

            # Plot alpha_hat values
            plt.subplot(2, 2, 2)
            plt.plot(diffusion.alpha_hat.cpu().numpy())
            plt.title('Alpha Hat (Cumulative Product)')
            plt.xlabel('Timestep')
            plt.ylabel('Alpha Hat Value')

            # Plot first derivative
            plt.subplot(2, 2, 3)
            plt.plot(first_diff.cpu().numpy())
            plt.title('First Derivative of Beta')
            plt.xlabel('Timestep')
            plt.ylabel('Rate of Change')

            # Plot second derivative
            plt.subplot(2, 2, 4)
            plt.plot(second_diff.cpu().numpy())
            plt.title('Second Derivative of Beta')
            plt.xlabel('Timestep')
            plt.ylabel('Curvature')

            plt.tight_layout()
            plt.show()

            # Also create a comparison with linear schedule
            t = torch.linspace(0, 1, noise_steps)
            linear_beta = beta_start + (beta_end - beta_start) * t

            plt.figure(figsize=(10, 6))
            plt.plot(linear_beta.numpy(), label='Linear Schedule')
            plt.plot(diffusion.beta.cpu().numpy(), label='Sigmoid Schedule')
            plt.title('Comparison of Beta Schedules')
            plt.xlabel('Timestep')
            plt.ylabel('Beta Value')
            plt.legend()
            plt.grid(True)
            plt.show()

    except Exception as e:
        results["passed"] = False
        results["errors"].append(f"Unexpected error: {str(e)}")

    # Print summary
    if results["passed"]:
        print("✅ All tests passed! The SigmoidDiffusion implementation is correct.")
    else:
        error_msg = f"Tests failed with {len(results['errors'])} errors:\n"
        for i, error in enumerate(results["errors"]):
            error_msg += f"  {i+1}. {error}\n"
        print(f"❌ {error_msg}")
        raise AssertionError(error_msg)

    return results

def test_cosine_diffusion(student_diffusion_class, plot=True):
    """
    Tests if a CosineDiffusion class is implemented correctly.

    Args:
        student_diffusion_class: The student's CosineDiffusion class
        plot: Whether to plot the noise schedule for visualization

    Returns:
        dict: Test results with pass/fail status and error messages
    """
    results = {
        "passed": True,
        "errors": []
    }

    noise_steps = 1000
    beta_start = 1e-4
    beta_end = 0.02
    img_size = 64
    device = "cpu"

    try:
        diffusion = student_diffusion_class(
            noise_steps=noise_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            img_size=img_size,
            device=device
        )

        # Check beta attribute exists and shape
        if not hasattr(diffusion, 'beta'):
            results["passed"] = False
            results["errors"].append("Missing 'beta' attribute")
        elif diffusion.beta.shape != torch.Size([noise_steps]):
            results["passed"] = False
            results["errors"].append(
                f"Beta has wrong shape: {diffusion.beta.shape}, "
                f"expected: {torch.Size([noise_steps])}")

        # Check alpha and alpha_hat
        if not hasattr(diffusion, 'alpha'):
            results["passed"] = False
            results["errors"].append("Missing 'alpha' attribute")
        if not hasattr(diffusion, 'alpha_hat'):
            results["passed"] = False
            results["errors"].append("Missing 'alpha_hat' attribute")

        # Check beta values are non-negative and bounded
        if hasattr(diffusion, 'beta'):
            if torch.min(diffusion.beta) < 0:
                results["passed"] = False
                results["errors"].append(
                    f"Min beta {torch.min(diffusion.beta).item():.6f} < 0")
            if torch.max(diffusion.beta) > 1.0:
                results["passed"] = False
                results["errors"].append(
                    f"Max beta {torch.max(diffusion.beta).item():.6f} > 1.0")

        # alpha_hat should be monotonically decreasing
        if hasattr(diffusion, 'alpha_hat'):
            ah_diffs = diffusion.alpha_hat[1:] - diffusion.alpha_hat[:-1]
            n_increasing = (ah_diffs > 1e-8).sum().item()
            if n_increasing > noise_steps * 0.01:
                results["passed"] = False
                results["errors"].append(
                    f"alpha_hat is not monotonically decreasing "
                    f"({n_increasing} increasing steps)")

        # Verify cosine shape: alpha_hat should correlate with reference
        if hasattr(diffusion, 'alpha_hat'):
            s = 0.008
            ref_t = torch.arange(noise_steps + 1) / noise_steps
            ref_ah = torch.cos((ref_t + s) / (1 + s) * math.pi / 2) ** 2
            ref_ah = ref_ah / ref_ah[0]
            ref_ah = ref_ah[1:]  # Drop t=0 to match noise_steps length
            corr = torch.corrcoef(
                torch.stack([diffusion.alpha_hat.cpu().float(),
                             ref_ah.float()])
            )[0, 1]
            if corr < 0.99:
                results["passed"] = False
                results["errors"].append(
                    f"alpha_hat does not follow cosine shape "
                    f"(correlation={corr.item():.4f}, expected > 0.99)")

        # Test noise_images method
        try:
            x = torch.randn(4, 3, img_size, img_size).to(device)
            t = torch.tensor([100, 200, 300, 400]).to(device)
            noised_x, noise = diffusion.noise_images(x, t)
            if noised_x.shape != x.shape:
                results["passed"] = False
                results["errors"].append(
                    f"noise_images wrong shape: {noised_x.shape}, "
                    f"expected: {x.shape}")
        except Exception as e:
            results["passed"] = False
            results["errors"].append(f"Error in noise_images: {str(e)}")

        # Plot if requested
        if plot and hasattr(diffusion, 'beta'):
            linear_diff = Diffusion(noise_steps=noise_steps,
                                    beta_start=beta_start,
                                    beta_end=beta_end,
                                    img_size=img_size, device=device)

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for label, d in [("Linear", linear_diff), ("Cosine", diffusion)]:
                axes[0].plot(d.beta.cpu().numpy(), label=label)
                axes[1].plot(d.alpha_hat.cpu().numpy(), label=label)
                snr = d.alpha_hat / (1 - d.alpha_hat + 1e-8)
                axes[2].plot(torch.log(snr).cpu().numpy(), label=label)
            axes[0].set_title('Beta Schedule')
            axes[0].set_xlabel('Timestep')
            axes[0].legend()
            axes[1].set_title('Alpha Hat')
            axes[1].set_xlabel('Timestep')
            axes[1].legend()
            axes[2].set_title('log(SNR)')
            axes[2].set_xlabel('Timestep')
            axes[2].legend()
            plt.tight_layout()
            plt.show()

    except Exception as e:
        results["passed"] = False
        results["errors"].append(f"Unexpected error: {str(e)}")

    if results["passed"]:
        print("All tests passed! The CosineDiffusion implementation is correct.")
    else:
        error_msg = f"Tests failed with {len(results['errors'])} errors:\n"
        for i, error in enumerate(results["errors"]):
            error_msg += f"  {i+1}. {error}\n"
        print(f"Tests failed:\n{error_msg}")
        raise AssertionError(error_msg)

    return results
