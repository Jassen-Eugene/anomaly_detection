import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

import lpips

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512 * 8 * 8),
            nn.ReLU(True),
            nn.Unflatten(1, (512, 8, 8)),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()  # IMPORTANT : sortie entre -1 et 1
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1)
        )

    def forward(self, x):
        return self.model(x)

def hinge_d_loss(real_preds, fake_preds):
    loss_real = torch.mean(F.relu(1. - real_preds))
    loss_fake = torch.mean(F.relu(1. + fake_preds))
    return loss_real + loss_fake

def hinge_g_loss(fake_preds):
    return -torch.mean(fake_preds)

def show_generated_images(images, epoch):
    # images entre -1 et 1 -> on remet à 0-1 pour affichage
    images = (images + 1) / 2
    grid = make_grid(images.cpu(), nrow=4, normalize=False)
    plt.figure(figsize=(8,8))
    plt.imshow(grid.permute(1,2,0))
    plt.title(f'Images générées à l\'epoch {epoch}')
    plt.axis('off')
    plt.show()

def train_gan(generator, discriminator, dataloader, device, num_epochs=100, lr=1e-4, patience=10, display_interval=10):
    loss_fn_lpips = lpips.LPIPS(net='alex').to(device)
    
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    best_lpips = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, num_epochs+1):
        generator.train()
        discriminator.train()

        total_d_loss = 0
        total_g_loss = 0
        total_lpips = 0
        count = 0

        for imgs in dataloader:
            # imgs attendus dans [-1, 1], assure-toi que ton dataset soit normalisé ainsi
            imgs = imgs.to(device)
            batch_size = imgs.size(0)

            # ====== Discriminator ======
            z = torch.randn(batch_size, generator.latent_dim, device=device)
            fake_imgs = generator(z)

            real_preds = discriminator(imgs)
            fake_preds = discriminator(fake_imgs.detach())

            d_loss = hinge_d_loss(real_preds, fake_preds)
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # ====== Generator ======
            fake_preds = discriminator(fake_imgs)
            g_loss = hinge_g_loss(fake_preds)
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            # LPIPS (attend images dans [-1, 1])
            batch_lpips = loss_fn_lpips(fake_imgs, imgs).mean().item()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            total_lpips += batch_lpips
            count += 1

        avg_d_loss = total_d_loss / count
        avg_g_loss = total_g_loss / count
        avg_lpips = total_lpips / count

        print(f"Epoch {epoch}/{num_epochs} | D loss: {avg_d_loss:.4f} | G loss: {avg_g_loss:.4f} | LPIPS: {avg_lpips:.4f}")

        # Early stopping basé sur LPIPS
        if avg_lpips < best_lpips:
            best_lpips = avg_lpips
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping au bout de {epoch} epochs (aucune amélioration LPIPS depuis {patience} epochs).")
                break


def compute_anomaly_score_gan(generator, img):
    generator.eval()
    with torch.no_grad():
        batch_size = img.size(0)
        z = torch.randn(batch_size, generator.latent_dim, device=img.device)
        recon = generator(z)
        loss = torch.mean((img - recon) ** 2)
    return loss.item(), recon
