import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
import numpy as np

# -------------------------
# Modules
# -------------------------

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # (64, 64, 64)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # (128, 32, 32)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),  # (256, 16, 16)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # (128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # (64, 64, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),  # (3, 128, 128)
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 16, 16)
        return self.decoder(x)


class VAEGANDiscriminator(nn.Module):
    def __init__(self):
        super(VAEGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # (64, 64, 64)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # (128, 32, 32)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),  # (256, 16, 16)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# -------------------------
# VAE-GAN Entraînement
# -------------------------

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def train_vae_gan(encoder, decoder, discriminator, dataloader, device, num_epochs=50, lr=1e-4):
    encoder.train()
    decoder.train()
    discriminator.train()

    opt_enc_dec = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr)

    bce = nn.BCELoss()
    recon_loss_fn = nn.MSELoss()

    loss_init = None

    for epoch in range(num_epochs):
        total_loss = 0
        for imgs in dataloader:
            imgs = imgs.to(device)

            mu, logvar = encoder(imgs)
            z = reparameterize(mu, logvar)
            recon = decoder(z)

            recon_loss = recon_loss_fn(recon, imgs)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            real_labels = torch.ones(imgs.size(0), 1).to(device)
            fake_labels = torch.zeros(imgs.size(0), 1).to(device)

            disc_real = discriminator(imgs)
            disc_fake = discriminator(recon.detach())

            loss_disc = bce(disc_real, real_labels) + bce(disc_fake, fake_labels)
            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            disc_fake = discriminator(recon)
            gen_adv_loss = bce(disc_fake, real_labels)

            total_vae_gan_loss = recon_loss + kl_loss + 1e-3 * gen_adv_loss

            opt_enc_dec.zero_grad()
            total_vae_gan_loss.backward()
            opt_enc_dec.step()

            total_loss += total_vae_gan_loss.item()

        avg_loss = total_loss / len(dataloader)
        if loss_init is None:
            loss_init = avg_loss

        loss_percent = (avg_loss / loss_init) * 100  # % par rapport à la 1ère epoch

        print(f"[Epoch {epoch+1}/{num_epochs}] VAE-GAN Loss: {avg_loss:.4f} ({loss_percent:.1f} % de la loss initiale)")


# -------------------------
# Score d’anomalie
# -------------------------

def compute_anomaly_score_vae_gan(encoder, decoder, img, device):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        mu, logvar = encoder(img)
        z = reparameterize(mu, logvar)
        recon = decoder(z)
        loss = F.mse_loss(recon, img).item()
    return loss, recon
