import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # [B, 32, 64, 64]  (128/2)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # [B, 64, 32, 32]  (64/2)
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),# [B, 128, 16, 16] (32/2)
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),# [B, 256, 8, 8]  (16/2)
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 256 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # -> (128, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # -> (64, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # -> (32, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),     # -> (3, 128, 128)
            # Pas de Sigmoid ici
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x_encoded = self.encoder(x)
        mu = self.fc_mu(x_encoded)
        logvar = self.fc_logvar(x_encoded)
        z = self.reparameterize(mu, logvar)
        x_decoded = self.decoder_input(z)
        x_recon = self.decoder(x_decoded)
        return x_recon, mu, logvar


def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + KLD


def train_vae(model, dataloader, device, num_epochs=5, lr=1e-3, patience=3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        total_loss = 0
        for imgs in dataloader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recon_imgs, mu, logvar = model(imgs)
            loss = loss_function(recon_imgs, imgs, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        c, h, w = imgs.size(1), imgs.size(2), imgs.size(3)
        num_pixels = c * h * w

        avg_loss = total_loss / len(dataloader.dataset)
        loss_percent = (avg_loss / num_pixels) * 100

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_percent:.4f} %")

        if loss_percent < best_loss:
            best_loss = loss_percent
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹️  Early stopping after {epoch+1} epochs (no improvement for {patience} epochs).")
                break


def compute_anomaly_score_vae(model, img):
    model.eval()
    with torch.no_grad():
        recon_img, mu, logvar = model(img)
        loss = loss_function(recon_img, img, mu, logvar)
    return loss.item(), recon_img
