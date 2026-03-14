import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import random
import matplotlib.pyplot as plt

from .data_loader import MVTecDataset
from .vae_model import VAE, train_vae, compute_anomaly_score_vae
from .gan_model import Generator, Discriminator, train_gan, compute_anomaly_score_gan
from .vae_gan_model import Encoder, Decoder, VAEGANDiscriminator, train_vae_gan, compute_anomaly_score_vae_gan
from .patchcore_inference import test_patchcore  # Import PatchCore test function

# Fix seed pour reproductibilité
seed = 20
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configs
image_dir = r"C:\Users\maserati\Downloads\mvtec_anomaly_detection\transistor"
batch_size = 16
latent_dim = 100
num_epochs = 100
model_dir = "./models"
os.makedirs(model_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_dataset = MVTecDataset(image_dir, "train")
train_dataset.transform = transform
test_dataset = MVTecDataset(image_dir, "test")
test_dataset.transform = transform

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Modèles chemins
vae_path = os.path.join(model_dir, "vae.pth")
gan_gen_path = os.path.join(model_dir, "gan_generator.pth")
gan_disc_path = os.path.join(model_dir, "gan_discriminator.pth")
vae_gan_enc_path = os.path.join(model_dir, "vae_gan_encoder.pth")
vae_gan_dec_path = os.path.join(model_dir, "vae_gan_decoder.pth")
vae_gan_disc_path = os.path.join(model_dir, "vae_gan_discriminator.pth")

def models_exist(model_choice):
    if model_choice == "1":
        return os.path.exists(vae_path)
    elif model_choice == "2":
        return os.path.exists(gan_gen_path) and os.path.exists(gan_disc_path)
    elif model_choice == "3":
        return (os.path.exists(vae_gan_enc_path) and
                os.path.exists(vae_gan_dec_path) and
                os.path.exists(vae_gan_disc_path))
    # Pour PatchCore pas de modèle sauvegardé à vérifier ici
    elif model_choice == "4":
        return True
    return False

def denormalize(img):
    img = (img + 1) / 2
    return np.clip(img, 0, 1)

def test_model(model_choice):
    print("\nMode TEST activé.\n")

    if model_choice not in ["1", "2", "3", "4"]:
        print("Choix de modèle invalide pour test.")
        return

    if not models_exist(model_choice):
        print("Modèles pas trouvés, impossible de tester.")
        return

    # Seuils spécifiques par modèle
    thresholds = {"1": 350, "2": 0.1, "3": 0.1, "4": 0.3}  # Ajuste seuil PatchCore si besoin
    seuil = thresholds.get(model_choice, 150)

    if model_choice == "1":
        vae = VAE(latent_dim).to(device)
        vae.load_state_dict(torch.load(vae_path, map_location=device))
        vae.eval()

    elif model_choice == "2":
        generator = Generator(latent_dim).to(device)
        discriminator = Discriminator().to(device)
        generator.load_state_dict(torch.load(gan_gen_path, map_location=device))
        discriminator.load_state_dict(torch.load(gan_disc_path, map_location=device))
        generator.eval()
        discriminator.eval()

    elif model_choice == "3":
        encoder = Encoder(latent_dim).to(device)
        decoder = Decoder(latent_dim).to(device)
        discriminator = VAEGANDiscriminator().to(device)
        encoder.load_state_dict(torch.load(vae_gan_enc_path, map_location=device))
        decoder.load_state_dict(torch.load(vae_gan_dec_path, map_location=device))
        discriminator.load_state_dict(torch.load(vae_gan_disc_path, map_location=device))
        encoder.eval()
        decoder.eval()
        discriminator.eval()

    elif model_choice == "4":
        # Appel à la fonction PatchCore dédiée
        test_patchcore(image_dir)
        return

    for i, img in enumerate(test_loader):
        if i >= 5:
            break
        img = img.to(device)
        if model_choice == "1":
            score, recon = compute_anomaly_score_vae(vae, img)
        elif model_choice == "2":
            score, recon = compute_anomaly_score_gan(generator, img)
        else:
            score, recon = compute_anomaly_score_vae_gan(encoder, decoder, img, device)

        img_np = denormalize(img.cpu().squeeze().permute(1,2,0).numpy())
        recon_np = denormalize(recon.cpu().squeeze().permute(1,2,0).numpy())
        error_map = np.abs(img_np - recon_np).mean(axis=2)

        print(f"Image {i+1} - Score anomalie : {score:.4f} - Anomalie détectée ? {'OUI' if score > seuil else 'NON'}")

        plt.figure(figsize=(9,3))
        plt.subplot(1,3,1)
        plt.imshow(img_np)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.imshow(recon_np)
        plt.title("Reconstruction")
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.imshow(error_map, cmap="hot")
        plt.title("Carte erreur")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

def train_model(model_choice):
    if model_choice == "1":
        vae = VAE(latent_dim).to(device)
        print("Début entraînement VAE...")
        train_vae(vae, train_loader, device, num_epochs=num_epochs, patience=20)
        torch.save(vae.state_dict(), vae_path)
        print("Modèle VAE sauvegardé.")

    elif model_choice == "2":
        generator = Generator(latent_dim).to(device)
        discriminator = Discriminator().to(device)
        print("Début entraînement GAN...")
        train_gan(generator, discriminator, train_loader, device, num_epochs=num_epochs, patience=20)
        torch.save(generator.state_dict(), gan_gen_path)
        torch.save(discriminator.state_dict(), gan_disc_path)
        print("Modèles GAN sauvegardés.")

    elif model_choice == "3":
        encoder = Encoder(latent_dim).to(device)
        decoder = Decoder(latent_dim).to(device)
        discriminator = VAEGANDiscriminator().to(device)
        print("Début entraînement VAE-GAN...")
        train_vae_gan(encoder, decoder, discriminator, train_loader, device, num_epochs=num_epochs, lr=1e-4)
        torch.save(encoder.state_dict(), vae_gan_enc_path)
        torch.save(decoder.state_dict(), vae_gan_dec_path)
        torch.save(discriminator.state_dict(), vae_gan_disc_path)
        print("Modèles VAE-GAN sauvegardés.")

    elif model_choice == "4":
        print("PatchCore ne nécessite pas d'entraînement explicite.")

# --- Interaction utilisateur ---
while True:
    choix = input("Choisissez le modèle : (1) VAE / (2) GAN / (3) VAE-GAN / (4) PatchCore / (t) tester les modèles : ").strip().lower()

    if choix == "t":
        modele_test = input("Quel modèle tester ? (1) VAE / (2) GAN / (3) VAE-GAN / (4) PatchCore : ").strip()
        test_model(modele_test)
        break

    elif choix in ["1", "2", "3", "4"]:
        train_model(choix)
        break

    else:
        print("Choix invalide. Tapez 1, 2, 3, 4 ou t.")

print("Programme terminé.")
