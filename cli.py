# anomaly_detection/cli.py
import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
import logging
import numpy as np
import sys
import os

from .data_loader import MVTecDataset
from .vae_model import VAE, train_vae, compute_anomaly_score_vae
from .gan_model import Generator, Discriminator, train_gan, compute_anomaly_score_gan
from .vae_gan_model import Encoder, Decoder, VAEGANDiscriminator, train_vae_gan, compute_anomaly_score_vae_gan
from .patchcore_inference import test_patchcore

logger = logging.getLogger("anomaly_detection")


def get_transform():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def denormalize(img_tensor):
    # input torch tensor in [-1,1] or a tensor normalized with mean=0.5,std=0.5
    # convert to numpy HWC in [0,1]
    img = img_tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    # if values in [-1,1]
    if img.min() < 0.0:
        img = (img + 1.0) / 2.0
    # clip
    return np.clip(img, 0.0, 1.0)


def find_model_file(dir_path: Path, candidates):
    for c in candidates:
        p = dir_path / c
        if p.exists():
            return p
    return None


def test_model_images(test_loader, score_fn, model_objects, device, model_dir: Path, no_display: bool, max_images=5):
    """
    test_loader: DataLoader yielding images (tensor normalized in [-1,1])
    score_fn: function returning (score, recon_tensor) when called with model_objects and img
    model_objects: either single model or tuple passed to score_fn
    """
    # headless handling: if no_display, use Agg and save to disk
    if no_display:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir = model_dir / "test_outputs"
    if no_display:
        out_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(test_loader):
        if i >= max_images:
            break
        img = img.to(device)
        # score_fn must accept (model_objects, img, device) or (model_obj, img)
        try:
            score, recon = score_fn(model_objects, img, device) if model_objects is not None else score_fn(img)
        except TypeError:
            # fallback: score_fn(model_obj, img)
            score, recon = score_fn(model_objects, img)

        img_np = denormalize(img)
        recon_np = denormalize(recon)
        error_map = np.abs(img_np - recon_np).mean(axis=2)

        print(f"[Test] Image {i+1} - Score anomalie : {score:.4f}")

        plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(img_np)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(recon_np)
        plt.title("Reconstruction")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(error_map, cmap="hot")
        plt.title("Carte erreur")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis("off")

        plt.tight_layout()
        if no_display:
            save_path = out_dir / f"test_{i+1:02d}.png"
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"[Test] visualisation sauvegardée -> {save_path}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(prog="anomaly_detection")
    parser.add_argument("--mode", choices=["train", "test"], required=True, help="Mode: train or test")
    parser.add_argument("--model", choices=["vae", "gan", "vae-gan", "patchcore"], required=True, help="Quel modèle")
    parser.add_argument("--data-dir", required=True, help="Chemin vers le dossier mvtec (ex: .../transistor)")
    parser.add_argument("--model-dir", default="./models", help="Dossier pour charger/sauver les modèles")
    parser.add_argument("--epochs", type=int, default=100, help="Nombre d'epochs pour l'entraînement")
    parser.add_argument("--batch-size", type=int, default=16, help="Taille des batches")
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"], help="Forcer device: cpu ou cuda ou auto (None)")
    parser.add_argument("--no-display", action="store_true", help="Ne pas afficher les plots (utile en docker/headless)")
    parser.add_argument("--max-test-images", type=int, default=5, help="Nombre d'images de test à afficher/sauvegarder")
    args = parser.parse_args()

    # device selection
    use_cuda = (args.device is None and torch.cuda.is_available()) or args.device == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Device sélectionné: {device}")

    model_dir = Path(args.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    transform = get_transform()

    # datasets & loaders
    train_ds = MVTecDataset(args.data_dir, mode="train", transform=transform)
    test_ds = MVTecDataset(args.data_dir, mode="test", transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # -----------------------
    # TRAIN
    # -----------------------
    if args.mode == "train":
        if args.model == "vae":
            vae = VAE().to(device)
            train_vae(vae, train_loader, device, num_epochs=args.epochs)
            torch.save(vae.state_dict(), model_dir / "vae.pth")
            print(f"[Train] VAE sauvegardé -> {model_dir / 'vae.pth'}")

        elif args.model == "gan":
            gen = Generator().to(device)
            disc = Discriminator().to(device)
            train_gan(gen, disc, train_loader, device, num_epochs=args.epochs)
            # save with multiple possible names to be compatible
            torch.save(gen.state_dict(), model_dir / "gan_generator.pth")
            torch.save(disc.state_dict(), model_dir / "gan_discriminator.pth")
            print(f"[Train] GAN saved -> {model_dir}")

        elif args.model == "vae-gan":
            enc = Encoder(128).to(device)
            dec = Decoder(128).to(device)
            disc = VAEGANDiscriminator().to(device)
            train_vae_gan(enc, dec, disc, train_loader, device, num_epochs=args.epochs)
            torch.save(enc.state_dict(), model_dir / "vae_gan_encoder.pth")
            torch.save(dec.state_dict(), model_dir / "vae_gan_decoder.pth")
            torch.save(disc.state_dict(), model_dir / "vae_gan_discriminator.pth")
            print(f"[Train] VAE-GAN saved -> {model_dir}")

        elif args.model == "patchcore":
            logger.info("PatchCore: pas d'entraînement explicite (utilisez --mode test pour évaluer).")

        return

    # -----------------------
    # TEST
    # -----------------------
    if args.mode == "test":
        if args.model == "patchcore":
            # PatchCore evaluation uses its own dataset loading internally
            test_patchcore(args.data_dir)
            return

        # For other models: load model weights then perform per-image test using compute_anomaly_score_*
        if args.model == "vae":
            vae_path = find_model_file(model_dir, ["vae.pth", "vae_model.pth"])
            if vae_path is None:
                print(f"[Error] fichier VAE introuvable dans {model_dir} (cherche: vae.pth).")
                sys.exit(2)
            vae = VAE().to(device)
            vae.load_state_dict(torch.load(vae_path, map_location=device))
            vae.eval()
            # create wrapper score function that matches test_model_images signature
            def score_fn(model_obj, img, device=device):
                return compute_anomaly_score_vae(model_obj, img)
            test_model_images(test_loader, score_fn, vae, device, model_dir, args.no_display, max_images=args.max_test_images)
            return

        elif args.model == "gan":
            # try multiple filenames (compatibilité)
            gen_path = find_model_file(model_dir, ["gan_generator.pth", "gan_gen.pth", "gan_generator.pt", "gan_generator"])
            disc_path = find_model_file(model_dir, ["gan_discriminator.pth", "gan_disc.pth", "gan_discriminator.pt", "gan_discriminator"])
            if gen_path is None or disc_path is None:
                print(f"[Error] fichiers du GAN introuvables dans {model_dir}. Cherché: gan_generator.pth & gan_discriminator.pth")
                sys.exit(2)
            generator = Generator().to(device)
            discriminator = Discriminator().to(device)
            generator.load_state_dict(torch.load(gen_path, map_location=device))
            discriminator.load_state_dict(torch.load(disc_path, map_location=device))
            generator.eval()
            discriminator.eval()

            # score_fn for GAN: compute_anomaly_score_gan(generator, img) signature in module returns (loss, recon)
            def score_fn(models_tuple, img, device=device):
                gen = models_tuple
                return compute_anomaly_score_gan(gen, img)

            test_model_images(test_loader, score_fn, generator, device, model_dir, args.no_display, max_images=args.max_test_images)
            return

        elif args.model == "vae-gan":
            enc_path = find_model_file(model_dir, ["vae_gan_encoder.pth", "vae_gan_encoder.pt"])
            dec_path = find_model_file(model_dir, ["vae_gan_decoder.pth", "vae_gan_decoder.pt"])
            disc_path = find_model_file(model_dir, ["vae_gan_discriminator.pth", "vae_gan_discriminator.pt"])
            if enc_path is None or dec_path is None or disc_path is None:
                print(f"[Error] fichiers VAE-GAN introuvables dans {model_dir}. Cherché: vae_gan_encoder / decoder / discriminator")
                sys.exit(2)
            encoder = Encoder(128).to(device)
            decoder = Decoder(128).to(device)
            discriminator = VAEGANDiscriminator().to(device)
            encoder.load_state_dict(torch.load(enc_path, map_location=device))
            decoder.load_state_dict(torch.load(dec_path, map_location=device))
            discriminator.load_state_dict(torch.load(disc_path, map_location=device))
            encoder.eval()
            decoder.eval()
            discriminator.eval()

            def score_fn(models_tuple, img, device=device):
                enc, dec = models_tuple
                return compute_anomaly_score_vae_gan(enc, dec, img, device)
            test_model_images(test_loader, score_fn, (encoder, decoder), device, model_dir, args.no_display, max_images=args.max_test_images)
            return

        else:
            print("[Error] modèle de test non supporté.")
            sys.exit(2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
