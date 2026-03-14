import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import faiss
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2
from tqdm import tqdm

from .data_loader import MVTecDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_features(model, dataloader):
    model.eval()
    features = []

    with torch.no_grad():
        for img in tqdm(dataloader, desc="Extraction des features"):
            img = img.to(device)
            feat = model(img).squeeze()
            features.append(feat.cpu().numpy())

    return np.vstack(features)


def test_patchcore(base_dir):
    print("\n=== Évaluation avec PatchCore ===")

    # Chargement des datasets via MVTecDataset (base_dir passé ici)
    train_dataset = MVTecDataset(base_dir=base_dir, mode="train")
    test_dataset = MVTecDataset(base_dir=base_dir, mode="test")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Modèle backbone (WideResNet)
    backbone = wide_resnet50_2(pretrained=True).to(device)
    backbone.fc = torch.nn.Identity()  # on retire la couche FC pour obtenir les features

    print(">> Extraction des features d'entraînement...")
    train_feats = extract_features(backbone, train_loader)

    print(">> Construction de l'index FAISS...")
    dim = train_feats.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(train_feats.astype(np.float32))

    print(">> Évaluation des images test...")
    for i, img in enumerate(test_loader):
        img = img.to(device)

        with torch.no_grad():
            feat = backbone(img).squeeze().cpu().numpy().reshape(1, -1)

        D, _ = index.search(feat.astype(np.float32), k=5)
        score = D.mean()

        print(f"Image {i+1} - Score anomalie : {score:.4f} - Anomalie détectée ? {'OUI' if score > 300.0 else 'NON'}")
