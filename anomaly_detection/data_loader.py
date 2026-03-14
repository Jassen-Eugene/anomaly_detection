# anomaly_detection/data_loader.py
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random

class MVTecDataset(Dataset):
    """
    Dataset simple pour la structure MVTec:
      base_dir/
        train/good/*.png
        test/<class>/*.png
    - en mode train: prend tous les fichiers de train/good
    - en mode test: pour chaque sous-dossier de test/, choisit up to test_sample_per_class images aléatoires (seedable)
    """
    def __init__(self, base_dir, mode="train", transform=None, test_sample_per_class=3, seed=0):
        self.base_dir = Path(base_dir)
        assert mode in ["train","test"], "mode must be 'train' or 'test'"
        self.mode = mode
        self.transform = transform
        self.img_paths = []
        random.seed(seed)

        if mode == "train":
            img_dir = self.base_dir / "train" / "good"
            if not img_dir.exists():
                raise FileNotFoundError(f"Train directory not found: {img_dir}")
            self.img_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        else:
            test_root = self.base_dir / "test"
            if not test_root.exists():
                raise FileNotFoundError(f"Test directory not found: {test_root}")
            for sub in sorted([d for d in test_root.iterdir() if d.is_dir()]):
                images = sorted([p for p in sub.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
                if images:
                    chosen = random.sample(images, min(test_sample_per_class, len(images)))
                    self.img_paths.extend(chosen)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        p = self.img_paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform is not None:
            return self.transform(img)
        return img