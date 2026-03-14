from anomaly_detection.data_loader import MVTecDataset
from torchvision import transforms

def test_dataset_len(tmp_path):
    d = tmp_path/"train"/"good"
    d.mkdir(parents=True)
    (d/"img1.png").write_bytes(b"not_an_image")
    ds = MVTecDataset(tmp_path, mode="train", transform=transforms.ToTensor())
    assert len(ds) == 1
