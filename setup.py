# setup.py
from setuptools import setup, find_packages

setup(
    name="anomaly_detection",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "pillow",
        "matplotlib",
        "lpips",
        "faiss-cpu",
        "tqdm",
        "numpy",
        "pyyaml"
    ],
)