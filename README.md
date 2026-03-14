# anomaly_detection

## Présentation

Ce dépôt contient un programme Python `anomaly_detection` et une CLI pour **entraîner** et **tester** des modèles d’anomalie image (VAE, GAN, VAE-GAN) ainsi qu’un script d’inférence PatchCore. Il est conçu pour être dockerisable et simple à adapter : vous pouvez améliorer les architectures, changer les hyperparamètres, ou remplacer le dataset MVTec par n’importe quel autre jeu d’images en adaptant le dataset loader.

---

## Structure recommandée du projet

```
anomaly_detection/                # racine du projet
├─ anomaly_detection/             # package python
│  ├─ __init__.py
│  ├─ cli.py
│  ├─ data_loader.py
│  ├─ vae_model.py
│  ├─ gan_model.py
│  ├─ vae_gan_model.py
│  ├─ patchcore_inference.py
│  └─ ...
├─ models/                        # modèles sauvegardés (.pth)
├─ data/                          # dataset local
├─ docker/                        # Dockerfile
├─ tests/                         # tests unitaires (pytest)
├─ requirements.txt
├─ setup.py
├─ config.yaml
└─ README.md
```

---

## Prérequis

* Python 3.8+ (recommandé 3.10)
* virtualenv / venv ou conda
* pip
* Dépendances Python listées dans `requirements.txt` (torch, torchvision, pillow, matplotlib, lpips, faiss-cpu, tqdm, numpy, pyyaml, etc.)
* (Pour GPU) CUDA compatible et wheels torch/faiss correspondant à la version CUDA

---

## Installation (locale / dev)

```bash
# depuis la racine du projet
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

# installer dépendances
pip install -r requirements.txt

# ou installer le package en mode editable
pip install -e .
```

---

## Structure attendue des données

Le loader `MVTecDataset` attend un dossier au format MVTec ou similaire :

```
BASE_DIR/
  train/
    good/
      img_0001.png
      ...
  test/
    class_1/
      img_a.png
      ...
    class_2/
      ...
```

Si votre dataset a une autre structure :

* modifiez `anomaly_detection/data_loader.py` (classe `MVTecDataset`) pour lire vos chemins, ou
* fournissez un nouveau `Dataset` compatible PyTorch et passez-le dans le CLI.

---

## Commandes d’utilisation (CLI)

Exemples depuis la racine (PowerShell) :

* Entraîner un VAE :

```powershell
python -m anomaly_detection.cli --mode train --model vae --data-dir "C:\chemin\vers\transistor" --model-dir ".\models" --epochs 50 --batch-size 16
```

* Tester un GAN (headless, sauvegarde des visuels dans `models/test_outputs`) :

```powershell
python -m anomaly_detection.cli --mode test --model gan --data-dir "C:\chemin\vers\transistor" --model-dir ".\models" --no-display --max-test-images 10
```

* Tester PatchCore :

```powershell
python -m anomaly_detection.cli --mode test --model patchcore --data-dir "C:\chemin\vers\transistor"
```

Options utiles :

* `--device cpu` ou `--device cuda` (ou laisser auto)
* `--no-display` : pas d’affichage, sauvegarde des figures
* `--max-test-images N` : nombre d’images test à afficher/enregistrer

---

## Noms de fichiers modèles attendus

Par convention le CLI cherche plusieurs variantes ; exemples attendus :

* GAN : `gan_generator.pth` et `gan_discriminator.pth`
* VAE : `vae.pth`
* VAE-GAN : `vae_gan_encoder.pth`, `vae_gan_decoder.pth`, `vae_gan_discriminator.pth`

Si vos fichiers portent d’autres noms, soit renommez-les, soit modifiez la logique `find_model_file` dans `cli.py`.

---

## Comment interpréter les scores d’anomalie

* Les implémentations actuelles (VAE / GAN / VAE-GAN) calculent une erreur de reconstruction (MSE ou fonction définie) : score faible → image normale ; score élevé → anomalie probable.
* Il faut définir un **seuil** (calibrer sur un split validation) pour convertir score continu → décision binaire (anomalie oui/non).
* Pour PatchCore, le score provient d’une distance FAISS (L2 avg des voisins), la logique de seuil est similaire mais dépend des embeddings.
