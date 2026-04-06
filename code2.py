import argparse
from pathlib import Path
import cv2
import numpy as np

SIZE = 512
CLIPLIMIT = 2.0
TILEGRID = (8, 8)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def LoadImage(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def ResizeImage(img):
    return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LANCZOS4)

def GreenChannel(img):
    return img[:, :, 1]

def Denoise(gray):
    smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.medianBlur(smoothed, 3)

def CircularMask(img):
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.circle(mask, (SIZE // 2, SIZE // 2), int(SIZE * 0.48), 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)

def Clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=CLIPLIMIT, tileGridSize=TILEGRID)
    return clahe.apply(gray)

def Preprocess(image_path):
    rgb = LoadImage(image_path)
    resized = ResizeImage(rgb)
    green = GreenChannel(resized)
    noise = Denoise(green)
    masked = CircularMask(noise)
    return Clahe(masked)

def ProcessImages(input_dir, output_dir):
    input_path = Path(input_dir)
    out_path = Path(output_dir)
    (out_path / "npy").mkdir(parents=True, exist_ok=True)
    (out_path / "images").mkdir(parents=True, exist_ok=True)

    images = []
    for ext in EXTENSIONS:
        images.extend(input_path.rglob(f"*{ext}"))

    print(f"{len(images)} images. Preprocessing")

    for img_p in images:
        try:
            processed = Preprocess(str(img_p))
            fname = img_p.stem
            cv2.imwrite(str(out_path / "images" / f"{fname}_prep.png"), processed)
            np.save(str(out_path / "npy" / f"{fname}.npy"), processed.astype(np.float32)/255.0)
            print(f"OK: {img_p.name}")
        except Exception as e:
            print(f"Error {img_p.name}: {e}")

if __name__ == "__main__":
    BASE = Path(__file__).parent

    splits = [
        ("partitioned_randomly/training_set/glaucoma", "output/training/glaucoma"),
        ("partitioned_randomly/training_set/normal",   "output/training/normal"),
        ("partitioned_randomly/test_set/glaucoma",     "output/test/glaucoma"),
        ("partitioned_randomly/test_set/normal",       "output/test/normal"),
    ]

    for input_rel, output_rel in splits:
        print(f"\n--- {input_rel} ---")
        ProcessImages(str(BASE / input_rel), str(BASE / output_rel))




import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from pathlib import Path

METHOD = 'uniform'
RADIUS = 1
N_POINTS = 8 * RADIUS

def ExtractLBP(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image introuvable : {image_path}")
    lbp = local_binary_pattern(img, N_POINTS, RADIUS, METHOD)
    n_bins = N_POINTS + 2
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def ProcessFeatures(input_dir, output_csv, label):
    input_path = Path(input_dir)
    images = list(input_path.glob("*_prep.png"))
    print(f"Extraction LBP sur {len(images)} images...")
    all_features = []
    for img_p in images:
        try:
            hist = ExtractLBP(str(img_p))
            row_data = {f"LBP_{i}": val for i, val in enumerate(hist)}
            row_data["Filename"] = img_p.name
            row_data["Label"] = label
            all_features.append(row_data)
        except Exception as e:
            print(f"Erreur avec {img_p.name}: {e}")
    df = pd.DataFrame(all_features)
    cols = [c for c in df.columns if c not in ['Filename', 'Label']] + ['Filename', 'Label']
    df = df[cols]
    df.to_csv(output_csv, index=False)
    print(f"Terminé ! Fichier sauvegardé : {output_csv}")

if __name__ == "__main__":
    BASE = Path(__file__).parent

    splits = [
        ("output/training/glaucoma/images", "output/features_train_glaucoma.csv", 1),
        ("output/training/normal/images",   "output/features_train_normal.csv",   0),
        ("output/test/glaucoma/images",     "output/features_test_glaucoma.csv",  1),
        ("output/test/normal/images",       "output/features_test_normal.csv",    0),
    ]

    for input_rel, output_rel, label in splits:
        print(f"\n--- {input_rel} ---")
        ProcessFeatures(str(BASE / input_rel), str(BASE / output_rel), label)
