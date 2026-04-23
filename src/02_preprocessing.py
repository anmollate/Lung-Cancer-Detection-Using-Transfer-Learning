import os
import json
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
OUT_DIR  = os.path.join(BASE_DIR, 'data')
PLOT_DIR = os.path.join(BASE_DIR, 'notebooks', 'plots')
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
IMG_SIZE  = 224
CLASSES   = ['Normal_cases', 'Bengin_cases', 'Malignant_cases']
LABEL_MAP = {'Normal_cases': 0, 'Bengin_cases': 1, 'Malignant_cases': 2}
CLASS_COLORS = {
    'Normal_cases':    '#2ecc71',
    'Bengin_cases':    '#f39c12',
    'Malignant_cases': '#e74c3c'
}

# ── Helper: Load Single Image ──────────────────────────────────────────────────
def load_image(path, size=IMG_SIZE):
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0

# ── Helper: Augment Single Image ──────────────────────────────────────────────
def augment_image(img_array):
    img = Image.fromarray((img_array * 255).astype(np.uint8))

    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    angle = random.uniform(-20, 20)
    img = img.rotate(angle, fillcolor=(0, 0, 0))

    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))

    if random.random() > 0.8:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    return np.array(img, dtype=np.float32) / 255.0

# ── Step 1: Load All Images ────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — LOADING & RESIZING IMAGES (512x512 → 224x224)")
print("=" * 60)

images = []
labels = []
loaded_counts = {}

for cls in CLASSES:
    cls_path = os.path.join(DATA_DIR, cls)
    files    = [f for f in os.listdir(cls_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    count = 0
    for fname in files:
        try:
            arr = load_image(os.path.join(cls_path, fname))
            images.append(arr)
            labels.append(LABEL_MAP[cls])
            count += 1
        except Exception as e:
            print(f"  ⚠️  Skipped {fname}: {e}")

    loaded_counts[cls] = count
    print(f"  {cls:20} : {count} images loaded ✅")

images = np.array(images, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

print(f"\n  Dataset shape  : {images.shape}")
print(f"  Labels shape   : {labels.shape}")
print(f"  Memory usage   : {images.nbytes / 1024 / 1024:.1f} MB")
print(f"  Pixel range    : [{images.min():.2f}, {images.max():.2f}] ✅ Normalized")

# ── Step 2: Augment Benign (Minority Class) ───────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — AUGMENTING BENIGN CLASS (Minority Oversampling)")
print("=" * 60)

TARGET   = loaded_counts['Malignant_cases']
benign_idx  = np.where(labels == LABEL_MAP['Bengin_cases'])[0]
current     = len(benign_idx)
needed      = TARGET - current

print(f"  Benign current  : {current}")
print(f"  Target count    : {TARGET}  (match Malignant)")
print(f"  Generating      : {needed} augmented images...")

aug_images = []
aug_labels = []

for i in range(needed):
    src = benign_idx[i % current]
    aug_images.append(augment_image(images[src]))
    aug_labels.append(LABEL_MAP['Bengin_cases'])

aug_images = np.array(aug_images, dtype=np.float32)
aug_labels = np.array(aug_labels, dtype=np.int32)

X = np.concatenate([images, aug_images], axis=0)
y = np.concatenate([labels, aug_labels], axis=0)

# Shuffle
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

print(f"\n  After augmentation:")
for cls, lbl in LABEL_MAP.items():
    print(f"    {cls:20} : {np.sum(y == lbl)} images")
print(f"  Total : {len(X)} images")

# ── Step 3: Augmentation Examples Plot ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — SAVING AUGMENTATION EXAMPLES PLOT")
print("=" * 60)

original = images[benign_idx[0]]
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
fig.suptitle('Augmentation Examples — Benign Class', fontsize=13, fontweight='bold')

axes[0][0].imshow(original)
axes[0][0].set_title('Original', fontweight='bold', color='red')
axes[0][0].axis('off')

for i in range(1, 10):
    r, c = divmod(i, 5)
    aug = augment_image(original)
    axes[r][c].imshow(aug)
    axes[r][c].set_title(f'Aug #{i}')
    axes[r][c].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'augmentation_examples.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: augmentation_examples.png")

# ── Step 4: Train / Val / Test Split (70 / 15 / 15) ──────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — TRAIN / VAL / TEST SPLIT  (70% / 15% / 15%)")
print("=" * 60)

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.176, random_state=42, stratify=y_trainval
)

print(f"  Train : {len(X_train)} images  ({len(X_train)/len(X)*100:.0f}%)")
print(f"  Val   : {len(X_val)}  images  ({len(X_val)/len(X)*100:.0f}%)")
print(f"  Test  : {len(X_test)}  images  ({len(X_test)/len(X)*100:.0f}%)")

print(f"\n  Train class breakdown:")
for cls, lbl in LABEL_MAP.items():
    print(f"    {cls:20} : {np.sum(y_train == lbl)}")

print(f"\n  Test class breakdown:")
for cls, lbl in LABEL_MAP.items():
    print(f"    {cls:20} : {np.sum(y_test == lbl)}")

# ── Step 5: Save .npy Files ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — SAVING DATA ARRAYS TO data/ FOLDER")
print("=" * 60)

np.save(os.path.join(OUT_DIR, 'X_train.npy'), X_train)
np.save(os.path.join(OUT_DIR, 'X_val.npy'),   X_val)
np.save(os.path.join(OUT_DIR, 'X_test.npy'),  X_test)
np.save(os.path.join(OUT_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(OUT_DIR, 'y_val.npy'),   y_val)
np.save(os.path.join(OUT_DIR, 'y_test.npy'),  y_test)

with open(os.path.join(OUT_DIR, 'label_map.json'), 'w') as f:
    json.dump({
        'label_map': LABEL_MAP,
        'classes':   CLASSES,
        'img_size':  IMG_SIZE
    }, f, indent=2)

print(f"  ✅ X_train.npy  : {X_train.shape}  — {X_train.nbytes/1024/1024:.1f} MB")
print(f"  ✅ X_val.npy    : {X_val.shape}")
print(f"  ✅ X_test.npy   : {X_test.shape}")
print(f"  ✅ y_train.npy  : {y_train.shape}")
print(f"  ✅ y_val.npy    : {y_val.shape}")
print(f"  ✅ y_test.npy   : {y_test.shape}")
print(f"  ✅ label_map.json")

# ── Step 6: Split Distribution Plot ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Class Distribution Across Splits', fontsize=13, fontweight='bold')

for ax, (name, y_split) in zip(axes, [
    ('Train', y_train), ('Validation', y_val), ('Test', y_test)
]):
    counts = [np.sum(y_split == lbl) for lbl in LABEL_MAP.values()]
    bars   = ax.bar(
        ['Normal', 'Benign', 'Malignant'], counts,
        color=[CLASS_COLORS[c] for c in CLASSES],
        edgecolor='white', linewidth=1.5
    )
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1, str(cnt),
                ha='center', fontsize=10, fontweight='bold')
    ax.set_title(f'{name} Set', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count')
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'split_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved: split_distribution.png")

print("\n✅ Preprocessing Complete! Run 03_model_cnn.py next.")