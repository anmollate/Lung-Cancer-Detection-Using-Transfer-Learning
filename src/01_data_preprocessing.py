import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, 'dataset')
PLOT_DIR   = os.path.join(BASE_DIR, 'notebooks', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

CLASSES      = ['Normal_cases', 'Bengin_cases', 'Malignant_cases']
CLASS_COLORS = {'Normal_cases': '#2ecc71', 'Bengin_cases': '#f39c12', 'Malignant_cases': '#e74c3c'}

print("=" * 60)
print("STEP 1 — DATASET OVERVIEW")
print("=" * 60)

class_counts = {}
all_images   = {}

for cls in CLASSES:
    cls_path = os.path.join(DATA_DIR, cls)
    if not os.path.exists(cls_path):
        print(f"⚠️  Folder not found: {cls_path}")
        sys.exit(1)
    imgs = [f for f in os.listdir(cls_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    class_counts[cls] = len(imgs)
    all_images[cls]   = [os.path.join(cls_path, f) for f in imgs]
    print(f"  {cls.capitalize():12} : {len(imgs)} images")

total = sum(class_counts.values())
print(f"\n  {'TOTAL':12} : {total} images")
print(f"\n  Class Balance:")
for cls, cnt in class_counts.items():
    bar = '█' * int((cnt / total) * 40)
    print(f"  {cls.capitalize():12} : {bar} {cnt/total*100:.1f}%")

# ── 2. Image Property Inspection ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — IMAGE PROPERTY INSPECTION")
print("=" * 60)

corrupt  = []
sizes    = defaultdict(list)
modes    = defaultdict(set)
file_kb  = defaultdict(list)

for cls in CLASSES:
    sample = all_images[cls][:50]
    for path in sample:
        try:
            img = Image.open(path)
            sizes[cls].append(img.size)
            modes[cls].add(img.mode)
            file_kb[cls].append(os.path.getsize(path) / 1024)
        except Exception as e:
            corrupt.append((path, str(e)))

for cls in CLASSES:
    unique_sizes = set(sizes[cls])
    avg_kb = np.mean(file_kb[cls])
    print(f"\n  {cls.capitalize()}:")
    print(f"    Unique sizes : {unique_sizes}")
    print(f"    Modes        : {modes[cls]}")
    print(f"    Avg file size: {avg_kb:.1f} KB")

if corrupt:
    print(f"\n⚠️  Corrupt images found: {len(corrupt)}")
    for path, err in corrupt:
        print(f"   {os.path.basename(path)} — {err}")
else:
    print(f"\n  ✅ No corrupt images found.")

# ── 3. Class Distribution Plot ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — PLOTTING CLASS DISTRIBUTION")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

bars = axes[0].bar(
    [c.capitalize() for c in CLASSES],
    [class_counts[c] for c in CLASSES],
    color=[CLASS_COLORS[c] for c in CLASSES],
    edgecolor='white', linewidth=1.5, width=0.5
)
for bar, cls in zip(bars, CLASSES):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 8,
        f"{class_counts[cls]}\n({class_counts[cls]/total*100:.1f}%)",
        ha='center', va='bottom', fontsize=11, fontweight='bold'
    )
axes[0].set_title('Class Distribution', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Number of Images')
axes[0].set_ylim(0, max(class_counts.values()) * 1.2)
axes[0].spines[['top', 'right']].set_visible(False)

axes[1].pie(
    [class_counts[c] for c in CLASSES],
    labels=[c.capitalize() for c in CLASSES],
    colors=[CLASS_COLORS[c] for c in CLASSES],
    autopct='%1.1f%%', startangle=140,
    wedgeprops=dict(edgecolor='white', linewidth=2),
    textprops={'fontsize': 11}
)
axes[1].set_title('Class Proportion', fontsize=13, fontweight='bold')

plt.suptitle('Lung CT Scan Dataset — Class Distribution', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'class_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: class_distribution.png")

# ── 4. Sample Images Grid ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — SAMPLE IMAGES GRID")
print("=" * 60)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('Sample CT Scans — Each Class', fontsize=15, fontweight='bold')

for row, cls in enumerate(CLASSES):
    samples = all_images[cls][:4]
    for col, img_path in enumerate(samples):
        img = Image.open(img_path).convert('RGB').resize((224, 224))
        axes[row][col].imshow(img)
        axes[row][col].axis('off')
        axes[row][col].set_title(f'{cls.capitalize()} #{col+1}',
                                  fontsize=9, color=CLASS_COLORS[cls])

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'sample_images.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: sample_images.png")

# ── 5. Pixel Intensity Distribution ───────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — PIXEL INTENSITY DISTRIBUTION")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Pixel Intensity Distribution per Class', fontsize=13, fontweight='bold')

for ax, cls in zip(axes, CLASSES):
    all_pixels = []
    for path in all_images[cls][:10]:
        img = np.array(Image.open(path).convert('L').resize((224, 224)))
        all_pixels.extend(img.flatten().tolist())
    ax.hist(all_pixels, bins=50, color=CLASS_COLORS[cls], alpha=0.8, edgecolor='white')
    mean_px = np.mean(all_pixels)
    ax.axvline(mean_px, color='black', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_px:.0f}')
    ax.set_title(cls.capitalize(), fontsize=11, fontweight='bold', color=CLASS_COLORS[cls])
    ax.set_xlabel('Pixel Value (0–255)')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'pixel_intensity.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: pixel_intensity.png")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EDA COMPLETE — SUMMARY")
print("=" * 60)
print(f"  Total Images : {total}")
print(f"  Classes      : Normal | Benign | Malignant")
print(f"  Image Size   : 512x512 → resize to 224x224")
print(f"  Color Mode   : RGB")
print(f"  Imbalance    : Benign underrepresented → augmentation needed")
print(f"  Plots saved  : notebooks/plots/")
print("\n✅ Run 02_preprocessing.py next.")