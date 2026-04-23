"""
=============================================================
 Lung Cancer Detection — Model 4: Hybrid Model
=============================================================
VGG16 Feature Extractor + SVM + Random Forest
"""

import os
import sys
import traceback
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

print("Step 1: Basic imports OK")

try:
    import tensorflow as tf
    import keras
    from keras import layers, models
    from keras.applications import VGG16
    from keras.applications.vgg16 import preprocess_input
    print(f"Step 2: TF {tf.__version__} | Keras {keras.__version__} OK")
except Exception as e:
    print(f"Step 2 FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, label_binarize
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        roc_auc_score, accuracy_score
    )
    print("Step 3: Sklearn imports OK")
except Exception as e:
    print(f"Step 3 FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PLOT_DIR  = os.path.join(BASE_DIR, 'notebooks', 'plots')
os.makedirs(MODEL_DIR, exist_ok=True)

CLASSES = ['Normal', 'Benign', 'Malignant']

# ── Load Data ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1 — LOADING DATA")
print("=" * 60)

try:
    print("  Loading X_train...")
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    print(f"  X_train: {X_train.shape} — {X_train.nbytes/1024/1024:.1f} MB")

    print("  Loading X_val...")
    X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
    print(f"  X_val: {X_val.shape}")

    print("  Loading X_test...")
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    print(f"  X_test: {X_test.shape}")

    print("  Loading labels...")
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    y_val   = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
    y_test  = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    print("  ✅ All data loaded")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Combine train + val
X_trainval = np.concatenate([X_train, X_val], axis=0)
y_trainval = np.concatenate([y_train, y_val], axis=0)
print(f"\n  Train+Val : {X_trainval.shape}")
print(f"  Test      : {X_test.shape}")

# Free memory
del X_train, X_val
print("  ✅ Freed X_train, X_val from memory")

# VGG16 preprocessing
try:
    print("\n  Applying VGG16 preprocessing...")
    X_trainval_vgg = preprocess_input(X_trainval * 255.0)
    X_test_vgg     = preprocess_input(X_test     * 255.0)
    del X_trainval, X_test
    print("  ✅ VGG16 preprocessing applied")
except Exception as e:
    print(f"  Preprocessing FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── Build Feature Extractor ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — BUILDING VGG16 FEATURE EXTRACTOR")
print("=" * 60)

try:
    print("  Loading VGG16 weights...")
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    inputs  = layers.Input(shape=(224, 224, 3))
    x       = base_model(inputs, training=False)
    outputs = layers.GlobalAveragePooling2D()(x)

    feature_extractor = models.Model(
        inputs, outputs, name='VGG16_FeatureExtractor'
    )
    print(f"  Output feature dim : {feature_extractor.output_shape[-1]}")

    fe_path = os.path.join(MODEL_DIR, 'hybrid_feature_extractor.keras')
    feature_extractor.save(fe_path)
    print(f"  ✅ Saved: hybrid_feature_extractor.keras")
except Exception as e:
    print(f"  Feature extractor FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── Extract Features ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — EXTRACTING FEATURES")
print("=" * 60)

try:
    print("  Extracting train+val features (this takes ~5 mins)...")
    features_trainval = feature_extractor.predict(
        X_trainval_vgg, batch_size=16, verbose=1
    )
    print(f"  Train+Val features: {features_trainval.shape}")

    print("  Extracting test features...")
    features_test = feature_extractor.predict(
        X_test_vgg, batch_size=16, verbose=1
    )
    print(f"  Test features: {features_test.shape}")

    # Free GPU/memory
    del X_trainval_vgg, X_test_vgg
    print("  ✅ Features extracted, image arrays freed")
except Exception as e:
    print(f"  Feature extraction FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Scale features
try:
    scaler = StandardScaler()
    features_trainval_sc = scaler.fit_transform(features_trainval)
    features_test_sc     = scaler.transform(features_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'hybrid_scaler.pkl'))
    print("  ✅ Features scaled | hybrid_scaler.pkl saved")
except Exception as e:
    print(f"  Scaling FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── Train SVM ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — TRAINING SVM CLASSIFIER")
print("=" * 60)

try:
    print("  Training SVM (rbf kernel, C=10)...")
    svm = SVC(
        kernel='rbf', C=10, gamma='scale',
        probability=True, random_state=42
    )
    svm.fit(features_trainval_sc, y_trainval)

    svm_pred = svm.predict(features_test_sc)
    svm_prob = svm.predict_proba(features_test_sc)
    svm_acc  = accuracy_score(y_test, svm_pred)
    svm_auc  = roc_auc_score(
        label_binarize(y_test, classes=[0,1,2]),
        svm_prob, multi_class='ovr', average='macro'
    )
    print(f"\n  SVM Test Accuracy : {svm_acc*100:.2f}%")
    print(f"  SVM ROC-AUC       : {svm_auc:.4f}")
    print(classification_report(y_test, svm_pred,
          target_names=CLASSES, zero_division=0))

    joblib.dump(svm, os.path.join(MODEL_DIR, 'hybrid_svm.pkl'))
    print("  ✅ Saved: hybrid_svm.pkl")
except Exception as e:
    print(f"  SVM FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── Train Random Forest ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — TRAINING RANDOM FOREST")
print("=" * 60)

try:
    print("  Training Random Forest (300 trees)...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None,
        random_state=42, n_jobs=-1
    )
    rf.fit(features_trainval_sc, y_trainval)

    rf_pred = rf.predict(features_test_sc)
    rf_prob = rf.predict_proba(features_test_sc)
    rf_acc  = accuracy_score(y_test, rf_pred)
    rf_auc  = roc_auc_score(
        label_binarize(y_test, classes=[0,1,2]),
        rf_prob, multi_class='ovr', average='macro'
    )
    print(f"\n  RF Test Accuracy  : {rf_acc*100:.2f}%")
    print(f"  RF ROC-AUC        : {rf_auc:.4f}")
    print(classification_report(y_test, rf_pred,
          target_names=CLASSES, zero_division=0))

    joblib.dump(rf, os.path.join(MODEL_DIR, 'hybrid_rf.pkl'))
    print("  ✅ Saved: hybrid_rf.pkl")
except Exception as e:
    print(f"  Random Forest FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── Pick Best ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — SELECTING BEST HYBRID MODEL")
print("=" * 60)

if svm_acc >= rf_acc:
    best_name = 'SVM'
    best_pred = svm_pred
    best_prob = svm_prob
    best_acc  = svm_acc
    best_auc  = svm_auc
    joblib.dump(svm, os.path.join(MODEL_DIR, 'hybrid_best.pkl'))
else:
    best_name = 'Random Forest'
    best_pred = rf_pred
    best_prob = rf_prob
    best_acc  = rf_acc
    best_auc  = rf_auc
    joblib.dump(rf, os.path.join(MODEL_DIR, 'hybrid_best.pkl'))

meta = {
    'best_classifier': best_name,
    'svm_accuracy':    round(svm_acc * 100, 2),
    'rf_accuracy':     round(rf_acc  * 100, 2),
    'best_accuracy':   round(best_acc * 100, 2),
    'best_auc':        round(best_auc, 4),
    'feature_dim':     features_trainval.shape[1]
}
with open(os.path.join(MODEL_DIR, 'hybrid_meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"  SVM Accuracy : {svm_acc*100:.2f}%  AUC: {svm_auc:.4f}")
print(f"  RF  Accuracy : {rf_acc*100:.2f}%  AUC: {rf_auc:.4f}")
print(f"  🏆 Best      : {best_name} ({best_acc*100:.2f}%)")

# ── Plot Confusion Matrices ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — PLOTTING CONFUSION MATRICES")
print("=" * 60)

try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Hybrid Model — Confusion Matrices',
                 fontsize=13, fontweight='bold')

    for ax, pred, title, cmap in zip(
        axes,
        [svm_pred, rf_pred],
        [f'Hybrid SVM ({svm_acc*100:.1f}%)',
         f'Hybrid RF  ({rf_acc*100:.1f}%)'],
        ['Oranges', 'YlOrBr']
    ):
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap=cmap, ax=ax,
            xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=1, linecolor='white',
            annot_kws={'size': 13, 'weight': 'bold'}
        )
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'hybrid_confusion_matrix.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: hybrid_confusion_matrix.png")
except Exception as e:
    print(f"  Plot FAILED: {e}")
    traceback.print_exc()

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("HYBRID MODEL COMPLETE — FINAL SUMMARY")
print("=" * 60)
print(f"  SVM  : {svm_acc*100:.2f}%  AUC: {svm_auc:.4f}")
print(f"  RF   : {rf_acc*100:.2f}%  AUC: {rf_auc:.4f}")
print(f"  Best : {best_name} ({best_acc*100:.2f}%)")
print(f"\n  Files saved:")
print(f"    ✅ hybrid_feature_extractor.keras")
print(f"    ✅ hybrid_svm.pkl")
print(f"    ✅ hybrid_rf.pkl")
print(f"    ✅ hybrid_best.pkl")
print(f"    ✅ hybrid_scaler.pkl")
print(f"    ✅ hybrid_meta.pkl")
print(f"    ✅ hybrid_confusion_matrix.png")
print("\n✅ All 4 models done! Ready for Streamlit App.")