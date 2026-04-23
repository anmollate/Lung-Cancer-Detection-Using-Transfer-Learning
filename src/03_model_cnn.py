"""
=============================================================
 Lung Cancer Detection — Model 1: Custom CNN
=============================================================
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import keras
from keras import layers, models, callbacks, regularizers
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

print(f"TensorFlow : {tf.__version__}")
print(f"Keras      : {keras.__version__}")
print(f"GPU        : {len(tf.config.list_physical_devices('GPU')) > 0}")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PLOT_DIR  = os.path.join(BASE_DIR, 'notebooks', 'plots')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
NUM_CLASSES = 3
BATCH_SIZE  = 16      # smaller batch = better generalization
EPOCHS      = 50
LR          = 1e-4    # lower LR = more stable training
CLASSES     = ['Normal', 'Benign', 'Malignant']

# ── Load Data ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LOADING DATA")
print("=" * 60)

X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
X_val   = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
X_test  = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
y_val   = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
y_test  = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

print(f"  X_train : {X_train.shape}  y_train : {y_train.shape}")
print(f"  X_val   : {X_val.shape}    y_val   : {y_val.shape}")
print(f"  X_test  : {X_test.shape}   y_test  : {y_test.shape}")

# Verify normalization
print(f"\n  Pixel range train : [{X_train.min():.3f}, {X_train.max():.3f}]")
print(f"  Pixel range val   : [{X_val.min():.3f}, {X_val.max():.3f}]")
print(f"  Pixel range test  : [{X_test.min():.3f}, {X_test.max():.3f}]")

# Class distribution check
print(f"\n  Train class counts : { {i: int(np.sum(y_train==i)) for i in range(3)} }")
print(f"  Val   class counts : { {i: int(np.sum(y_val==i))   for i in range(3)} }")
print(f"  Test  class counts : { {i: int(np.sum(y_test==i))  for i in range(3)} }")

# One-hot encode
y_train_oh = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val_oh   = keras.utils.to_categorical(y_val,   NUM_CLASSES)
y_test_oh  = keras.utils.to_categorical(y_test,  NUM_CLASSES)

# ── Build Simpler CNN ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BUILDING CNN (Simplified for small dataset)")
print("=" * 60)

def build_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Dropout(0.3)(x)

    # Block 2
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Dropout(0.3)(x)

    # Block 3
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Dropout(0.4)(x)

    # Block 4
    x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Dropout(0.4)(x)

    # Classifier head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs, name='LungCancer_CNN_v2')

model = build_cnn()
model.summary()
print(f"\n  Total parameters : {model.count_params():,}")

# ── Compile ────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=Adam(learning_rate=LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ── Callbacks ──────────────────────────────────────────────────────────────────
cb_list = [
    callbacks.EarlyStopping(
        monitor='val_loss', patience=12,
        restore_best_weights=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.3,
        patience=5, min_lr=1e-7, verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'cnn_best.keras'),
        monitor='val_accuracy',
        save_best_only=True, verbose=1
    )
]

# ── Train ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"TRAINING  (Epochs={EPOCHS}, Batch={BATCH_SIZE}, LR={LR})")
print("=" * 60)

history = model.fit(
    X_train, y_train_oh,
    validation_data=(X_val, y_val_oh),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cb_list,
    shuffle=True,
    verbose=1
)

# Save model + history
model.save(os.path.join(MODEL_DIR, 'cnn_model.keras'))
with open(os.path.join(MODEL_DIR, 'cnn_history.pkl'), 'wb') as f:
    pickle.dump(history.history, f)
print("\n  ✅ Saved: models/cnn_model.keras")

# ── Evaluate ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EVALUATION ON TEST SET")
print("=" * 60)

test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
y_pred_prob = model.predict(X_test, verbose=0)
y_pred      = np.argmax(y_pred_prob, axis=1)

print(f"\n  Test Accuracy : {test_acc*100:.2f}%")
print(f"  Test Loss     : {test_loss:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=CLASSES, zero_division=0))

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
auc = roc_auc_score(y_test_bin, y_pred_prob,
                    multi_class='ovr', average='macro')
print(f"  Macro ROC-AUC : {auc:.4f}")

# ── Plot 1: Training Curves ────────────────────────────────────────────────────
h = history.history
epochs_ran = range(1, len(h['accuracy']) + 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Custom CNN — Training Curves', fontsize=13, fontweight='bold')

axes[0].plot(epochs_ran, h['accuracy'],     label='Train', color='#2980b9', linewidth=2)
axes[0].plot(epochs_ran, h['val_accuracy'], label='Val',   color='#e74c3c', linewidth=2, linestyle='--')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(epochs_ran, h['loss'],     label='Train', color='#2980b9', linewidth=2)
axes[1].plot(epochs_ran, h['val_loss'], label='Val',   color='#e74c3c', linewidth=2, linestyle='--')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'cnn_training_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: cnn_training_curves.png")

# ── Plot 2: Confusion Matrix ───────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=0.5, linecolor='white')
ax.set_title('Custom CNN — Confusion Matrix', fontsize=13, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('Actual', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'cnn_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: cnn_confusion_matrix.png")

# ── Final Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CNN COMPLETE — FINAL SUMMARY")
print("=" * 60)
print(f"  Test Accuracy : {test_acc*100:.2f}%")
print(f"  ROC-AUC       : {auc:.4f}")
print(f"  Model saved   : models/cnn_model.keras")
print(f"  Best model    : models/cnn_best.keras")
print("\n✅ Run 04_model_vgg16.py next.")