"""
=============================================================
 Lung Cancer Detection — Model 3: ResNet50 Transfer Learning
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
from keras import layers, models, callbacks
from keras.optimizers import Adam
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
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
BATCH_SIZE  = 16
EPOCHS_HEAD = 20
EPOCHS_FINE = 20
LR_HEAD     = 1e-3
LR_FINE     = 1e-5
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

print(f"  X_train : {X_train.shape}")
print(f"  X_val   : {X_val.shape}")
print(f"  X_test  : {X_test.shape}")
print(f"  Train class counts : { {i: int(np.sum(y_train==i)) for i in range(3)} }")
print(f"  Val   class counts : { {i: int(np.sum(y_val==i))   for i in range(3)} }")
print(f"  Test  class counts : { {i: int(np.sum(y_test==i))  for i in range(3)} }")

# ResNet50 preprocessing (scales to [-1, 1])
X_train_res = preprocess_input(X_train * 255.0)
X_val_res   = preprocess_input(X_val   * 255.0)
X_test_res  = preprocess_input(X_test  * 255.0)
print("  ✅ ResNet50 preprocessing applied")

# One-hot encode
y_train_oh = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val_oh   = keras.utils.to_categorical(y_val,   NUM_CLASSES)
y_test_oh  = keras.utils.to_categorical(y_test,  NUM_CLASSES)

# ── Build ResNet50 Model ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BUILDING RESNET50 TRANSFER LEARNING MODEL")
print("=" * 60)

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False
print(f"  Base model layers : {len(base_model.layers)}")

# Custom head
inputs  = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(512, activation='relu')(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dropout(0.5)(x)
x       = layers.Dense(256, activation='relu')(x)
x       = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs, outputs, name='LungCancer_ResNet50')
model.summary()
print(f"\n  Total params     : {model.count_params():,}")

# ── Phase 1: Train Head Only ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"PHASE 1 — TRAINING HEAD ONLY (Epochs={EPOCHS_HEAD}, LR={LR_HEAD})")
print("=" * 60)

model.compile(
    optimizer=Adam(learning_rate=LR_HEAD),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cb_phase1 = [
    callbacks.EarlyStopping(
        monitor='val_loss', patience=8,
        restore_best_weights=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.3,
        patience=4, min_lr=1e-7, verbose=1
    )
]

history1 = model.fit(
    X_train_res, y_train_oh,
    validation_data=(X_val_res, y_val_oh),
    epochs=EPOCHS_HEAD,
    batch_size=BATCH_SIZE,
    callbacks=cb_phase1,
    shuffle=True,
    verbose=1
)

print(f"\n  Phase 1 best val_accuracy : "
      f"{max(history1.history['val_accuracy'])*100:.2f}%")

# ── Phase 2: Fine-tune Last 10 Layers ─────────────────────────────────────────
print("\n" + "=" * 60)
print(f"PHASE 2 — FINE-TUNING LAST 10 LAYERS (LR={LR_FINE})")
print("=" * 60)

base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False

trainable = sum([np.prod(v.shape) for v in model.trainable_weights])
print(f"  Trainable params after unfreeze : {trainable:,}")

model.compile(
    optimizer=Adam(learning_rate=LR_FINE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cb_phase2 = [
    callbacks.EarlyStopping(
        monitor='val_loss', patience=8,
        restore_best_weights=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.3,
        patience=4, min_lr=1e-8, verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'resnet50_best.keras'),
        monitor='val_accuracy',
        save_best_only=True, verbose=1
    )
]

history2 = model.fit(
    X_train_res, y_train_oh,
    validation_data=(X_val_res, y_val_oh),
    epochs=EPOCHS_FINE,
    batch_size=BATCH_SIZE,
    callbacks=cb_phase2,
    shuffle=True,
    verbose=1
)

# Save model
model.save(os.path.join(MODEL_DIR, 'resnet50_model.keras'))

# Combine histories
combined = {
    'accuracy':     history1.history['accuracy']     + history2.history['accuracy'],
    'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
    'loss':         history1.history['loss']         + history2.history['loss'],
    'val_loss':     history1.history['val_loss']     + history2.history['val_loss'],
}
with open(os.path.join(MODEL_DIR, 'resnet50_history.pkl'), 'wb') as f:
    pickle.dump(combined, f)
print("\n  ✅ Saved: models/resnet50_model.keras")

# ── Evaluate ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EVALUATION ON TEST SET")
print("=" * 60)

test_loss, test_acc = model.evaluate(X_test_res, y_test_oh, verbose=0)
y_pred_prob = model.predict(X_test_res, verbose=0)
y_pred      = np.argmax(y_pred_prob, axis=1)

print(f"\n  Test Accuracy : {test_acc*100:.2f}%")
print(f"  Test Loss     : {test_loss:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=CLASSES, zero_division=0))

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
auc_score  = roc_auc_score(y_test_bin, y_pred_prob,
                            multi_class='ovr', average='macro')
print(f"  Macro ROC-AUC : {auc_score:.4f}")

# ── Plot 1: Training Curves ────────────────────────────────────────────────────
h          = combined
phase1_end = len(history1.history['accuracy'])
epochs_ran = range(1, len(h['accuracy']) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('ResNet50 — Training Curves', fontsize=13, fontweight='bold')

axes[0].plot(epochs_ran, h['accuracy'],     label='Train',
             color='#9b59b6', linewidth=2)
axes[0].plot(epochs_ran, h['val_accuracy'], label='Val',
             color='#e74c3c', linewidth=2, linestyle='--')
axes[0].axvline(x=phase1_end, color='gray', linestyle=':',
                linewidth=1.5, label='Fine-tune starts')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(epochs_ran, h['loss'],     label='Train',
             color='#9b59b6', linewidth=2)
axes[1].plot(epochs_ran, h['val_loss'], label='Val',
             color='#e74c3c', linewidth=2, linestyle='--')
axes[1].axvline(x=phase1_end, color='gray', linestyle=':',
                linewidth=1.5, label='Fine-tune starts')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'resnet50_training_curves.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: resnet50_training_curves.png")

# ── Plot 2: Confusion Matrix ───────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax,
            xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 13, 'weight': 'bold'})
ax.set_title('ResNet50 — Confusion Matrix', fontsize=13, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('Actual', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'resnet50_confusion_matrix.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: resnet50_confusion_matrix.png")

# ── Final Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESNET50 COMPLETE — FINAL SUMMARY")
print("=" * 60)
print(f"  Test Accuracy : {test_acc*100:.2f}%")
print(f"  ROC-AUC       : {auc_score:.4f}")
print(f"  Model saved   : models/resnet50_model.keras")
print(f"  Best model    : models/resnet50_best.keras")
print("\n✅ Run 07_train_hybrid.py next.")