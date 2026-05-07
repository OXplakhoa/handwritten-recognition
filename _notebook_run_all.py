import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib; matplotlib.use("Agg")

# === CELL 3 ===
# Create output directories first (needed by later cells)
import os
os.makedirs("models", exist_ok=True)
os.makedirs("report",  exist_ok=True)

# Imports
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Load MNIST
from tensorflow.keras.datasets import mnist

(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

print(f"Training set : {X_train_full.shape[0]:,} samples, shape {X_train_full.shape[1:]}px")
print(f"Test set     : {X_test.shape[0]:,} samples, shape {X_test.shape[1:]}px")
print(f"Pixel range  : [{X_train_full.min()}, {X_train_full.max()}]")

# === CELL 4 ===
# Class Distribution
fig, ax = plt.subplots(figsize=(7, 3))
counts = Counter(y_train_full)
classes = list(range(10))
values  = [counts[c] for c in classes]
bars = ax.bar(classes, values, color="#4C6EF5", edgecolor="white", linewidth=0.5)
ax.set_xlabel("Digit")
ax.set_ylabel("Samples")
ax.set_title("Training Set -- Class Distribution")
ax.set_xticks(classes)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            f"{v:,}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("report/class_distribution.png", dpi=150)
plt.show()

# === CELL 5 ===
# Sample Images
np.random.seed(42)
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for digit, ax in zip(range(10), axes.flat):
    idx = np.where(y_train_full == digit)[0][0]
    ax.imshow(X_train_full[idx], cmap="gray_r")
    ax.set_title(f"Label: {digit}")
    ax.axis("off")
plt.suptitle("Sample Images -- One per Digit Class")
plt.tight_layout()
plt.savefig("report/sample_images.png", dpi=150)
plt.show()

# === CELL 7 ===
# Split: 50k train / 10k validation
from tensorflow.keras.utils import to_categorical

X_train, X_val = X_train_full[:50_000], X_train_full[50_000:]
y_train, y_val = y_train_full[:50_000], y_train_full[50_000:]

# Reshape to (samples, 28, 28, 1) and normalise to [0, 1]
X_train = X_train.astype("float32").reshape(-1, 28, 28, 1) / 255.0
X_val   = X_val.astype("float32").reshape(-1, 28, 28, 1) / 255.0
X_test  = X_test.astype("float32").reshape(-1, 28, 28, 1) / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_val_cat   = to_categorical(y_val, 10)
y_test_cat  = to_categorical(y_test, 10)

print(f"Train : {X_train.shape}  |  {y_train_cat.shape}")
print(f"Val   : {X_val.shape}    |  {y_val_cat.shape}")
print(f"Test  : {X_test.shape}   |  {y_test_cat.shape}")

# === CELL 10 ===
from tensorflow.keras import Sequential
from tensorflow.keras import layers as L

def make_baseline(input_shape=(28, 28, 1), num_classes=10):
    model = Sequential([
        L.Input(shape=input_shape),
        L.Conv2D(32, (3, 3), activation="relu"),
        L.MaxPooling2D((2, 2)),
        L.Conv2D(64, (3, 3), activation="relu"),
        L.MaxPooling2D((2, 2)),
        L.Flatten(),
        L.Dense(128, activation="relu"),
        L.Dropout(0.3),
        L.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

baseline_model = make_baseline()
baseline_model.summary()

# === CELL 14 ===
def make_tuned(input_shape=(28, 28, 1), num_classes=10):
    model = Sequential([
        L.Input(shape=input_shape),
        # Augmentation
        L.RandomRotation(0.1),
        L.RandomZoom(0.1),
        L.RandomTranslation(0.1, 0.1),
        # Block 1
        L.Conv2D(32, (3, 3), activation="relu", padding="same"),
        L.BatchNormalization(),
        L.Conv2D(32, (3, 3), activation="relu", padding="same"),
        L.BatchNormalization(),
        L.MaxPooling2D((2, 2)),
        L.Dropout(0.25),
        # Block 2
        L.Conv2D(64, (3, 3), activation="relu", padding="same"),
        L.BatchNormalization(),
        L.Conv2D(64, (3, 3), activation="relu", padding="same"),
        L.BatchNormalization(),
        L.MaxPooling2D((2, 2)),
        L.Dropout(0.25),
        # Classifier
        L.Flatten(),
        L.Dense(256, activation="relu"),
        L.Dropout(0.4),
        L.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

tuned_model = make_tuned()
tuned_model.summary()

# === CELL 18 ===
import IPython.display as display
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback


class LivePlotCallback(Callback):
    """
    Keras callback that updates an inline matplotlib figure after each epoch.

    Each subplot shows two metrics (train + val):
      - Raw  -> solid line, full opacity
      - Smoothed EMA -> translucent shaded area (alpha=0.3) under the line
    """

    def __init__(self, title,
                 train_acc_key="accuracy", val_acc_key="val_accuracy",
                 train_loss_key="loss",     val_loss_key="val_loss"):
        super().__init__()
        self.title         = title
        self.train_acc_key = train_acc_key
        self.val_acc_key   = val_acc_key
        self.train_loss_key = train_loss_key
        self.val_loss_key   = val_loss_key
        self.history = {k: [] for k in
                       [train_acc_key, val_acc_key, train_loss_key, val_loss_key]}
        self._ema    = {k: None for k in self.history}
        self._ema_smoothing = 0.9
        self._fig   = None
        self._axes  = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key in self.history:
            raw = logs.get(key, 0.0)
            self.history[key].append(raw)
            if self._ema[key] is None:
                self._ema[key] = raw
            else:
                self._ema[key] = (self._ema_smoothing * self._ema[key]
                                 + (1 - self._ema_smoothing) * raw)

        if self._fig is None:
            self._build_fig()

        for ax, t_key, v_key, metric_name in [
            (self._axes[0], self.train_acc_key,  self.val_acc_key,  "Accuracy"),
            (self._axes[1], self.train_loss_key, self.val_loss_key, "Loss"),
        ]:
            ax.clear()
            for key, color, label in [
                (t_key, "#4C6EF5", "Train"),
                (v_key, "#FD7E14", "Val"),
            ]:
                epochs   = range(1, len(self.history[key]) + 1)
                raw_vals = self.history[key]
                ema_val  = self._ema[key]
                ema_vals = np.full(len(raw_vals), ema_val)
                ax.plot(epochs, raw_vals, color=color, linewidth=1.5, alpha=0.9)
                ax.fill_between(epochs, raw_vals, ema_vals,
                               color=color, alpha=0.25)
            ax.set_title(f"{metric_name} -- {self.title}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        display.clear_output(wait=True)
        display.display(self._fig)

    def _build_fig(self):
        self._fig, self._axes = plt.subplots(1, 2, figsize=(12, 4))
        self._fig.suptitle(self.title, fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._fig.patch.set_facecolor("#111827")

    def save_fig(self, path):
        if self._fig:
            self._fig.savefig(path, dpi=150, bbox_inches="tight",
                              facecolor=self._fig.get_facecolor())

# === CELL 19 ===
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
)
from tensorflow.keras.backend import clear_session

# Shared training config
EPOCHS   = 20
BATCH    = 64
PATIENCE = 5

callbacks_cfg = dict(
    reduce_lr   = ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                  patience=3, min_lr=1e-6, verbose=1),
    early_stop  = EarlyStopping(monitor="val_loss", patience=PATIENCE,
                               restore_best_weights=True, verbose=1),
    checkpoint  = lambda name: ModelCheckpoint(
        f"models/best_{name}.keras", monitor="val_loss",
        save_best_only=True, verbose=1),
)

# === CELL 21 ===
clear_session()
baseline_model = make_baseline()

cb_baseline = [
    callbacks_cfg["reduce_lr"],
    callbacks_cfg["early_stop"],
    callbacks_cfg["checkpoint"]("baseline"),
    LivePlotCallback("Baseline Training"),
]

hist_baseline = baseline_model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=EPOCHS, batch_size=BATCH,
    callbacks=cb_baseline,
    verbose=0,
)

print("\nBaseline training complete.")
cb_baseline[-1].save_fig("report/baseline_training_curves.png")

# === CELL 23 ===
clear_session()
tuned_model = make_tuned()

cb_tuned = [
    callbacks_cfg["reduce_lr"],
    callbacks_cfg["early_stop"],
    callbacks_cfg["checkpoint"]("tuned"),
    LivePlotCallback("Tuned Training"),
]

hist_tuned = tuned_model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=EPOCHS, batch_size=BATCH,
    callbacks=cb_tuned,
    verbose=0,
)

print("\nTuned model training complete.")
cb_tuned[-1].save_fig("report/tuned_training_curves.png")

# === CELL 25 ===
# Polished static comparison plot -- 2x2 grid, Baseline vs Tuned overlaid
def ema_series(values, smoothing=0.9):
    result = []
    for v in values:
        if not result:
            result.append(v)
        else:
            result.append(smoothing * result[-1] + (1 - smoothing) * v)
    return result

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

histories = {"Baseline": hist_baseline, "Tuned": hist_tuned}
colors    = {"Baseline": "#4C6EF5", "Tuned": "#FD7E14"}

for col, (metric, ylabel) in enumerate([("accuracy", "Accuracy"), ("loss", "Loss")]):
    for row, (split, key) in enumerate([("Train", metric), ("Val", f"val_{metric}")]):
        ax = axes[row, col]
        for name, hist in histories.items():
            raw   = hist.history[key]
            ema   = ema_series(raw)
            epochs = range(1, len(raw) + 1)
            color = colors[name]
            ax.plot(epochs, raw, color=color, linewidth=1.5, alpha=0.85,
                    label=f"{name} (raw)")
            ax.fill_between(epochs, raw, ema, color=color, alpha=0.22,
                            label=f"{name} (smooth)")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Epoch")
        ax.set_title(f"{split} {ylabel}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

plt.suptitle("Training Comparison: Baseline vs Tuned", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("report/model_comparison.png", dpi=150)
plt.show()

# === CELL 26 ===
# Best-epoch summary table
import pandas as pd

best_baseline_epoch = int(np.argmax(hist_baseline.history["val_accuracy"])) + 1
best_tuned_epoch    = int(np.argmax(hist_tuned.history["val_accuracy"]))    + 1
best_baseline_acc   = float(np.max(hist_baseline.history["val_accuracy"]))
best_tuned_acc      = float(np.max(hist_tuned.history["val_accuracy"]))

summary = pd.DataFrame({
    "Model":          ["Baseline CNN", "Tuned CNN"],
    "Best Val Acc":   [f"{best_baseline_acc:.4f}", f"{best_tuned_acc:.4f}"],
    "Final Val Acc":  [f"{hist_baseline.history['val_accuracy'][-1]:.4f}",
                       f"{hist_tuned.history['val_accuracy'][-1]:.4f}"],
    "Epochs Trained": [len(hist_baseline.history["val_accuracy"]),
                       len(hist_tuned.history["val_accuracy"])],
    "Best Epoch":     [best_baseline_epoch, best_tuned_epoch],
})

print(summary.to_string(index=False))
summary.to_csv("report/model_summary.csv", index=False)

# === CELL 28 ===
from sklearn.metrics import confusion_matrix, classification_report

# Load best weights
baseline_model.load_weights("models/best_baseline.keras")
tuned_model.load_weights("models/best_tuned.keras")

# Test predictions
baseline_pred = np.argmax(baseline_model.predict(X_test, verbose=0), axis=1)
tuned_pred    = np.argmax(tuned_model.predict(X_test, verbose=0), axis=1)

baseline_test_acc = float(np.mean(baseline_pred == y_test))
tuned_test_acc    = float(np.mean(tuned_pred == y_test))

print(f"Baseline -- Test Accuracy: {baseline_test_acc:.4f}")
print(f"Tuned    -- Test Accuracy: {tuned_test_acc:.4f}")
print()
print("Classification Report (Tuned):")
print(classification_report(y_test, tuned_pred, digits=4))

# === CELL 29 ===
# Confusion Matrix -- row-normalised heatmap with confused-pair annotations
import seaborn as sns

cm     = confusion_matrix(y_test, tuned_pred)
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_norm, annot=False, fmt=".0%", cmap="Blues",
            xticklabels=range(10), yticklabels=range(10), ax=ax,
            cbar_kws={"label": "Recall (%)"})

# Top 5 confused pairs (highest off-diagonal values)
off_diagonal = [(cm_norm[i, j], i, j)
                 for i in range(10) for j in range(10) if i != j]
top_confused = sorted(off_diagonal, reverse=True)[:5]
confused_label = ", ".join(f"{i}<->{j}" for _, i, j in top_confused)

# Red boxes around confused cells
for _, i, j in top_confused:
    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                               edgecolor="red", linewidth=2.5))

ax.annotate(f"Most confused: {confused_label}",
            xy=(0.5, -0.12), xycoords="axes fraction",
            ha="center", fontsize=9, color="#555", style="italic")

ax.set_xlabel("Predicted Digit")
ax.set_ylabel("True Digit")
ax.set_title("Confusion Matrix (row-normalised -- recall per true class)")
plt.tight_layout()
plt.savefig("report/confusion_matrix.png", dpi=150)
plt.show()
print("Top confused digit pairs:", confused_label)

# === CELL 31 ===
# 20 test images with predicted / true labels
np.random.seed(7)
sample_idx = np.random.choice(len(X_test), size=20, replace=False)

fig, axes = plt.subplots(4, 5, figsize=(12, 10))
for idx, ax in zip(sample_idx, axes.flat):
    img   = X_test[idx].squeeze()
    pred  = int(tuned_pred[idx])
    true  = int(y_test[idx])
    color = "#2e9a2e" if pred == true else "#e03030"
    ax.imshow(img, cmap="gray_r")
    ax.set_title(f"True: {true}  Pred: {pred}", color=color, fontsize=11)
    ax.axis("off")

plt.suptitle("Sample Predictions -- Green=Correct, Red=Incorrect", fontsize=13)
plt.tight_layout()
plt.savefig("report/sample_predictions.png", dpi=150)
plt.show()

# === CELL 34 ===
import gradio as gr
import numpy as np


def predict_digit(img):
    """
    Receives a (H, W) uint8 numpy array from Gradio Sketchpad.
    Converts to (1, 28, 28, 1) float32 in [0, 1] range, runs inference.
    Returns top-3 predictions as a dict {digit: probability}.
    """
    # Sketchpad gives (H, W) grayscale; invert (white stroke -> black digit)
    gray = np.array(img).astype("float32") / 255.0
    gray = 1.0 - gray
    batch = gray.reshape(1, 28, 28, 1)
    probs = tuned_model.predict(batch, verbose=0)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    return {str(d): float(p) for d, p in zip(top3_idx, probs[top3_idx])}


demo = gr.Blocks(title="MNIST Digit Recognition -- Tuned CNN")

with demo:
    gr.Markdown("## Draw a digit (0-9) and submit")
    with gr.Row():
        canvas = gr.Sketchpad(
            label="Draw here",
            height=300, width=300,
        )
        with gr.Column():
            submit_btn = gr.Button("Submit", variant="primary")
            clear_btn  = gr.Button("Clear")
            output     = gr.Label(label="Top-3 Predictions",
                                  num_top_classes=3)

    gr.Examples(
        [[X_test[i].squeeze()] for i in range(10)],
        inputs=canvas,
    )

    submit_btn.click(fn=predict_digit, inputs=canvas, outputs=output)
    clear_btn.click(fn=lambda: None, inputs=None, outputs=canvas)

demo.launch(inline=True, share=True, server_port=7860)
print("Demo launched.")

# === CELL 36 ===
# Auto-generate report/outline.md with embedded plot images
# Variables baseline_test_acc and tuned_test_acc are from Section 8.

report_md = f"""
# MNIST Handwritten Digit Recognition -- Report Outline

> Auto-generated from notebook. Replace TODO markers with your team's text.

---

## 1. Introduction
- TODO: Context about MNIST and why digit recognition matters
- TODO: Project goal and scope

## 2. Dataset
- MNIST: 60,000 training / 10,000 test images, 28x28 grayscale
- Class distribution is roughly uniform (~6,000 samples per digit class)
- Preprocessing: pixel normalisation to [0,1], one-hot labels

![Class distribution](report/class_distribution.png)
![Sample images](report/sample_images.png)

## 3. Model Architectures

### Baseline CNN
```
Input(28,28,1)
Conv2D(32, 3x3, ReLU) -> MaxPooling2D(2x2)
Conv2D(64, 3x3, ReLU) -> MaxPooling2D(2x2)
Flatten -> Dense(128, ReLU) -> Dropout(0.3)
Dense(10, softmax)
```

### Tuned CNN
```
Input(28,28,1)
RandomRotation / RandomZoom / RandomTranslation (augmentation)
Conv2D(32, 3x3, ReLU) -> BatchNorm -> Conv2D(32, 3x3, ReLU) -> BatchNorm -> MaxPooling2D -> Dropout(0.25)
Conv2D(64, 3x3, ReLU) -> BatchNorm -> Conv2D(64, 3x3, ReLU) -> BatchNorm -> MaxPooling2D -> Dropout(0.25)
Flatten -> Dense(256, ReLU) -> Dropout(0.4)
Dense(10, softmax)
```

## 4. Training
- Optimiser: Adam with ReduceLROnPlateau (factor=0.5, patience=3)
- Early stopping: patience=5, restore_best_weights=True
- Batch size: 64, max epochs: 20

## 5. Results

### Training Curves
![Baseline training](report/baseline_training_curves.png)
![Tuned training](report/tuned_training_curves.png)

### Model Comparison
![Comparison](report/model_comparison.png)

### Test Evaluation
- Baseline Test Accuracy: {baseline_test_acc:.4f}
- Tuned Test Accuracy:    {tuned_test_acc:.4f}

![Confusion Matrix](report/confusion_matrix.png)

### Sample Predictions
![Predictions](report/sample_predictions.png)

## 6. Demo
The Gradio demo (Section 10) lets users draw a digit in the browser and receive
top-3 predictions from the tuned model in real-time.

---

*Report generated automatically. Review all TODO markers before submission.*
"""

with open("report/outline.md", "w", encoding="utf-8") as f:
    f.write(report_md)

print("report/outline.md written.")