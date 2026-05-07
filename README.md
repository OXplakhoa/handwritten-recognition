# MNIST Handwritten Digit Recognition

A complete CNN project: two models (baseline and tuned), live training curves, Gradio demo, and auto-generated report outline.

## Project Structure

```
.
|- notebooks/
|   |- mnist_cnn.ipynb          # Full experiment — run cell by cell
|- models/
|   |- best_baseline.keras       # Saved baseline weights (after Section 6)
|   |- best_tuned.keras          # Saved tuned weights (after Section 6)
|- report/
|   |- outline.md                # Auto-generated report (after Section 11)
|   |- *.png                     # Embedded figures
|- scripts/
|   |- build_notebook.py         # Regenerates the notebook from source
|   |- sanity_check.py           # Quick model-shape sanity test
|- requirements.txt
|- README.md
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Notebook

```powershell
jupyter notebook notebooks\mnist_cnn.ipynb
```

Walk through all 11 sections in order. Sections 6 (training) and 10 (Gradio) take the most time.

## Quick Sanity Check

```powershell
.venv\Scripts\python scripts\sanity_check.py
```

## Notebooks on GitHub / Google Colab

Open directly in the browser with no setup:

- **Google Colab**: https://colab.research.google.com/github/YOUR_USERNAME/handwritten_recognition/blob/main/notebooks/mnist_cnn.ipynb
- **GitHub Viewer**: https://github.com/YOUR_USERNAME/handwritten_recognition/blob/main/notebooks/mnist_cnn.ipynb

> Replace `YOUR_USERNAME` with your GitHub username after pushing.

## Notebook Sections

| # | Section | Key Output |
|---|---|---|
| 1 | Problem Statement | Intro text |
| 2 | Dataset Overview | Class distribution chart, sample images |
| 3 | Preprocessing | Normalised arrays, one-hot labels |
| 4 | Baseline CNN | Model summary, architecture diagram |
| 5 | Tuned CNN + Augmentation | Deeper model with augmentation layers |
| 6 | Training Pipeline | Live TensorBoard-style curves, saved weights |
| 7 | Validation Comparison | Polished 2x2 comparison plot, metrics table |
| 8 | Test Evaluation | Row-normalised confusion matrix, confused-pair labels |
| 9 | Sample Predictions | 20-image grid (green=correct, red=wrong) |
| 10 | Gradio Demo | Interactive canvas + top-3 predictions |
| 11 | Report Outline | `report/outline.md` with embedded PNG plots |

## Notes

- The Gradio demo runs inline with `share=True` (public link) inside the notebook.
- Train the tuned model (Section 6) before running the demo cell so the weights are available.
- Drawing is not real-time/continuous -- click **Submit** after each stroke. This is a Gradio behaviour.
