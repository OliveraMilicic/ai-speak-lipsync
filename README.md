# AI Speak — audio-driven facial animation

Speech audio in, facial animation out. The model predicts the **52 ARKit blendshape coefficients** at **60 FPS** directly from a voice recording.

## How it works

```
audio (16 kHz)
   │
   ▼
HuBERT (facebook/hubert-base-ls960) ── 768-dim features, ~50 FPS
   │
   ▼
linear interpolation to 60 FPS
   │
   ▼
11-frame window (5 before + current + 5 after)
   │
   ▼
BiLSTM model ── 52 blendshapes for the centre frame
   │
   ▼
post-processing ── smoothing, silence handling, blinks
   │
   ▼
CSV (frames × 52) + meta.json
```

**Model.** `Linear 768→256` → 2-layer bidirectional LSTM (hidden 256) → centre frame → `Linear 512→256 → ReLU → Dropout(0.3) → Linear 256→52 → Sigmoid`. About 3M parameters.

**Training.** About 30 minutes of speech from two speakers (358 recordings, ~108k frames), split 90/10 by file so the same recording never appears in both train and validation. MSE loss, Adam (lr 1e-3) with ReduceLROnPlateau, 30 epochs. Best validation MSE was 0.0016.

**Post-processing.** The raw predictions are cleaned up before export.
- The inhale at the start of a clip is suppressed
- Mouth shapes are amplified, and the mouth closes when the audio goes silent
- Eyes and brows are smoothed so they don't jitter
- Natural blinks are added every 3–5 seconds
- The lower face gets a light Gaussian smoothing

**Latency.** The model looks 5 frames ahead, which is **83.3 ms of lookahead** at 60 FPS.

## Repository

| File | Description |
|---|---|
| `train.ipynb` | Feature extraction, dataset, model and training. Produces `best_model_v1.pth` |
| `inference.ipynb` | Runs the model on a folder of audio files and writes CSVs + `meta.json` |
| `best_model_v1.pth` | Trained model weights |
| `examples/` | Example output |
| `requirements.txt` | Python dependencies |

## Running it

Both notebooks are made for **Google Colab** with a GPU runtime (Runtime → Change runtime type → GPU).

### Inference on your own audio

1. Open `inference.ipynb` in Colab.
2. Put `best_model_v1.pth` and a folder with your `.wav` / `.mp3` files in your Google Drive.
3. Set `BASE`, `MODEL_PATH`, `INPUT_DIR` and `OUTPUT_DIR` in the *Paths* cell.
4. Run all cells.

Each audio file gives one CSV with one row per frame and 52 columns in the standard ARKit order (`browInnerUp`, `browDownLeft`, … `jawOpen` is column 24 … `tongueOut`). Values are between 0 and 1. `meta.json` records the inference time, real-time factor (RTF), lookahead and output FPS for every file.

### Training

`train.ipynb` expects one folder per speaker, with matching `.wav` and `.csv` files.

```
BASE/
├── spk08_blendshapes/renamed_spk08/   spk08_001.wav, spk08_001.csv, ...
└── spk14_blendshapes/renamed_spk14/   spk14_001.wav, spk14_001.csv, ...
```

The training data is not included in this repository.

### Running locally

```bash
pip install -r requirements.txt
```

Remove the `google.colab` / `drive.mount` lines and point the paths to local folders. A CUDA GPU is recommended, but the code falls back to CPU.

## Authors

- Olivera Miličić
- Iva Bojanić

Faculty of Technical Sciences, University of Novi Sad
