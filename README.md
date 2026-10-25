# 🧠 Brain CT Hemorrhage & Fracture Classification

A multi-label CNN (TensorFlow/Keras) that reads brain CT slices and flags the type of hemorrhage and the presence of a fracture.

> Research / coursework project. It is **not** a medical device and must not be used for diagnosis.

---

## 🎯 The Problem

Finding intracranial hemorrhage in brain CT scans is time-critical and depends on specialist review of many slices. A single slice can show several findings at once (for example a hemorrhage *and* a fracture), so the task is **multi-label**, not a simple yes/no.

## 💡 The Approach

A compact CNN is trained on labeled CT slices and predicts 7 independent binary labels per image.

| Label | Meaning |
|---|---|
| `Intraventricular` | Bleeding inside the ventricles |
| `Intraparenchymal` | Bleeding inside the brain tissue |
| `Subarachnoid` | Bleeding in the subarachnoid space |
| `Epidural` | Bleeding above the dura |
| `Subdural` | Bleeding below the dura |
| `No_Hemorrhage` | No hemorrhage present |
| `Fracture_Yes_No` | Skull fracture present |

## 🏗️ Model & Training

| Item | Value |
|---|---|
| Input | 224 × 224 RGB slice, pixel values scaled to [0, 1] |
| Architecture | 3 × (Conv2D → BatchNorm → MaxPool) with 16, 32, 64 filters, then Dense(256) |
| Regularization | Dropout 0.6, early stopping (patience 5, restores best weights) |
| Output | 7 sigmoid units |
| Loss / optimizer | Binary cross-entropy / Adam (lr = 0.001) |
| Training | up to 30 epochs, batch size 8 |
| Metrics | Accuracy, plus per-label precision, recall and F1 (classification report) |

## 📊 Results

Running the script prints the test accuracy and a per-label classification report, and plots the training/validation accuracy and loss curves.

<!-- TODO: add the real test accuracy / per-label F1 table here -->
<!-- TODO: add the training curves image, e.g. ![Training curves](docs/curves.png) -->

## 📁 Dataset

The dataset is **not included** (it is private). The script expects this layout:

```text
Patients_CT/
├── train/
│   ├── 1/brain/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   └── 2/brain/
├── validation/
├── test/
└── hemorrhage_diagnosis.csv
```

`hemorrhage_diagnosis.csv` must contain these columns:
`PatientNumber, SliceNumber, Intraventricular, Intraparenchymal, Subarachnoid, Epidural, Subdural, No_Hemorrhage, Fracture_Yes_No`

Unreadable images are skipped automatically.

## 🚀 Getting Started

```bash
git clone https://github.com/mersad31/Hemorrhage-Detection.git
cd Hemorrhage-Detection
pip install numpy pandas opencv-python tensorflow scikit-learn matplotlib
```

**Google Colab (default):** upload `project_in_ml.py`, mount Google Drive and put the dataset in `/content/drive/MyDrive/Patients_CT`.

**Local:** change `data_folder` at the top of `project_in_ml.py` to your dataset path, remove the Drive-mount lines, then run:

```bash
python project_in_ml.py
```

## 📂 Repository Layout

| File | Purpose |
|---|---|
| `project_in_ml.py` | Data loading, preprocessing, model, training, evaluation, plots |
| `LICENSE` | License |

## 🔭 Possible Improvements

- Class-imbalance handling (class weights or focal loss)
- Transfer learning (e.g. a pretrained backbone)
- Per-label thresholds and ROC/AUC reporting

## 📝 License

See [LICENSE](LICENSE).
