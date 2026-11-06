# RealReveal: Deepfake Video Detection System

**RealReveal** is a web-based deepfake video detection platform and hybrid deep learning framework developed by the **Department of Information Technology** at **JIS College of Engineering, Kalyani** under the guidance of **Assistant Professor Mrs. Tanusree Saha**.

The system integrates spatial feature extraction via a pre-trained **InceptionV3** Convolutional Neural Network (CNN) with temporal sequence modeling using a 3-layer stacked **Gated Recurrent Unit (GRU)** Recurrent Neural Network (RNN). It is deployed through an interactive **Flask** web application featuring **Firebase Authentication** and **MongoDB** persistence for real-time video verification and forensic history tracking.

## Research Paper

* **Title:** *RealReveal: Bringing Deepfake Detection to the Web with Advanced Machine Learning*
* **Journal Submission:** *Baghdad Science Journal* (Manuscript ID: `EMID:188806401e1896d0`)
* **Manuscript File:** [My\_Research\_Paper.03.2026.docx](https://www.google.com/search?q=./My_Research_Paper.03.2026.docx)
* **Abstract Summary:** This study presents a systematic evaluation of contemporary deepfake detection paradigms—spanning deep learning architectures, conventional machine learning algorithms, and statistical inference—and introduces **RealReveal**, an end-to-end web platform that bridges academic digital forensics and real-world multimedia verification. By combining InceptionV3 frame-level embeddings (`2048-D`) with a masked multi-layer GRU sequence classifier across 20 frames per video, the framework captures both subtle pixel-level spatial artifacts and inter-frame temporal inconsistencies.

## Project Team & Academic Guidance

### Project Guide
* **Mrs. Tanusree Saha** — Assistant Professor, Department of Information Technology, **JIS College of Engineering** (Autonomous Institution, NAAC 'A' Accredited & NIRF Ranked), Kalyani, Nadia, West Bengal (`tanusree.saha@jiscollege.ac.in`)

### Team Members (B.Tech in Information Technology, Batch 2021–2025)

| Sl. No. | Name | University Roll No. | College ID | Role & Contribution |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Md Dilshad Alam** | `123211004039` | `JIS/2021/0726` | Deep Learning Pipeline, Flask Backend & Research Manuscript |
| 2 | **Arani Mishra** | `123211004012` | `JIS/2021/0502` | Model Training, Preprocessing Pipeline & Dataset Curation |
| 3 | **Soumyadip Saha** | `123211004078` | `JIS/2021/0991` | GRU Temporal Architecture, Ablation Experiments & Evaluation |
| 4 | **Suraj Sarkar** | `123211004081` | `JIS/2021/0551` | Web Application Frontend, UI/UX Templates & Visualization |
| 5 | **Surajit Matia** | `123211004082` | `JIS/2021/1259` | Firebase Authentication, MongoDB Integration & System Testing |

## Key Features

* **Hybrid Spatial-Temporal Architecture:** Extracts `2048`-dimensional spatial feature vectors per frame using ImageNet pre-trained **InceptionV3** and models temporal frame transitions across `20` frames using a 3-layer **GRU** (`64 -> 32 -> 16` units) with binary sequence masking.
* **Automated Video Preprocessing:** Performs OpenCV frame extraction, center-square cropping (`crop_center_square`), `224x224` bilinear resizing, and RGB channel normalization (`preprocess_input`).
* **Real-Time Web Verification:** Allows users to upload video files (`.mp4`, `.avi`, `.mov`, `.wmv`) up to `100 MB` via a drag-and-drop interface and receive immediate `FAKE` or `REAL` classifications (`>= 0.51` threshold) with confidence scores.
* **Dual Authentication & Cloud Storage:** Supports **Firebase Authentication** (Email/Password and Google Sign-In) with automatic local session fallback and stores user profiles and timestamped analysis records in **MongoDB** (`deepfake_detection` database).
* **Interactive Analytics History:** Provides a filterable dashboard (`All`, `Real`, `Fake`), sorting controls (`Newest`, `Oldest`, `Confidence`), and detailed forensic reports for every analyzed video.

## System Architecture & Methodology

### 1. Video Preprocessing Pipeline
1. **Frame Sampling:** Reads input video streams via OpenCV (`cv2.VideoCapture`) and extracts up to `MAX_SEQ_LENGTH = 20` sequential frames.
2. **Spatial Normalization:** Applies center-square cropping around the shortest frame dimension, resizes each frame to `IMG_SIZE = 224x224`, converts `BGR` to `RGB`, and applies Keras `inception_v3.preprocess_input` scaling (`[-1, 1]`).
3. **Padding & Masking:** Pads videos shorter than 20 frames with zero vectors and generates a boolean mask tensor (`shape: [1, 20]`) so padded frames do not distort temporal hidden states.

### 2. Hybrid InceptionV3 + GRU Network

| Layer | Type / Configuration | Output Dimension | Purpose |
| :--- | :--- | :--- | :--- |
| **Spatial Backbone** | Pre-trained `InceptionV3` (`include_top=False`, `pooling="avg"`, frozen weights) | `2048` per frame | Extracts high-level spatial and facial artifact representations |
| **Sequence Input** | `frame_features_input` + `mask_input` | `(20, 2048)` + `(20,)` | Feeds 20-frame feature sequences with temporal validity mask |
| **Recurrent Layer 1** | `GRU(64, return_sequences=True)` + `BatchNormalization()` | `(20, 64)` | Captures fine-grained short-range inter-frame transitions |
| **Recurrent Layer 2** | `GRU(32, return_sequences=True)` + `BatchNormalization()` | `(20, 32)` | Models mid-range temporal consistency across facial movements |
| **Recurrent Layer 3** | `GRU(16)` + `Dropout(0.5)` | `(16,)` | Condenses sequence into a video-level temporal embedding |
| **Dense Classifier** | `Dense(32, activation="relu")` -> `Dense(1, activation="sigmoid")` | `(1,)` | Outputs probability $P(\text{FAKE}) \in [0, 1]$ |

## Experimental Results & Benchmarks

### Overall Performance on DFDC Dataset
Trained on `401` videos (`80%` train / `20%` validation split) and evaluated on `468` unseen test videos from the **Deepfake Detection Challenge (DFDC)** dataset using `Adam` (`lr = 1e-4`), `BinaryCrossentropy`, and `EarlyStopping` (`patience = 5`):

| Metric | Value |
| :--- | :--- |
| **Test Accuracy** | **81.25%** |
| **Test Loss** | **0.6758** |
| **Precision (FAKE)** | **0.80** |
| **Recall (FAKE)** | **0.82** |
| **F1-Score (FAKE)** | **0.81** |
| **Feature Extraction Speed** | **0.12 sec / frame** |

### Ablation Study: Sequence Length (`MAX_SEQ_LENGTH`)

| Sequence Length | Validation Accuracy | Observation |
| :--- | :--- | :--- |
| `10` Frames | 76.50% | Insufficient temporal context for subtle lip-sync and blink anomalies |
| **`20` Frames** | **81.25%** | **Optimal balance between accuracy and real-time web inference latency** |
| `30` Frames | 82.10% | Marginal accuracy gain (`+0.85%`) at `1.5x` higher computational cost |

### Backbone Comparison: InceptionV3 vs. ResNet50

| Feature Extractor | Feature Vector Size | Test Accuracy | Extraction Speed (sec/frame) |
| :--- | :--- | :--- | :--- |
| **InceptionV3 (Ours)** | `2048` | **81.25%** | **0.12 s** |
| ResNet50 | `2048` | 79.00% | 0.15 s |

### Cross-Dataset Generalization

| Dataset | Test Accuracy | F1-Score (FAKE) |
| :--- | :--- | :--- |
| **DFDC (In-Domain)** | **81.25%** | **0.81** |
| **FaceForensics++ (Cross-Domain)** | **76.50%** | **0.77** |

## Repository Structure

```text
deepfake-detection/
├── app.py                                      # Flask web application & InceptionV3-GRU inference server
├── My_Research_Paper.03.2026.docx              # Full research paper manuscript (Baghdad Science Journal)
├── requirements.txt                            # Python dependencies
├── Procfile                                    # Gunicorn deployment configuration
├── .gitignore                                  # Excludes credentials, virtual environments, and local uploads
├── model/
│   └── deepfake_video_model.h5                 # Trained GRU sequence classifier weights (5.1 MB)
├── notebook/
│   └── deep-fake-detection-on-videos.ipynb     # End-to-end training & evaluation Jupyter Notebook
├── dataset/
│   ├── sample_submission.csv                   # Sample prediction format
│   ├── train_sample_videos/                    # DFDC training sample videos & metadata.json
│   └── test_videos/                            # DFDC test evaluation videos
├── static/
│   ├── logo.svg                                # RealReveal brand logo
│   ├── hero-illustration.svg                   # Landing page vector illustration
│   ├── spinner.gif                             # Processing indicator animation
│   └── spinner1.gif                            # Secondary loading animation
└── templates/
    ├── base.html                               # Master layout with responsive navigation & dark/light styling
    ├── index.html                              # Landing page showcasing platform workflow & capabilities
    ├── login.html                              # Firebase & session login interface
    ├── register.html                           # User registration interface
    ├── upload.html                             # Drag-and-drop video upload & live analysis results view
    ├── history.html                            # Filterable & sortable past analysis dashboard
    ├── analysis_details.html                   # Detailed per-video forensic report view
    └── profile.html                            # User profile & account statistics page
```

## Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/dilshad790/deepfake-detection.git
cd deepfake-detection
```

### 2. Create and Activate a Virtual Environment
```bash
# Using Conda (Python 3.9 recommended for TensorFlow 2.13 compatibility)
conda create -n deepfake python=3.9 -y
conda activate deepfake

# Or using standard venv
python -m venv venv
source venv/bin/activate   # On Windows: venv\\Scripts\\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install pymongo python-dotenv gunicorn
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory (never commit `.env` or `firebase-credentials.json` to version control):

```env
SECRET_KEY=your_flask_secret_key
DEBUG=True
PORT=5000

# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017/deepfake_detection

# Firebase Web SDK Configuration
FIREBASE_API_KEY=your_firebase_api_key
FIREBASE_AUTH_DOMAIN=your-project-id.firebaseapp.com
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_STORAGE_BUCKET=your-project-id.appspot.com
FIREBASE_MESSAGING_SENDER_ID=your_sender_id
FIREBASE_APP_ID=your_app_id
FIREBASE_MEASUREMENT_ID=your_measurement_id
FIREBASE_CREDENTIALS_PATH=firebase-credentials.json
```

*(Note: If Firebase credentials or MongoDB are not configured locally, `app.py` automatically falls back to local session authentication and local JSON storage inside `data/analyses/`.)*

### 5. Run the Application
```bash
python app.py
```

Open `http://localhost:5000` in your browser to register, upload a video, and run deepfake detection.

## Citation

If you reference this project or manuscript in your academic work, please cite:

```bibtex
@article{saha2026realreveal,
  title   = {RealReveal: Bringing Deepfake Detection to the Web with Advanced Machine Learning},
  author  = {Saha, Tanusree and Alam, Md Dilshad and Mishra, Arani and Saha, Soumyadip and Sarkar, Suraj and Matia, Surajit},
  journal = {Baghdad Science Journal},
  year    = {2026},
  note    = {Department of Information Technology, JIS College of Engineering, Kalyani, West Bengal, India}
}
```\n