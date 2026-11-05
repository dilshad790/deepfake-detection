# deepfake-detection

RealReveal: Deepfake Video Detection SystemRealReveal is a web-based deepfake video detection platform and hybrid deep learning framework developed by the Department of Information Technology at JIS College of Engineering, Kalyani under the guidance of Assistant Professor Mrs. Tanusree Saha.The system integrates spatial feature extraction via a pre-trained InceptionV3 Convolutional Neural Network (CNN) with temporal sequence modeling using a 3-layer stacked Gated Recurrent Unit (GRU) Recurrent Neural Network (RNN). It is deployed through an interactive Flask web application featuring Firebase Authentication and MongoDB persistence for real-time video verification and forensic history tracking.Research PaperTitle: RealReveal: Bringing Deepfake Detection to the Web with Advanced Machine LearningJournal Submission: Baghdad Science Journal (Manuscript ID: EMID:188806401e1896d0)Manuscript File: My_Research_Paper.03.2026.docxAbstract Summary: This study presents a systematic evaluation of contemporary deepfake detection paradigms—spanning deep learning architectures, conventional machine learning algorithms, and statistical inference—and introduces RealReveal, an end-to-end web platform that bridges academic digital forensics and real-world multimedia verification. By combining InceptionV3 frame-level embeddings (2048-D) with a masked multi-layer GRU sequence classifier across 20 frames per video, the framework captures both subtle pixel-level spatial artifacts and inter-frame temporal inconsistencies.Project Team & Academic GuidanceProject GuideMrs. Tanusree Saha — Assistant Professor, Department of Information Technology, JIS College of Engineering (Autonomous Institution, NAAC 'A' Accredited & NIRF Ranked), Kalyani, Nadia, West Bengal (tanusree.saha@jiscollege.ac.in)Team Members (B.Tech in Information Technology, Batch 2021–2025)Sl. No.NameUniversity Roll No.College IDRole & Contribution1Md Dilshad Alam123211004039JIS/2021/0726Deep Learning Pipeline, Flask Backend & Research Manuscript2Arani Mishra123211004012JIS/2021/0502Model Training, Preprocessing Pipeline & Dataset Curation3Soumyadip Saha123211004078JIS/2021/0991GRU Temporal Architecture, Ablation Experiments & Evaluation4Suraj Sarkar123211004081JIS/2021/0551Web Application Frontend, UI/UX Templates & Visualization5Surajit Matia123211004082JIS/2021/1259Firebase Authentication, MongoDB Integration & System TestingKey FeaturesHybrid Spatial-Temporal Architecture: Extracts 2048-dimensional spatial feature vectors per frame using ImageNet pre-trained InceptionV3 and models temporal frame transitions across 20 frames using a 3-layer GRU (64 -> 32 -> 16 units) with binary sequence masking.Automated Video Preprocessing: Performs OpenCV frame extraction, center-square cropping (crop_center_square), 224x224 bilinear resizing, and RGB channel normalization (preprocess_input).Real-Time Web Verification: Allows users to upload video files (.mp4, .avi, .mov, .wmv) up to 100 MB via a drag-and-drop interface and receive immediate FAKE or REAL classifications (>= 0.51 threshold) with confidence scores.Dual Authentication & Cloud Storage: Supports Firebase Authentication (Email/Password and Google Sign-In) with automatic local session fallback and stores user profiles and timestamped analysis records in MongoDB (deepfake_detection database).Interactive Analytics History: Provides a filterable dashboard (All, Real, Fake), sorting controls (Newest, Oldest, Confidence), and detailed forensic reports for every analyzed video.System Architecture & Methodology1. Video Preprocessing PipelineFrame Sampling: Reads input video streams via OpenCV (cv2.VideoCapture) and extracts up to MAX_SEQ_LENGTH = 20 sequential frames.Spatial Normalization: Applies center-square cropping around the shortest frame dimension, resizes each frame to IMG_SIZE = 224x224, converts BGR to RGB, and applies Keras inception_v3.preprocess_input scaling ([-1, 1]).Padding & Masking: Pads videos shorter than 20 frames with zero vectors and generates a boolean mask tensor (shape: [1, 20]) so padded frames do not distort temporal hidden states.2. Hybrid InceptionV3 + GRU NetworkLayerType / ConfigurationOutput DimensionPurposeSpatial BackbonePre-trained InceptionV3 (include_top=False, pooling="avg", frozen weights)2048 per frameExtracts high-level spatial and facial artifact representationsSequence Inputframe_features_input + mask_input(20, 2048) + (20,)Feeds 20-frame feature sequences with temporal validity maskRecurrent Layer 1GRU(64, return_sequences=True) + BatchNormalization()(20, 64)Captures fine-grained short-range inter-frame transitionsRecurrent Layer 2GRU(32, return_sequences=True) + BatchNormalization()(20, 32)Models mid-range temporal consistency across facial movementsRecurrent Layer 3GRU(16) + Dropout(0.5)(16,)Condenses sequence into a video-level temporal embeddingDense ClassifierDense(32, activation="relu") -> Dense(1, activation="sigmoid")(1,)Outputs probability $P(\text{FAKE}) \in [0, 1]$Experimental Results & BenchmarksOverall Performance on DFDC DatasetTrained on 401 videos (80% train / 20% validation split) and evaluated on 468 unseen test videos from the Deepfake Detection Challenge (DFDC) dataset using Adam (lr = 1e-4), BinaryCrossentropy, and EarlyStopping (patience = 5):MetricValueTest Accuracy81.25%Test Loss0.6758Precision (FAKE)0.80Recall (FAKE)0.82F1-Score (FAKE)0.81Feature Extraction Speed0.12 sec / frameAblation Study: Sequence Length (MAX_SEQ_LENGTH)Sequence LengthValidation AccuracyObservation10 Frames76.50%Insufficient temporal context for subtle lip-sync and blink anomalies20 Frames81.25%Optimal balance between accuracy and real-time web inference latency30 Frames82.10%Marginal accuracy gain (+0.85%) at 1.5x higher computational costBackbone Comparison: InceptionV3 vs. ResNet50Feature ExtractorFeature Vector SizeTest AccuracyExtraction Speed (sec/frame)InceptionV3 (Ours)204881.25%0.12 sResNet50204879.00%0.15 sCross-Dataset GeneralizationDatasetTest AccuracyF1-Score (FAKE)DFDC (In-Domain)81.25%0.81FaceForensics++ (Cross-Domain)76.50%0.77Repository StructureDeepFakeVideoDetection/
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

Installation & Local Setup1. Clone the Repositorygit clone https://github.com/dilshad790/DeepFakeVideoDetection.git
cd DeepFakeVideoDetection

2. Create and Activate a Virtual Environment# Using Conda (Python 3.9 recommended for TensorFlow 2.13 compatibility)
conda create -n deepfake python=3.9 -y
conda activate deepfake

# Or using standard venv
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

3. Install Dependenciespip install -r requirements.txt
pip install pymongo python-dotenv gunicorn

4. Configure Environment VariablesCreate a .env file in the root directory (never commit .env or firebase-credentials.json to version control):SECRET_KEY=your_flask_secret_key
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

(Note: If Firebase credentials or MongoDB are not configured locally, app.py automatically falls back to local session authentication and local JSON storage inside data/analyses/.)5. Run the Applicationpython app.py

Open http://localhost:5000 in your browser to register, upload a video, and run deepfake detection.CitationIf you reference this project or manuscript in your academic work, please cite:@article{saha2026realreveal,
  title   = {RealReveal: Bringing Deepfake Detection to the Web with Advanced Machine Learning},
  author  = {Saha, Tanusree and Alam, Md Dilshad and Mishra, Arani and Saha, Soumyadip and Sarkar, Suraj and Matia, Surajit},
  journal = {Baghdad Science Journal},
  year    = {2026},
  note    = {Department of Information Technology, JIS College of Engineering, Kalyani, West Bengal, India}
}

