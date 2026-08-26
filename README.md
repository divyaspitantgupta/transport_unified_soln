# UrbanEdge AI: Mobile Urban Intelligence Platform 🚌🏙️

![SIH](https://img.shields.io/badge/SIH-2026-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![EdgeAI](https://img.shields.io/badge/Edge-AI-orange)

An AI-powered onboard and centralized software platform that transforms public transport buses into mobile urban sensing units. Designed to detect road defects, analyze traffic density, and monitor infrastructure in real-time while maintaining strict bandwidth and computational efficiency.

## 📖 Table of Contents
1. [Overview & Problem Statement](#overview)
2. [End-to-End Solution Architecture](#architecture)
3. [Technology Stack](#tech-stack)
4. [Machine Learning Pipeline & Models](#ml-pipeline)
5. [Model Training & Testing Strategy](#training-testing)
6. [Project Structure](#structure)
7. [Installation & Setup](#setup)

---

## 🌍 <a name="overview"></a> Overview & Problem Statement
City authorities typically rely on fixed CCTV cameras and reactive citizen complaints to monitor road health and traffic. Fixed cameras have blind spots and static views, while manual inspections are slow. **UrbanEdge AI** leverages the extensive spatial-temporal coverage of city bus fleets. By outfitting buses with edge-compute nodes, the system analyzes multiple camera feeds in real-time, extracts structured metadata, and streams low-bandwidth telemetry to a centralized GIS dashboard for proactive city management.

---

## 📐 <a name="architecture"></a> End-to-End Solution Architecture

The architecture follows a strict **Edge-to-Cloud** paradigm to minimize cellular bandwidth usage and cloud compute costs.

### Flow Diagram (Mermaid)

```mermaid
graph TD
    subgraph Edge System [On-Bus Edge Node (Jetson/Pi)]
        A[Cameras & GPS] -->|Frames + NMEA| B(Frame Ingest & Sampling - 5 FPS)
        B --> C{Unified YOLOv8n Pass}
        C -->|Vehicle/Pedestrian Box| D[ByteTrack - Object Tracking]
        C -->|Defect Box| E[Spatial Deduplication Buffer]
        D --> F{Anomaly/Hit-and-Run?}
        F -->|Yes| G[Conditional ANPR OCR]
        F -->|No| H[Event Payload Builder]
        E --> H
        G --> H
    end
    
    subgraph Cloud Platform [Central Command Cloud]
        H -->|MQTT / HTTPS JSON| I[API Gateway & Message Broker]
        I --> J[PostGIS / TimescaleDB]
        J --> K[Spatial Clustering DBSCAN]
        K --> L[Next.js GIS Dashboard]
    end
```

### Pipeline Steps:
1. **Data Ingestion:** Video frames are sampled at 5-10 FPS and synchronized with GPS coordinates.
2. **Edge Inference:** A unified neural network detects vehicles, pedestrians, and road defects in a single pass.
3. **Tracking & Logic:** ByteTrack assigns IDs to calculate traffic density. 
4. **Conditional OCR:** ANPR is triggered *only* upon anomaly detection (e.g., erratic driving) to save compute.
5. **Deduplication:** Spatial algorithms drop redundant defect detections (e.g., same pothole within 5m).
6. **Transmission:** Structured JSON metadata (+ optional WebP snapshot) is sent via MQTT.
7. **Cloud Analytics:** Backend aggregates multi-bus data, clusters defects using PostGIS, and visualizes on a Mapbox/Leaflet dashboard.

---

## 🛠️ <a name="tech-stack"></a> Technology Stack

### Edge (On-Bus)
*   **Inference Engine:** TensorRT, ONNX Runtime
*   **Computer Vision:** OpenCV, YOLOv8 (Ultralytics), ByteTrack, PaddleOCR
*   **Hardware Target:** NVIDIA Jetson Orin Nano / Raspberry Pi 5 + Hailo-8
*   **Transport Layer:** MQTT (Paho), Protocol Buffers

### Central Cloud
*   **Backend:** FastAPI (Python) or Node.js (Express), Celery (Background tasks)
*   **Database:** PostgreSQL (with PostGIS extension), Redis (Message Broker)
*   **Storage:** AWS S3 / MinIO (for compressed snapshots)

### Frontend (GIS Dashboard)
*   **Framework:** Next.js (React), TypeScript
*   **Mapping:** Leaflet / Mapbox GL JS (React-Map-GL)
*   **Styling:** Tailwind CSS, Shadcn UI

---

## 🧠 <a name="ml-pipeline"></a> Machine Learning Pipeline & Models

### 1. Unified Object & Defect Detector
*   **Model:** YOLOv8 Nano (YOLOv8n)
*   **Reasoning:** Merging tasks into one model reduces memory bandwidth and forward-pass overhead.
*   **Classes:** `Pothole`, `Crack`, `Missing_Marking`, `Car`, `Bus`, `Truck`, `Two-Wheeler`, `Pedestrian`, `License_Plate`.

### 2. Multi-Object Tracking (MOT)
*   **Model:** ByteTrack
*   **Reasoning:** Purely spatial tracking (Kalman Filter + IoU). Zero deep learning ReID cost. Runs on CPU in milliseconds.

### 3. Automatic Number Plate Recognition (ANPR)
*   **Model:** PaddleOCR / Tesseract (Lightweight Mobile-v2 backbone)
*   **Reasoning:** Kept cold until triggered by the tracking logic to conserve GPU/NPU cycles.

---

## 📊 <a name="training-testing"></a> Model Training & Testing Strategy

### Datasets
*   **Vehicles & Pedestrians:** [IDD (Indian Driving Dataset)](https://idd.insaan.iiit.ac.in/) - Handles unstructured Indian traffic.
*   **Road Defects:** [RoadDamageDataset (RDD2022)](https://github.com/sekilab/RoadDamageDetector) / Kaggle Pothole datasets.
*   **Preparation:** Datasets are merged, standardized to YOLO format, and normalized.

### Training Workflow
1.  **Transfer Learning:** Initialize YOLOv8n with COCO weights.
2.  **Hyperparameters:** `imgsz=640`, `epochs=100`, `batch=32` (scaled based on GPU), Optimizer: AdamW.
3.  **Augmentation:** Heavy mosaic, random perspective, and motion blur to simulate bus camera vibration.
4.  **Export:** Convert PyTorch `.pt` to `.onnx` and finally to `.engine` (TensorRT) with FP16 or INT8 precision for edge deployment.

### Testing & Evaluation Metrics
*   **Accuracy (mAP@50-95):** Target > 0.75 for vehicles, > 0.65 for road defects.
*   **Throughput (FPS):** Must sustain > 20 FPS on target edge hardware (Jetson Nano) at 640x640 resolution to allow overhead for tracking and I/O.
*   **Latency:** Edge pipeline end-to-latency < 50ms per frame.

---

## 📂 <a name="structure"></a> Project Structure

```text
UrbanEdge-AI/
│
├── edge_node/                     # On-bus processing software
│   ├── models/                    # Exported .onnx / .engine weights
│   ├── pipeline/
│   │   ├── detector.py            # YOLO inference class
│   │   ├── tracker.py             # ByteTrack integration
│   │   └── ocr_engine.py          # Conditional ANPR logic
│   ├── utils/
│   │   └── gps_sync.py            # NMEA parsing and frame synchronization
│   ├── config.yaml                # Edge settings (FPS, thresholds)
│   └── main_edge.py               # Main edge execution loop
│
├── cloud_backend/                 # Centralized aggregation API
│   ├── api/
│   │   └── routes.py              # FastAPI endpoints (ingest, fetch)
│   ├── services/
│   │   └── spatial_cluster.py     # PostGIS DBSCAN deduplication
│   ├── models/                    # SQLAlchemy / ORM schemas
│   └── main_api.py                # Server entry point
│
├── gis_dashboard/                 # Next.js web application
│   ├── src/
│   │   ├── components/            # UI, Map, and Chart components
│   │   ├── pages/                 # Next.js routes
│   │   └── styles/                # Tailwind configs
│   └── package.json
│
├── docker-compose.yml             # Local dev deployment for backend/db
└── README.md
```

---

## 🚀 <a name="setup"></a> Installation & Setup

### 1. Edge Node Setup
```bash
cd edge_node
python -m venv venv
source venv/bin/activate
pip install -r requirements_edge.txt
# Run the simulated edge pipeline (uses local video if RTSP not available)
python main_edge.py --source data/test_video.mp4
```

### 2. Cloud Backend Setup
```bash
cd cloud_backend
docker-compose up -d db redis  # Starts PostgreSQL/PostGIS and Redis
pip install -r requirements_backend.txt
uvicorn main_api:app --reload --port 8000
```

### 3. GIS Dashboard Setup
```bash
cd gis_dashboard
npm install
npm run dev
# Access the dashboard at http://localhost:3000
```
