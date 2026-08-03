# ⚡ Distributed Training Simulator for Multi-GPU ML Systems

A real-time simulator that models distributed deep learning across multiple GPUs using **Data Parallelism (DP)**, **Pipeline Parallelism (PP)**, and **Tensor Parallelism (TP)**. It visualizes GPU allocation, communication overhead, execution timelines, memory usage, and detects Out-of-Memory (OOM) conditions.

## ✨ Features

- Simulates DP, PP, and TP strategies
- Supports High-Speed Ring and Switch Tree topologies
- Real-time GPU allocation and execution visualization
- 80 GB VRAM memory model with instant OOM detection
- Gantt chart execution timeline
- Compare Mode for side-by-side configuration analysis
- Interactive telemetry dashboard with performance metrics

## 🛠 Tech Stack

**Frontend**
- React
- SVG
- Recharts

**Backend**
- Python
- FastAPI
- Uvicorn

**Communication**
- REST API
- Fetch API

## 📐 Simulation Workflow

Each simulation follows a six-stage training lifecycle:

1. CPU → GPU
2. Distribute
3. Compute
4. Synchronize
5. Gather
6. GPU → CPU

## 📁 Project Structure

```text
Distributed-Training-Simulator/
├── backend/
├── frontend/
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Node.js (v16+)
- Python (3.10+)
- Git

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### Backend Setup

```bash
cd backend

python3 -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
# .\venv\Scripts\activate

pip install -r requirements.txt

uvicorn main:app --reload
```

Backend runs at:

```
http://127.0.0.1:8000
```

Swagger API:

```
http://127.0.0.1:8000/docs
```

### Frontend Setup

Open another terminal:

```bash
cd frontend

npm install

npm run dev
```

Frontend runs at:

```
http://localhost:5173
```

## 📊 Highlights

- GPU utilization analysis
- Memory estimation
- Communication overhead
- Execution timeline
- Compare Mode
- OOM detection

## 📄 License

MIT License
