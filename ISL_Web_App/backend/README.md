# 🖐️ ISL Recognition Backend

FastAPI backend for Indian Sign Language Recognition system.

## Features

- **REST API** for image-based ISL prediction
- **WebSocket** for real-time video stream processing
- **MediaPipe** integration for hand detection
- **TensorFlow** model inference
- **CORS** enabled for frontend integration

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Update model path** in `main.py`:
   ```python
   MODEL_PATH = r"D:\\WSL\\ISL_2003_final.h5"
   ```

3. **Run server**:
   ```bash
   python main.py
   ```

## API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints

- `GET /` - Health check
- `GET /health` - System status
- `POST /predict` - Image prediction
- `WS /ws` - Real-time prediction stream