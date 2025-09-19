# 🖐️ ISL Recognition Web Application

A complete web application for **Indian Sign Language (ISL) Recognition** built with **FastAPI** backend and **React** frontend using the **MERN** stack architecture.

## 🌟 Features

- **Real-time Recognition**: Live webcam ISL gesture recognition using WebSocket
- **Image Upload**: Upload and analyze ISL gesture images
- **Dashboard**: System statistics and supported gestures overview
- **Responsive UI**: Bootstrap-based responsive design
- **REST API**: FastAPI backend with automatic documentation
- **WebSocket Support**: Real-time communication for live video feed
- **Multi-hand Detection**: Support for single and two-hand gestures

## 🏗️ Architecture

```
ISL_Web_App/
├── backend/                    # FastAPI Backend
│   ├── main.py                # Main FastAPI application
│   └── requirements.txt       # Python dependencies
├── frontend/                  # React Frontend
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── pages/           # Page components
│   │   ├── App.js           # Main App component
│   │   └── index.js         # Entry point
│   ├── public/              # Static files
│   └── package.json         # Node.js dependencies
└── README.md                # This file
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Node.js 16+**
- **npm** or **yarn**
- **ISL Model File**: `ISL_2003_final.h5` (place in `D:\\WSL\\`)

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd ISL_Web_App/backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   .\\venv\\Scripts\\activate  # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Update model path** in `main.py` if needed:
   ```python
   MODEL_PATH = r"D:\\WSL\\ISL_2003_final.h5"  # Update this path
   ```

5. **Start the FastAPI server**:
   ```bash
   python main.py
   ```

   The API will be available at:
   - **API**: http://localhost:8000
   - **Documentation**: http://localhost:8000/docs
   - **WebSocket**: ws://localhost:8000/ws

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd ISL_Web_App/frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the React development server**:
   ```bash
   npm start
   ```

   The web application will be available at:
   - **Web App**: http://localhost:3000

## 📖 Usage

### 1. Dashboard
- **Overview**: System statistics and supported gestures
- **Features**: View total predictions, accuracy, and common gestures
- **Access**: http://localhost:3000/

### 2. Real-time Recognition
- **Live Recognition**: Use webcam for real-time ISL recognition
- **Features**: WebSocket-based live predictions with confidence scores
- **Access**: http://localhost:3000/realtime

### 3. Image Upload
- **Upload Recognition**: Upload images for ISL analysis
- **Features**: Drag & drop interface, prediction history
- **Access**: http://localhost:3000/upload

## 🔌 API Endpoints

### REST API

- **GET** `/` - Health check
- **GET** `/health` - System health and model status
- **POST** `/predict` - Upload image for prediction

### WebSocket

- **WS** `/ws` - Real-time prediction stream

### Example API Usage

```javascript
// Upload image
const formData = new FormData();
formData.append('file', imageFile);

fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));

// WebSocket connection
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
  const prediction = JSON.parse(event.data);
  console.log(prediction);
};
```

## 🛠️ Technology Stack

### Backend (FastAPI)
- **FastAPI**: Modern Python web framework
- **TensorFlow**: ISL model inference
- **MediaPipe**: Hand detection and landmarks
- **OpenCV**: Image processing
- **WebSockets**: Real-time communication
- **Uvicorn**: ASGI server

### Frontend (React)
- **React 18**: Frontend framework
- **React Bootstrap**: UI components
- **React Router**: Navigation
- **React Webcam**: Camera integration
- **Axios**: HTTP client
- **Socket.io**: WebSocket client
- **React Dropzone**: File upload

## 🎯 Model Information

- **Classes**: 36 (Numbers 0-9, Letters A-Z)
- **Input Size**: 128x128 pixels
- **Architecture**: CNN with MediaPipe preprocessing
- **Framework**: TensorFlow/Keras
- **Hand Detection**: MediaPipe Hands solution

## 🔧 Configuration

### Environment Variables

Create `.env` file in backend directory:

```env
MODEL_PATH=D:\\WSL\\ISL_2003_final.h5
API_PORT=8000
CORS_ORIGINS=http://localhost:3000
```

### Camera Settings

Modify camera settings in `RealtimeRecognition.js`:

```javascript
const videoConstraints = {
  width: 640,
  height: 480,
  facingMode: "user"  // or "environment" for back camera
};
```

## 🚀 Deployment

### Backend Deployment

```bash
# Production server
uvicorn main:app --host 0.0.0.0 --port 8000

# With Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Frontend Deployment

```bash
# Build for production
npm run build

# Serve static files
npx serve -s build
```

## 🐛 Troubleshooting

### Common Issues

1. **Model not found**:
   - Ensure `ISL_2003_final.h5` is in the correct path
   - Update `MODEL_PATH` in `main.py`

2. **CORS errors**:
   - Check if backend is running on port 8000
   - Verify CORS origins in FastAPI configuration

3. **WebSocket connection failed**:
   - Ensure backend WebSocket endpoint is accessible
   - Check firewall settings

4. **Camera not working**:
   - Grant camera permissions in browser
   - Try different camera indices if using external cameras

### Logs

- **Backend logs**: Check terminal where `python main.py` is running
- **Frontend logs**: Check browser developer console (F12)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **MediaPipe** for hand detection
- **TensorFlow** for model inference
- **React** community for frontend tools
- **FastAPI** for the excellent backend framework

---

**Happy Coding! 🖐️✨**