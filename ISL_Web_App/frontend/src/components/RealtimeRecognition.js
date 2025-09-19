import React, { useState, useRef, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import { toast } from 'react-toastify';

const RealtimeRecognition = () => {
  const webcamRef = useRef(null);
  const wsRef = useRef(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [handsDetected, setHandsDetected] = useState(0);
  const [handType, setHandType] = useState('none');
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);
  const [predictionHistory, setPredictionHistory] = useState([]);

  // WebSocket connection
  const connectWebSocket = useCallback(() => {
    try {
      const ws = new WebSocket('ws://localhost:8000/ws');
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setError(null);
        toast.success('Connected to ISL Recognition server');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.error) {
            setError(data.error);
            return;
          }

          if (data.success) {
            setPrediction(data.prediction);
            setConfidence(data.confidence);
            setHandsDetected(data.hands_detected);
            setHandType(data.hand_type);
            
            // Add to history if confident prediction
            if (data.confidence > 70 && data.prediction !== 'No Hand Detected' && data.prediction !== 'Uncertain') {
              setPredictionHistory(prev => [
                { 
                  prediction: data.prediction, 
                  confidence: data.confidence,
                  timestamp: new Date().toLocaleTimeString(),
                  handType: data.hand_type
                },
                ...prev.slice(0, 9) // Keep last 10 predictions
              ]);
            }
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        if (isCapturing) {
          toast.warning('Connection lost. Attempting to reconnect...');
          setTimeout(connectWebSocket, 2000);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setError('Connection error');
        setIsConnected(false);
      };

      wsRef.current = ws;
    } catch (err) {
      setError('Failed to connect to server');
      setIsConnected(false);
    }
  }, [isCapturing]);

  // Start/Stop capturing
  const toggleCapture = () => {
    if (isCapturing) {
      setIsCapturing(false);
      if (wsRef.current) {
        wsRef.current.close();
      }
      setPrediction(null);
      setConfidence(0);
      setHandsDetected(0);
      setHandType('none');
    } else {
      setIsCapturing(true);
      connectWebSocket();
    }
  };

  // Capture and send frame
  const captureFrame = useCallback(() => {
    if (webcamRef.current && wsRef.current && isCapturing && isConnected) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        try {
          wsRef.current.send(JSON.stringify({ image: imageSrc }));
        } catch (err) {
          console.error('Error sending frame:', err);
        }
      }
    }
  }, [isCapturing, isConnected]);

  // Send frames periodically
  useEffect(() => {
    let interval;
    if (isCapturing && isConnected) {
      interval = setInterval(captureFrame, 100); // 10 FPS
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isCapturing, isConnected, captureFrame]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const getConfidenceColor = (conf) => {
    if (conf >= 80) return 'success';
    if (conf >= 60) return 'warning';
    return 'danger';
  };

  const getHandTypeIcon = (type) => {
    switch(type) {
      case 'single_hand': return '🖐️';
      case 'two_hands': return '🙌';
      default: return '❌';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br text-white">
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent mb-2">
            🎥 Real-time ISL Recognition
          </h1>
          <p className="text-slate-400">Live hand gesture recognition with instant feedback</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Side - Camera Feed */}
          <div className="order-2 lg:order-1">
            <div className="bg-black-30 backdrop-blur-sm border border-blue-500-20 rounded-2xl p-6">
              <h3 className="text-xl font-bold text-blue-300 mb-4">📹 Live Camera Feed</h3>
              <div className="relative">
                <Webcam
                  ref={webcamRef}
                  audio={false}
                  width="100%"
                  height="auto"
                  screenshotFormat="image/jpeg"
                  videoConstraints={{
                    width: 640,
                    height: 480,
                    facingMode: "user"
                  }}
                  className="w-full rounded-lg border border-slate-600"
                  style={{ maxHeight: '400px' }}
                />
                
                {/* Connection status overlay */}
                <div className="absolute top-4 right-4">
                  <div className={`px-3 py-1 rounded-lg backdrop-blur-sm border text-sm ${
                    isConnected 
                      ? 'bg-green-500-20 border-green-500-30 text-green-300' 
                      : 'bg-gray-500-20 border-gray-500-30 text-gray-300'
                  }`}>
                    {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
                  </div>
                </div>
              </div>

              {/* Controls */}
              <div className="flex justify-center mt-6">
                <button 
                  className={`px-8 py-3 rounded-xl font-semibold text-lg transition-all duration-200 transform hover:scale-105 ${
                    isCapturing 
                      ? 'bg-red-500 hover:bg-red-600 text-white shadow-lg hover:shadow-red-500/25' 
                      : 'bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white shadow-lg hover:shadow-green-500/25'
                  } ${(!isConnected && isCapturing) ? 'opacity-50 cursor-not-allowed' : ''}`}
                  onClick={toggleCapture}
                  disabled={!isConnected && isCapturing}
                >
                  {isCapturing ? (
                    <div className="flex items-center space-x-2">
                      <div className="animate-spin rounded-full h-5 w-5 border-t-2 border-b-2 border-white"></div>
                      <span>Stop Recognition</span>
                    </div>
                  ) : (
                    '🚀 Start Recognition'
                  )}
                </button>
              </div>

              {/* Error display */}
              {error && (
                <div className="mt-4 p-4 bg-red-500-20 border border-red-500-30 rounded-lg">
                  <div className="text-red-300 font-semibold">Error:</div>
                  <div className="text-red-200">{error}</div>
                </div>
              )}
            </div>
          </div>

          {/* Right Side - Current Prediction & Info */}
          <div className="order-1 lg:order-2 space-y-6">
            {/* Current Prediction - Large Display */}
            <div className="bg-black-30 backdrop-blur-sm border border-purple-500-30 rounded-2xl p-8">
              <h3 className="text-2xl font-bold text-purple-300 mb-6 text-center">� Current Prediction</h3>
              {prediction ? (
                <div className="text-center">
                  {/* Large Prediction Display */}
                  <div className="mb-6">
                    <div className="text-6xl font-bold text-white mb-4 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                      {prediction}
                    </div>
                  </div>
                  
                  {/* Confidence Bar */}
                  <div className="mb-6">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-slate-400">Confidence</span>
                      <span className="text-sm font-semibold text-white">{confidence.toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-3">
                      <div 
                        className={`h-3 rounded-full transition-all duration-300 ${
                          confidence >= 80 ? 'bg-green-500' :
                          confidence >= 60 ? 'bg-yellow-500' :
                          'bg-red-500'
                        }`}
                        style={{ width: `${confidence}%` }}
                      ></div>
                    </div>
                  </div>

                  {/* Hand Info */}
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div className="bg-slate-800-20 rounded-lg p-4 border border-slate-600">
                      <div className="text-2xl mb-2">{getHandTypeIcon(handType)}</div>
                      <div className="text-sm text-slate-400">Hand Type</div>
                      <div className="font-semibold text-white">{handType.replace('_', ' ')}</div>
                    </div>
                    <div className="bg-slate-800-20 rounded-lg p-4 border border-slate-600">
                      <div className="text-2xl mb-2">🔢</div>
                      <div className="text-sm text-slate-400">Hands Detected</div>
                      <div className="font-semibold text-white">{handsDetected}</div>
                    </div>
                  </div>

                  {/* Status Badge */}
                  <div className="flex justify-center">
                    <span className={`px-4 py-2 rounded-lg text-sm font-medium ${
                      confidence >= 80 ? 'bg-green-500-20 border border-green-500-30 text-green-300' :
                      confidence >= 60 ? 'bg-yellow-500-20 border border-yellow-500-30 text-yellow-300' :
                      'bg-red-500-20 border border-red-500-30 text-red-300'
                    }`}>
                      {confidence >= 80 ? '🎯 High Confidence' :
                       confidence >= 60 ? '⚠️ Medium Confidence' :
                       '❌ Low Confidence'}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="text-center text-slate-400 py-12">
                  <div className="text-6xl mb-4">🖐️</div>
                  <div className="text-xl font-semibold mb-2">Ready for Recognition</div>
                  <p>Start the camera to see live predictions</p>
                  <div className="mt-4 text-sm text-slate-500">
                    Position your hand in front of the camera for best results
                  </div>
                </div>
              )}
            </div>

            {/* Prediction History */}
            <div className="bg-black-30 backdrop-blur-sm border border-cyan-500-20 rounded-2xl p-6">
              <h3 className="text-xl font-bold text-cyan-300 mb-4">📈 Recent Predictions</h3>
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {predictionHistory.length > 0 ? (
                  predictionHistory.map((pred, index) => (
                    <div key={index} className="flex justify-between items-center p-3 bg-slate-800-20 rounded-lg border border-slate-600">
                      <div>
                        <div className="font-semibold text-white">{pred.prediction}</div>
                        <div className="text-xs text-slate-400">{pred.timestamp}</div>
                      </div>
                      <div className="text-right">
                        <div className={`px-2 py-1 rounded text-xs font-medium ${
                          pred.confidence >= 80 ? 'bg-green-500-20 text-green-300' :
                          pred.confidence >= 60 ? 'bg-yellow-500-20 text-yellow-300' :
                          'bg-red-500-20 text-red-300'
                        }`}>
                          {pred.confidence.toFixed(1)}%
                        </div>
                        <div className="text-xs mt-1">{getHandTypeIcon(pred.handType)}</div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center text-slate-400 py-8">
                    <div className="text-3xl mb-2">📋</div>
                    <p>No predictions yet</p>
                    <p className="text-sm">Confident predictions will appear here</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RealtimeRecognition;