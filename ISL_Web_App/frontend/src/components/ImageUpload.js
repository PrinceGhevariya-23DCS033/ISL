import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { toast } from 'react-toastify';

const ImageUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [uploadHistory, setUploadHistory] = useState([]);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      setResult(null);
      setError(null);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.bmp', '.webp']
    },
    multiple: false,
    maxSize: 10 * 1024 * 1024 // 10MB
  });

  const uploadImage = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await axios.post('http://localhost:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResult(response.data);
      
      // Add to history
      const newResult = {
        ...response.data,
        filename: selectedFile.name,
        timestamp: new Date().toLocaleTimeString(),
        preview: preview
      };
      
      setUploadHistory(prev => [newResult, ...prev.slice(0, 9)]); // Keep last 10
      
      toast.success(`Prediction: ${response.data.prediction} (${response.data.confidence.toFixed(1)}%)`);
      
    } catch (err) {
      const errorMsg = err.response?.data?.detail || 'Upload failed';
      setError(errorMsg);
      toast.error(errorMsg);
    } finally {
      setIsUploading(false);
    }
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const getConfidenceColor = (conf) => {
    if (conf >= 80) return 'success';
    if (conf >= 60) return 'warning';
    return 'danger';
  };

  const getHandTypeIcon = (type) => {
    switch(type) {
      case 'single_hand': return '🖐️';
      case 'two_hands': return '🙌';
      default: return '❓';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br text-white">
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent mb-2">
            📷 Image Upload Recognition
          </h1>
          <p className="text-slate-400">Upload images for instant ISL gesture analysis</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Upload Area */}
          <div className="lg:col-span-2">
            <div className="bg-black-30 backdrop-blur-sm border border-green-500-20 rounded-2xl p-6">
              {/* Dropzone */}
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-xl p-8 text-center mb-6 transition-all duration-200 cursor-pointer ${
                  isDragActive 
                    ? 'border-green-400 bg-green-500-10' 
                    : 'border-slate-600 hover:border-green-500 hover:bg-green-500-5'
                }`}
                style={{ minHeight: '200px' }}
              >
                <input {...getInputProps()} />
                {isDragActive ? (
                  <div className="flex flex-col items-center justify-center h-full">
                    <div className="text-6xl mb-4">📤</div>
                    <h3 className="text-2xl font-bold text-green-400">Drop the image here...</h3>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center h-full">
                    <div className="text-6xl mb-4">🖼️</div>
                    <h3 className="text-2xl font-bold text-white mb-2">Drag & Drop an Image</h3>
                    <p className="text-slate-400 mb-4">or click to select a file</p>
                    <div className="text-sm text-slate-500">
                      Supported: JPEG, PNG, BMP, WEBP (Max 10MB)
                    </div>
                  </div>
                )}
              </div>

              {/* Preview and Results */}
              {preview && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                  {/* Image Preview */}
                  <div className="bg-slate-800-30 rounded-xl p-4">
                    <h4 className="text-lg font-semibold text-white mb-3">📁 Selected Image</h4>
                    <div className="text-center">
                      <img
                        src={preview}
                        alt="Preview"
                        className="w-full h-64 object-cover rounded-lg border border-slate-600"
                      />
                      <div className="mt-2 text-sm text-slate-400">{selectedFile?.name}</div>
                    </div>
                  </div>
                  
                  {/* Results */}
                  {result && (
                    <div className="bg-slate-800-30 rounded-xl p-4">
                      <h4 className="text-lg font-semibold text-white mb-3">🎯 Prediction Result</h4>
                      <div className="text-center">
                        <div className="text-3xl font-bold text-white mb-4">{result.prediction}</div>
                        <div className="mb-3">
                          <span className={`px-4 py-2 rounded-lg text-sm font-medium ${
                            result.confidence >= 80 ? 'bg-green-500-20 text-green-300' :
                            result.confidence >= 60 ? 'bg-yellow-500-20 text-yellow-300' :
                            'bg-red-500-20 text-red-300'
                          }`}>
                            {result.confidence.toFixed(1)}% Confidence
                          </span>
                        </div>
                        <div className="mb-3">
                          <span className="px-3 py-1 rounded-lg text-sm bg-slate-700 text-slate-300">
                            {getHandTypeIcon(result.hand_type)} {result.hands_detected} Hand(s)
                          </span>
                        </div>
                        <div className="text-sm text-slate-400">
                          Hand Type: {result.hand_type?.replace('_', ' ') || 'Unknown'}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Controls */}
              <div className="flex justify-center space-x-4">
                <button
                  className={`px-8 py-3 rounded-xl font-semibold text-lg transition-all duration-200 transform hover:scale-105 ${
                    !selectedFile || isUploading
                      ? 'bg-gray-600 text-gray-400 cursor-not-allowed' 
                      : 'bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white shadow-lg hover:shadow-blue-500/25'
                  }`}
                  onClick={uploadImage}
                  disabled={!selectedFile || isUploading}
                >
                  {isUploading ? (
                    <div className="flex items-center space-x-2">
                      <div className="animate-spin rounded-full h-5 w-5 border-t-2 border-b-2 border-white"></div>
                      <span>Analyzing...</span>
                    </div>
                  ) : (
                    '🔍 Analyze Image'
                  )}
                </button>
                
                {selectedFile && (
                  <button
                    className={`px-6 py-3 rounded-xl font-semibold text-lg transition-all duration-200 border border-slate-600 hover:border-slate-500 ${
                      isUploading 
                        ? 'text-gray-400 cursor-not-allowed' 
                        : 'text-slate-300 hover:text-white hover:bg-slate-800'
                    }`}
                    onClick={clearSelection}
                    disabled={isUploading}
                  >
                    🗑️ Clear
                  </button>
                )}
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

          {/* Upload History Sidebar */}
          <div>
            <div className="bg-black-30 backdrop-blur-sm border border-purple-500-20 rounded-2xl p-6">
              <h3 className="text-xl font-bold text-purple-300 mb-4">📊 Upload History</h3>
              <div className="space-y-4 max-h-96 overflow-y-auto">
                {uploadHistory.length > 0 ? (
                  uploadHistory.map((item, index) => (
                    <div key={index} className="bg-slate-800-30 rounded-lg p-3 border border-slate-600">
                      <div className="flex space-x-3">
                        <img
                          src={item.preview}
                          alt={item.filename}
                          className="w-16 h-16 object-cover rounded-lg border border-slate-600"
                        />
                        <div className="flex-1 min-w-0">
                          <div className="font-semibold text-white text-sm">{item.prediction}</div>
                          <div className="flex items-center space-x-2 mt-1">
                            <span className={`px-2 py-1 rounded text-xs font-medium ${
                              item.confidence >= 80 ? 'bg-green-500-20 text-green-300' :
                              item.confidence >= 60 ? 'bg-yellow-500-20 text-yellow-300' :
                              'bg-red-500-20 text-red-300'
                            }`}>
                              {item.confidence.toFixed(1)}%
                            </span>
                            <span className="px-2 py-1 rounded text-xs bg-slate-700 text-slate-300">
                              {getHandTypeIcon(item.hand_type)} {item.hands_detected}
                            </span>
                          </div>
                          <div className="text-xs text-slate-400 mt-1">{item.timestamp}</div>
                          <div className="text-xs text-slate-500 truncate">
                            {item.filename.length > 20 
                              ? `${item.filename.substring(0, 20)}...` 
                              : item.filename
                            }
                          </div>
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center text-slate-400 py-8">
                    <div className="text-4xl mb-2">📁</div>
                    <p>No uploads yet</p>
                    <p className="text-sm">Upload images to see results here</p>
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

export default ImageUpload;