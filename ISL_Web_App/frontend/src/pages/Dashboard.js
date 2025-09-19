import React, { useState, useEffect } from 'react';

const Dashboard = () => {
  const [selectedClass, setSelectedClass] = useState(null);
  const [sampleImages, setSampleImages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [imageLoading, setImageLoading] = useState(false);

  // Performance data based on your PDF report
  const performanceData = {
    overall_accuracy: 97.8,
    precision: 96.5,
    recall: 97.2,
    f1_score: 96.8,
    total_classes: 36,
    training_samples: 50000,
    validation_samples: 12000,
    model_architecture: "CNN + MediaPipe",
    input_size: "128x128 pixels",
    framework: "TensorFlow/Keras"
  };

  // Class performance data with grades
  const classPerformance = {
    // Numbers (0-9)
    '0': { accuracy: 98.2, grade: 'A+', color: '#10B981' },
    '1': { accuracy: 97.5, grade: 'A+', color: '#10B981' },
    '2': { accuracy: 96.5, grade: 'A+', color: '#10B981' },
    '3': { accuracy: 95.8, grade: 'A+', color: '#10B981' },
    '4': { accuracy: 94.3, grade: 'A', color: '#059669' },
    '5': { accuracy: 96.0, grade: 'A+', color: '#10B981' },
    '6': { accuracy: 98.5, grade: 'A+', color: '#10B981' },
    '7': { accuracy: 98.9, grade: 'A+', color: '#10B981' },
    '8': { accuracy: 95.9, grade: 'A+', color: '#10B981' },
    '9': { accuracy: 92.1, grade: 'A-', color: '#F59E0B' },
    
    // Letters (A-Z)
    'A': { accuracy: 98.5, grade: 'A+', color: '#10B981' },
    'B': { accuracy: 94.3, grade: 'A', color: '#059669' },
    'C': { accuracy: 96.7, grade: 'A+', color: '#10B981' },
    'D': { accuracy: 95.2, grade: 'A+', color: '#10B981' },
    'E': { accuracy: 93.8, grade: 'A', color: '#059669' },
    'F': { accuracy: 96.1, grade: 'A+', color: '#10B981' },
    'G': { accuracy: 97.4, grade: 'A+', color: '#10B981' },
    'H': { accuracy: 95.9, grade: 'A+', color: '#10B981' },
    'I': { accuracy: 92.6, grade: 'A-', color: '#F59E0B' },
    'J': { accuracy: 94.1, grade: 'A', color: '#059669' },
    'K': { accuracy: 91.3, grade: 'A-', color: '#F59E0B' },
    'L': { accuracy: 95.7, grade: 'A+', color: '#10B981' },
    'M': { accuracy: 89.2, grade: 'B+', color: '#EF4444' },
    'N': { accuracy: 93.4, grade: 'A', color: '#059669' },
    'O': { accuracy: 96.8, grade: 'A+', color: '#10B981' },
    'P': { accuracy: 95.1, grade: 'A+', color: '#10B981' },
    'Q': { accuracy: 92.7, grade: 'A-', color: '#F59E0B' },
    'R': { accuracy: 94.6, grade: 'A', color: '#059669' },
    'S': { accuracy: 93.9, grade: 'A', color: '#059669' },
    'T': { accuracy: 95.3, grade: 'A+', color: '#10B981' },
    'U': { accuracy: 92.0, grade: 'A-', color: '#F59E0B' },
    'V': { accuracy: 93.1, grade: 'A', color: '#059669' },
    'W': { accuracy: 96.0, grade: 'A+', color: '#10B981' },
    'X': { accuracy: 98.6, grade: 'A+', color: '#10B981' },
    'Y': { accuracy: 94.8, grade: 'A', color: '#059669' },
    'Z': { accuracy: 95.9, grade: 'A+', color: '#10B981' },
  };

  useEffect(() => {
    setLoading(false);
  }, []);

  const handleClassClick = async (className) => {
    setSelectedClass(className);
    setImageLoading(true);
    
    try {
      const response = await fetch(`http://localhost:8000/dataset/${className}`);
      const data = await response.json();
      setSampleImages(data.sample_images || []);
    } catch (error) {
      console.error('Error fetching sample images:', error);
      setSampleImages([]);
    } finally {
      setImageLoading(false);
    }
  };

  const closeModal = () => {
    setSelectedClass(null);
    setSampleImages([]);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-purple-500"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br text-white">
      {/* Header */}
      <div className="bg-black-20 backdrop-blur-sm border-b border-purple-500-20">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                🖐️ ISL Recognition Dashboard
              </h1>
              <p className="text-slate-300 mt-2">Advanced Indian Sign Language Recognition System</p>
            </div>
            <div className="text-right">
              <div className="text-3xl font-bold text-green-400">{performanceData.overall_accuracy}%</div>
              <div className="text-sm text-slate-400">Overall Accuracy</div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Performance Overview */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-green-500/10 backdrop-blur-sm border border-green-500-20 rounded-xl p-6">
            <div className="text-green-400 text-sm font-medium">PRECISION</div>
            <div className="text-3xl font-bold text-white mt-2">{performanceData.precision}%</div>
            <div className="text-xs text-green-300 mt-1">↗ Excellent</div>
          </div>
          
          <div className="bg-blue-500/10 backdrop-blur-sm border border-blue-500-20 rounded-xl p-6">
            <div className="text-blue-400 text-sm font-medium">RECALL</div>
            <div className="text-3xl font-bold text-white mt-2">{performanceData.recall}%</div>
            <div className="text-xs text-blue-300 mt-1">↗ Outstanding</div>
          </div>
          
          <div className="bg-purple-500/10 backdrop-blur-sm border border-purple-500-20 rounded-xl p-6">
            <div className="text-purple-400 text-sm font-medium">F1-SCORE</div>
            <div className="text-3xl font-bold text-white mt-2">{performanceData.f1_score}%</div>
            <div className="text-xs text-purple-300 mt-1">↗ Superior</div>
          </div>
          
          <div className="bg-orange-500/10 backdrop-blur-sm border border-orange-500-20 rounded-xl p-6">
            <div className="text-orange-400 text-sm font-medium">CLASSES</div>
            <div className="text-3xl font-bold text-white mt-2">{performanceData.total_classes}</div>
            <div className="text-xs text-orange-300 mt-1">0-9, A-Z</div>
          </div>
        </div>

        {/* Interactive Class Performance */}
        <div className="bg-black-30 backdrop-blur-sm border border-purple-500-30 rounded-2xl p-8 mb-8">
          <h2 className="text-2xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            📊 Interactive Class Performance
          </h2>
          <p className="text-slate-400 mb-6">Click on any letter/number to view sample images from the dataset</p>
          
          {/* Numbers Section */}
          <div className="mb-8">
            <h3 className="text-lg font-semibold text-blue-300 mb-4">🔢 Numbers (0-9)</h3>
            <div className="grid grid-cols-5 md:grid-cols-10 gap-3">
              {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(num => {
                const perf = classPerformance[num.toString()];
                return (
                  <div 
                    key={num}
                    onClick={() => handleClassClick(num.toString())}
                    className="relative group cursor-pointer transform hover:scale-105 transition-all duration-200"
                  >
                    <div 
                      className="h-20 w-20 rounded-xl flex flex-col items-center justify-center text-white font-bold text-lg border-2 shadow-lg hover:shadow-xl"
                      style={{ 
                        backgroundColor: `${perf.color}20`,
                        borderColor: perf.color
                      }}
                    >
                      <div>{num}</div>
                      <div className="text-xs">{perf.grade}</div>
                    </div>
                    <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-black text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity">
                      {perf.accuracy}%
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Letters Section */}
          <div>
            <h3 className="text-lg font-semibold text-green-300 mb-4">🔤 Letters (A-Z)</h3>
            <div className="grid grid-cols-6 md:grid-cols-13 gap-3">
              {Array.from({length: 26}, (_, i) => {
                const letter = String.fromCharCode(65 + i);
                const perf = classPerformance[letter];
                return (
                  <div 
                    key={letter}
                    onClick={() => handleClassClick(letter)}
                    className="relative group cursor-pointer transform hover:scale-105 transition-all duration-200"
                  >
                    <div 
                      className="h-20 w-20 rounded-xl flex flex-col items-center justify-center text-white font-bold text-lg border-2 shadow-lg hover:shadow-xl"
                      style={{ 
                        backgroundColor: `${perf.color}20`,
                        borderColor: perf.color
                      }}
                    >
                      <div>{letter}</div>
                      <div className="text-xs">{perf.grade}</div>
                    </div>
                    <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-black text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity">
                      {perf.accuracy}%
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* System Information */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Model Architecture */}
          <div className="bg-black-30 backdrop-blur-sm border border-cyan-500-30 rounded-2xl p-6">
            <h3 className="text-xl font-bold mb-4 text-cyan-300">🏗️ Model Architecture</h3>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-400">Framework:</span>
                <span className="text-white">{performanceData.framework}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Architecture:</span>
                <span className="text-white">{performanceData.model_architecture}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Input Size:</span>
                <span className="text-white">{performanceData.input_size}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Training Samples:</span>
                <span className="text-white">{performanceData.training_samples.toLocaleString()}+</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Validation Samples:</span>
                <span className="text-white">{performanceData.validation_samples.toLocaleString()}+</span>
              </div>
            </div>
          </div>

          {/* Key Features */}
          <div className="bg-black-30 backdrop-blur-sm border border-green-500-30 rounded-2xl p-6">
            <h3 className="text-xl font-bold mb-4 text-green-300">✨ Key Features</h3>
            <div className="space-y-3">
              {[
                'Real-time hand detection',
                'Single & two-hand gestures', 
                'High accuracy recognition',
                'Live camera integration',
                'Confidence scoring',
                'Multi-camera support'
              ].map((feature, index) => (
                <div key={index} className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                  <span className="text-slate-300">{feature}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Performance Metrics Summary */}
        <div className="mt-8 bg-black-30 backdrop-blur-sm border border-yellow-500-30 rounded-2xl p-6">
          <h3 className="text-xl font-bold mb-4 text-yellow-300">📈 Performance Summary</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div>
              <div className="text-green-400 text-2xl font-bold">A (98.5%)</div>
              <div className="text-xs text-slate-400">Best Class</div>
            </div>
            <div>
              <div className="text-orange-400 text-2xl font-bold">M (89.2%)</div>
              <div className="text-xs text-slate-400">Challenging Class</div>
            </div>
            <div>
              <div className="text-blue-400 text-2xl font-bold">25/36</div>
              <div className="text-xs text-slate-400">A+ Grade Classes</div>
            </div>
            <div>
              <div className="text-purple-400 text-2xl font-bold">97.8%</div>
              <div className="text-xs text-slate-400">Average Performance</div>
            </div>
          </div>
        </div>
      </div>

      {/* Sample Images Modal */}
      {selectedClass && (
        <div className="fixed inset-0 bg-black-80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-slate-800 rounded-2xl p-6 max-w-4xl w-full max-h-80vh overflow-y-auto border border-purple-500-30">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-2xl font-bold text-white">
                📸 Sample Images for "{selectedClass}"
              </h3>
              <button 
                onClick={closeModal}
                className="text-gray-400 hover:text-white text-2xl"
              >
                ×
              </button>
            </div>
            
            {imageLoading ? (
              <div className="flex justify-center py-12">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500"></div>
              </div>
            ) : sampleImages.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {sampleImages.map((image, index) => (
                  <div key={index} className="relative group">
                    <div className="bg-slate-800-30 rounded-lg p-3 border border-slate-600 group-hover:border-purple-400 transition-colors">
                      <img 
                        src={`http://localhost:8000/dataset/${selectedClass}/image/${image}`}
                        alt={`Sample ${selectedClass} ${index + 1}`}
                        className="w-full h-auto object-contain rounded-lg bg-white"
                        style={{ 
                          maxHeight: '250px',
                          minHeight: '150px'
                        }}
                        onError={(e) => {
                          e.target.style.display = 'none';
                        }}
                      />
                      <div className="mt-2 text-center">
                        <span className="text-xs text-slate-400 bg-slate-700 px-2 py-1 rounded">
                          {image}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-12 text-slate-400">
                <div className="text-6xl mb-4">📁</div>
                <div className="text-lg">No sample images available</div>
                <div className="text-sm">Sample images for "{selectedClass}" could not be loaded from the dataset.</div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;