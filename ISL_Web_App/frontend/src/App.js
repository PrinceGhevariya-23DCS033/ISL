import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

import Dashboard from './pages/Dashboard';
import RealtimeRecognition from './components/RealtimeRecognition';
import ImageUpload from './components/ImageUpload';

const Navigation = () => {
  const location = useLocation();
  
  const isActive = (path) => {
    return location.pathname === path;
  };
  
  return (
    <nav className="bg-black-20 backdrop-blur-sm border-b border-purple-500-20 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <Link 
              to="/" 
              className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent"
            >
              🖐️ ISL Recognition
            </Link>
          </div>
          
          <div className="flex space-x-4">
            <Link
              to="/"
              className={`px-4 py-2 rounded-lg transition-all duration-200 ${
                isActive('/') 
                  ? 'bg-purple-500-20 text-purple-300 border border-purple-500-30' 
                  : 'text-slate-300 hover:text-white hover:bg-white-10'
              }`}
            >
              📊 Dashboard
            </Link>
            <Link
              to="/realtime"
              className={`px-4 py-2 rounded-lg transition-all duration-200 ${
                isActive('/realtime') 
                  ? 'bg-purple-500-20 text-purple-300 border border-purple-500-30' 
                  : 'text-slate-300 hover:text-white hover:bg-white-10'
              }`}
            >
              📹 Real-time
            </Link>
            <Link
              to="/upload"
              className={`px-4 py-2 rounded-lg transition-all duration-200 ${
                isActive('/upload') 
                  ? 'bg-purple-500-20 text-purple-300 border border-purple-500-30' 
                  : 'text-slate-300 hover:text-white hover:bg-white-10'
              }`}
            >
              📤 Upload
            </Link>
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="px-4 py-2 rounded-lg text-slate-300 hover:text-white hover:bg-white-10 transition-all duration-200"
            >
              📖 API Docs
            </a>
          </div>
        </div>
      </div>
    </nav>
  );
};

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br text-white">
        <Navigation />
        
        <main>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/realtime" element={<RealtimeRecognition />} />
            <Route path="/upload" element={<ImageUpload />} />
          </Routes>
        </main>

        <footer className="bg-black-20 backdrop-blur-sm border-t border-purple-500-20 py-6 mt-8">
          <div className="max-w-7xl mx-auto px-6 text-center">
            <p className="text-slate-400">
              ISL Recognition System - Built with React & FastAPI
            </p>
            <div className="mt-2 text-sm text-slate-500">
              Advanced Indian Sign Language Recognition using Deep Learning
            </div>
          </div>
        </footer>

        <ToastContainer
          position="top-right"
          autoClose={3000}
          hideProgressBar={false}
          newestOnTop={false}
          closeOnClick
          rtl={false}
          pauseOnFocusLoss
          draggable
          pauseOnHover
          className="mt-16"
        />
      </div>
    </Router>
  );
}

export default App;