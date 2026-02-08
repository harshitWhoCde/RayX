import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, Scan, Shield, Zap, Upload, X, AlertCircle } from "lucide-react";

const API_BASE = "http://localhost:5001";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [heatmapUrl, setHeatmapUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileSelect = (e) => {
    if (e.target.files?.[0]) {
      const file = e.target.files[0];
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
      setHeatmapUrl(null);
    }
  };

  const handleClear = () => {
    setImage(null);
    setPreview(null);
    setPrediction(null);
    setHeatmapUrl(null);
  };

  const handleAnalyze = async () => {
    if (!image) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("image", image);

    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Backend connection failed");

      const json = await res.json();
      
      if (json.image) {
        setPrediction(json.prediction);
        setConfidence(json.confidence);
        
        // Encode the filename to handle spaces/special characters safely
        const safeFile = encodeURIComponent(json.image);
        const url = `${API_BASE}/files/${safeFile}?t=${new Date().getTime()}`;
        
        console.log("Loading heatmap from:", url);
        setHeatmapUrl(url);
      }
    } catch (err) {
      setError("Failed to reach the server. Is Python running on 5001?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-8">
      <div className="max-w-5xl mx-auto space-y-8">
        <header className="text-center">
          <h1 className="text-4xl font-bold text-cyan-400">Brain Tumor Detection</h1>
          <p className="text-slate-400">AI Deep Learning Analysis</p>
        </header>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Upload Card */}
          <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2"><Scan className="text-cyan-400"/> MRI Input</h2>
            {!preview ? (
              <label className="h-64 border-2 border-dashed border-slate-700 rounded-xl flex flex-col items-center justify-center cursor-pointer hover:border-cyan-500/50 transition-colors">
                <Upload className="w-10 h-10 text-slate-600 mb-2" />
                <span className="text-slate-400">Click to upload MRI</span>
                <input type="file" className="hidden" onChange={handleFileSelect} accept="image/*" />
              </label>
            ) : (
              <div className="relative aspect-video bg-black rounded-xl overflow-hidden">
                <img src={preview} className="w-full h-full object-contain" alt="Input Preview" />
                <button onClick={handleClear} className="absolute top-2 right-2 p-1.5 bg-red-500/20 text-red-400 rounded-full hover:bg-red-500 hover:text-white transition-all">
                  <X className="w-4 h-4" />
                </button>
              </div>
            )}
            <button 
              onClick={handleAnalyze} 
              disabled={!image || loading}
              className="w-full mt-4 py-4 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-800 rounded-xl font-bold transition-all"
            >
              {loading ? "Analyzing..." : "RUN ANALYSIS"}
            </button>
          </div>

          {/* Results Card */}
          <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 min-h-[400px]">
            <h2 className="text-xl font-semibold mb-6 flex items-center gap-2"><Brain className="text-cyan-400"/> Results</h2>
            {prediction ? (
              <div className="space-y-6">
                <div className="flex justify-between items-end border-b border-slate-800 pb-4">
                  <div>
                    <p className="text-xs text-slate-500 uppercase">Diagnosis</p>
                    <p className="text-2xl font-bold capitalize">{prediction}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-slate-500 uppercase">Confidence</p>
                    <p className="text-2xl font-mono text-cyan-400 font-bold">{confidence}%</p>
                  </div>
                </div>

                {heatmapUrl && (
                  <div className="space-y-2">
                    <p className="text-xs text-slate-500 font-bold uppercase tracking-widest">Activation Map (Grad-CAM)</p>
                    <div className="aspect-video rounded-xl overflow-hidden bg-black border border-cyan-500/20">
                      <img src={heatmapUrl} className="w-full h-full object-contain" alt="Heatmap Result" />
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="h-64 flex flex-col items-center justify-center opacity-40">
                <Brain className="w-16 h-16 mb-4" />
                <p>Waiting for MRI scan...</p>
              </div>
            )}
            {error && <p className="text-red-400 mt-4 text-sm bg-red-500/10 p-3 rounded-lg border border-red-500/20">{error}</p>}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;