import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, Scan, Upload, X, AlertCircle, Info, Activity } from "lucide-react";

const API_BASE = "http://localhost:5001";

// Medical intelligence dictionary
const TUMOR_INFO = {
  glioma: "Gliomas are tumors that occur in the brain and spinal cord, beginning in the supportive glial cells. They display variable growth rates and are monitored closely as they can affect neurological function over time.",
  meningioma: "A meningioma is a tumor that arises from the meninges — the membranes surrounding your brain and spinal cord. Most are slow-growing and benign, but require medical attention if pressing on critical neural pathways.",
  pituitary: "Pituitary tumors are abnormal growths developing in the pituitary gland. They are overwhelmingly noncancerous, but their presence can trigger significant changes in hormone regulation.",
  notumor: "No abnormal tumor mass detected in this MRI analysis. Note: This AI screening is a preliminary tool and cannot replace a comprehensive clinical evaluation. Regular check-ups are always encouraged."
};

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
      setPrediction(json.prediction);
      setConfidence(json.confidence);
      setHeatmapUrl(json.image_url + `?t=${new Date().getTime()}`);

    } catch (err) {
      setError("Failed to reach the deep learning server. Ensure Python API is running on port 5001.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-6 md:p-12 overflow-x-hidden bg-slate-50 text-slate-900 font-sans">
      <div className="max-w-6xl mx-auto space-y-12">
        {/* Header section */}
        <header className="text-center pt-8">
          <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.5 }}>
            <div className="inline-flex items-center justify-center p-3 bg-white border border-slate-200 rounded-2xl mb-6 shadow-sm">
              <Scan className="w-8 h-8 text-blue-600" />
            </div>
          </motion.div>
          <motion.h1 
            initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
            className="text-5xl md:text-6xl font-extrabold tracking-tight text-slate-900 pb-2"
          >
            RayX
          </motion.h1>
          <motion.p 
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}
            className="text-slate-500 mt-4 text-lg font-medium tracking-wide max-w-xl mx-auto"
          >
            Clinical-grade computational analysis of MRI scans using modern Deep Learning.
          </motion.p>
        </header>

        <div className="grid lg:grid-cols-12 gap-8 items-start">
          
          {/* Input Panel */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} className="lg:col-span-5 w-full">
            <div className="bg-white p-8 rounded-[2rem] relative overflow-hidden shadow-md border border-slate-200">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-slate-800 flex items-center gap-3">
                  <Activity className="text-blue-500 w-5 h-5"/> Input Source
                </h2>
              </div>

              {!preview ? (
                <label className="h-72 border-2 border-dashed border-slate-300 bg-slate-50/50 rounded-3xl flex flex-col items-center justify-center cursor-pointer hover:border-blue-400 hover:bg-blue-50/50 transition-all duration-300 overflow-hidden relative group">
                  <div className="p-4 bg-white shadow-sm border border-slate-100 rounded-full mb-4 transform group-hover:-translate-y-2 group-hover:scale-105 transition-all duration-300">
                    <Upload className="w-8 h-8 text-blue-500" />
                  </div>
                  <span className="text-slate-700 font-medium pb-1">Upload DICOM or Image File</span>
                  <span className="text-slate-400 text-xs text-center px-4">Supported formats: PNG, JPG, JPEG (Max 10MB)</span>
                  <input type="file" className="hidden" onChange={handleFileSelect} accept="image/*" />
                </label>
              ) : (
                <div className="relative aspect-square md:aspect-[4/3] bg-slate-100 rounded-3xl overflow-hidden shadow-sm border border-slate-200 group">
                  <img src={preview} className="w-full h-full object-cover" alt="Input Preview" />
                  <button onClick={handleClear} className="absolute top-4 right-4 p-2 bg-white/90 text-red-500 backdrop-blur-md rounded-full hover:bg-red-50 transition-all shadow-sm z-10">
                    <X className="w-5 h-5" />
                  </button>
                </div>
              )}

              <button 
                onClick={handleAnalyze} 
                disabled={!image || loading}
                className="w-full mt-6 py-4 relative overflow-hidden bg-blue-600 text-white rounded-2xl font-semibold tracking-wide transition-all duration-300 shadow hover:shadow-lg hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0 disabled:hover:shadow-none"
              >
                <div className="relative z-10 flex justify-center items-center gap-2">
                  {loading ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white/40 border-t-white rounded-full animate-spin"></div>
                      Processing Diagnostics...
                    </>
                  ) : "Initiate Analysis"}
                </div>
              </button>
            </div>
          </motion.div>

          {/* Results Panel */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }} className="lg:col-span-7 w-full h-full">
            <div className="bg-white p-8 rounded-[2rem] h-full flex flex-col relative overflow-hidden border border-slate-200 shadow-md">
              <h2 className="text-xl font-semibold mb-6 flex items-center gap-3 text-slate-800">
                <Brain className="text-blue-500 w-5 h-5"/> Diagnostics Output
              </h2>
              
              {prediction ? (
                <AnimatePresence>
                  <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} className="flex-1 flex flex-col space-y-6">
                    
                    {/* The Results Banner */}
                    <div className="flex flex-col md:flex-row gap-4 justify-between items-start md:items-end p-6 rounded-2xl bg-blue-50 border border-blue-100 shadow-sm">
                      <div>
                        <p className="text-xs text-blue-600/70 uppercase tracking-widest font-bold mb-1">Detected Signature</p>
                        <p className="text-3xl md:text-4xl font-extrabold capitalize text-blue-900">
                          {prediction === "notumor" ? "Negative" : prediction}
                        </p>
                      </div>
                      <div className="text-left md:text-right">
                        <p className="text-xs text-blue-600/70 uppercase tracking-widest font-bold mb-1">Model Confidence</p>
                        <p className="text-3xl md:text-4xl font-mono text-blue-600 font-bold whitespace-nowrap">
                          {confidence}%
                        </p>
                      </div>
                    </div>

                    {/* Biological context paragraph */}
                    <div className="p-5 rounded-2xl bg-slate-50 border border-slate-100 text-slate-600 text-sm leading-relaxed flex gap-4">
                      <div className="flex-shrink-0 mt-1">
                        <Info className="w-5 h-5 text-blue-500" />
                      </div>
                      <div>
                        <p className="font-semibold text-slate-800 mb-1">Clinical Context</p>
                        <p>{TUMOR_INFO[prediction] || "No biological information available."}</p>
                      </div>
                    </div>

                    {/* The Heatmap Display */}
                    {heatmapUrl && (
                      <div className="flex-1 flex flex-col mt-4">
                        <p className="text-xs text-slate-500 font-bold uppercase tracking-widest mb-3 flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                          Grad-CAM Heatmap Analysis
                        </p>
                        <div className="relative flex-1 rounded-2xl overflow-hidden bg-slate-100 p-1 shadow-sm border border-slate-200">
                          <img src={heatmapUrl} className="w-full h-full object-cover rounded-xl" alt="Heatmap Result" />
                        </div>
                      </div>
                    )}
                  </motion.div>
                </AnimatePresence>
              ) : (
                <div className="flex-1 flex flex-col items-center justify-center opacity-40 min-h-[300px]">
                  <Brain className="w-24 h-24 mb-6 text-slate-400 stroke-[1]" />
                  <p className="tracking-widest uppercase text-sm font-semibold text-slate-500">Awaiting visual input</p>
                </div>
              )}
              
              {/* Error state */}
              {error && (
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mt-6 text-red-700 text-sm bg-red-50 p-4 rounded-xl border border-red-200 flex gap-3 shadow-sm">
                  <AlertCircle className="w-5 h-5 flex-shrink-0" />
                  <p>{error}</p>
                </motion.div>
              )}
            </div>
          </motion.div>

        </div>
      </div>
    </div>
  );
}

export default App;