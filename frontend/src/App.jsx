import React, { useState, useEffect } from 'react';
import { Bot, FileText, Image as ImageIcon, Send, Upload, Settings, PanelLeftClose, PanelLeftOpen, Search, Link as LinkIcon, Sparkles } from 'lucide-react';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [prompt, setPrompt] = useState('');
  const [images, setImages] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [backendStatus, setBackendStatus] = useState('Checking...');
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [sourceFiles, setSourceFiles] = useState([]);

  // Check backend health on load
  useEffect(() => {
    axios.get(`${API_URL}/`)
      .then(() => setBackendStatus('Connected'))
      .catch(() => setBackendStatus('Disconnected'));
  }, []);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setIsUploading(true);
    try {
      const response = await axios.post(`${API_URL}/api/upload-script`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setAnalysisResults(response.data.scenes);
      setSourceFiles(prev => [
        { id: Date.now(), name: response.data.filename, sceneCount: response.data.scenes.length },
        ...prev
      ]);
    } catch (error) {
      console.error('Upload failed:', error);
      alert('Upload failed: ' + (error.response?.data?.detail || error.message));
    } finally {
      setIsUploading(false);
      event.target.value = null; // reset input
    }
  };

  const handleGenerate = async () => {
    if (!prompt.trim()) return;

    setIsGenerating(true);
    const currentPrompt = prompt;
    setPrompt('');

    try {
      // 1. Send generation request
      const startRes = await axios.post(`${API_URL}/api/generate`, { prompt: currentPrompt });
      const taskId = startRes.data.task_id;

      // 2. Poll for status
      const pollInterval = setInterval(async () => {
        try {
          const statusRes = await axios.get(`${API_URL}/api/task/${taskId}`);

          if (statusRes.data.status === 'completed') {
            clearInterval(pollInterval);
            setImages(prev => [...prev, {
              id: taskId,
              url: statusRes.data.result.image_url.startsWith('http') ? statusRes.data.result.image_url : `${API_URL}${statusRes.data.result.image_url}`,
              prompt: statusRes.data.result.enhanced_prompt,
              rag_context: statusRes.data.result.rag_context
            }]);
            setIsGenerating(false);
          } else if (statusRes.data.status === 'failed') {
            clearInterval(pollInterval);
            console.error("Generation failed:", statusRes.data.result.error);
            setIsGenerating(false);
            alert("Generation failed. Check console.");
          }
        } catch (pollErr) {
          console.error("Polling error:", pollErr);
          clearInterval(pollInterval);
          setIsGenerating(false);
        }
      }, 2000); // Poll every 2 seconds

    } catch (err) {
      console.error("Failed to start generation", err);
      setIsGenerating(false);
      alert("Failed to connect to backend");
    }
  };

  const handleExportHighRes = async (imgId) => {
    setImages(prev => prev.map(img => img.id === imgId ? { ...img, isUpscaling: true } : img));
    try {
      await axios.post(`${API_URL}/api/upscale`, { task_id: imgId });

      const pollInterval = setInterval(async () => {
        try {
          const statusRes = await axios.get(`${API_URL}/api/task/${imgId}`);
          if (statusRes.data.status === 'upscaled') {
            clearInterval(pollInterval);
            const newUrl = statusRes.data.result.upscaled_image_url;
            const absoluteUrl = newUrl.startsWith('http') ? newUrl : `${API_URL}${newUrl}`;

            // Auto trigger download
            const link = document.createElement('a');
            link.href = absoluteUrl;
            link.download = `HighRes_Frame_${imgId}.png`;
            link.target = "_blank";
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            // Update image source to the downloaded high-res image
            setImages(prev => prev.map(img => img.id === imgId ? { ...img, isUpscaling: false, url: absoluteUrl } : img));
          } else if (statusRes.data.status === 'upscale_failed') {
            clearInterval(pollInterval);
            console.error("Upscale failed:", statusRes.data.result.error);
            setImages(prev => prev.map(img => img.id === imgId ? { ...img, isUpscaling: false } : img));
            alert("Upscale failed. Check console.");
          }
        } catch (pollErr) {
          console.error("Polling error for upscaling:", pollErr);
          clearInterval(pollInterval);
          setImages(prev => prev.map(img => img.id === imgId ? { ...img, isUpscaling: false } : img));
        }
      }, 2000);

    } catch (err) {
      console.error("Failed to start upscale", err);
      setImages(prev => prev.map(img => img.id === imgId ? { ...img, isUpscaling: false } : img));
      alert("Failed to connect to backend for upscaling");
    }
  };

  return (
    <div className="flex h-screen bg-[#050810] text-slate-200 overflow-hidden font-sans selection:bg-cyan-900 selection:text-cyan-100">

      {/* Sidebar - Source Documents */}
      <div
        className={`${sidebarOpen ? 'w-64' : 'w-0'} transition-all duration-300 ease-in-out border-r border-[#0A1A2F]/50 bg-gradient-to-b from-[#0A1A2F]/80 to-[#050810] flex flex-col relative overflow-hidden`}
      >
        {/* Abstract Geometry Top */}
        <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-900/10 rounded-full blur-3xl" />

        <div className="p-4 border-b border-cyan-900/20 flex items-center justify-between z-10">
          <h2 className="text-xs font-semibold text-cyan-500/80 uppercase tracking-widest">Sources</h2>
          <div>
            <input type="file" accept="application/pdf" className="hidden" id="script-upload" onChange={handleFileUpload} />
            <label htmlFor="script-upload" className="cursor-pointer flex p-1.5 hover:bg-cyan-900/40 rounded-md text-cyan-400/70 transition-colors">
              <Upload size={16} />
            </label>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-3 z-10">
          {sourceFiles.length === 0 ? (
            <div className="text-center p-4 border border-dashed border-cyan-900/40 rounded-lg bg-[#0A0F1E]/50">
              <p className="text-xs text-cyan-600/70 font-medium">No scripts uploaded yet</p>
            </div>
          ) : (
            sourceFiles.map((src) => (
              <div key={src.id} className="group bg-[#0A0F1E]/80 p-3 rounded-lg border border-cyan-900/30 hover:border-cyan-400 hover:shadow-[0_0_12px_rgba(0,255,255,0.15)] cursor-pointer transition-all flex items-start gap-3 relative overflow-hidden">
                {/* Glow on hover */}
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-cyan-400/5 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700 ease-in-out" />
                <FileText className="text-cyan-400 mt-0.5 shrink-0" size={16} />
                <div className="overflow-hidden">
                  <p className="text-sm text-slate-200 font-medium group-hover:text-cyan-50 transition-colors truncate">{src.name}</p>
                  <p className="text-[10px] text-cyan-600/70 mt-1 uppercase tracking-wider font-semibold">{src.sceneCount} scenes extracted</p>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Settings/Info Area */}
        <div className="p-4 border-t border-cyan-900/20 bg-[#050810]/80 z-10">
          <div className="flex justify-between items-center text-sm">
            <div className="flex items-center gap-2 text-slate-400 hover:text-cyan-300 cursor-pointer transition-colors group">
              <Settings size={16} className="group-hover:rotate-45 transition-transform duration-300" />
              <span className="font-medium text-xs">Settings</span>
            </div>
            <span className="text-[10px] flex items-center gap-1.5 text-cyan-600/60 uppercase font-semibold">
              RTX ON
            </span>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col h-full overflow-hidden relative">

        {/* Header */}
        <header className="h-14 border-b border-[#0A1A2F]/50 flex items-center justify-between px-4 bg-[#050810]/50 backdrop-blur-md z-20 shrink-0">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-1.5 hover:bg-cyan-900/30 border border-transparent hover:border-cyan-900/50 rounded-md text-cyan-500 transition-all"
            >
              {sidebarOpen ? <PanelLeftClose size={18} /> : <PanelLeftOpen size={18} />}
            </button>
            <div className="flex items-center gap-2">
              <Sparkles className="text-cyan-400" size={18} />
              <h1 className="font-semibold text-transparent bg-clip-text bg-gradient-agentic drop-shadow-sm tracking-wide">
                ITS TV Storyboard AI
              </h1>
            </div>
          </div>

          <div className="flex items-center gap-4 text-sm">
            <span className="text-slate-400 hover:text-cyan-100 cursor-pointer text-xs font-medium transition-colors">Sign in</span>
            <div className="w-px h-4 bg-cyan-900/50"></div>
            <span className="flex items-center gap-2 text-cyan-100 bg-[#0A1A2F]/40 px-2.5 py-1 rounded-full border border-emerald-400/20 text-[11px] font-semibold tracking-wide shadow-[0_0_10px_rgba(0,255,191,0.05)]">
              <span className={`w-1.5 h-1.5 rounded-full ${backendStatus === 'Connected' ? 'bg-emerald-400 shadow-[0_0_8px_rgba(0,255,191,0.8)] animate-pulse-slow' : 'bg-rose-500'}`}></span>
              {backendStatus === 'Connected' ? 'Backend Active' : 'Offline'}
            </span>
          </div>
        </header>

        {/* Workspace Area - Split View */}
        <div className="flex-1 flex overflow-hidden">

          {/* Analysis/Chat Area */}
          <div className="w-1/2 flex flex-col min-w-[350px] border-r border-[#0A1A2F] bg-[#050810] relative">

            {/* Background Ambient Glow */}
            <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-blue-900/10 rounded-full blur-3xl pointer-events-none" />

            {/* Messages/Analysis Output */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6 z-10 w-full relative">
              {isUploading ? (
                <div className="flex flex-col items-center justify-center h-full w-full mt-20">
                  <div className="w-10 h-10 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin mb-4" />
                  <p className="text-cyan-400 text-sm font-medium animate-pulse tracking-wide uppercase">Analyzing Script with LLM...</p>
                  <p className="text-xs text-cyan-600 mt-2 font-light">Parsing scenes via Llama3...</p>
                </div>
              ) : analysisResults ? (
                <div className="flex flex-col w-full">
                  <h3 className="text-xs font-semibold text-cyan-500/80 uppercase tracking-widest mb-4 sticky top-0 bg-[#050810] py-2 z-20 border-b border-cyan-900/30">
                    Extracted Scenes ({analysisResults.length})
                  </h3>
                  <div className="space-y-3">
                    {analysisResults.map((scene, idx) => (
                      <div
                        key={idx}
                        className="bg-[#0A0F1E]/80 border border-cyan-800/40 p-4 rounded-xl shadow-md cursor-pointer hover:border-cyan-400/50 hover:shadow-glow-subtle transition-all group relative overflow-hidden"
                        onClick={() => setPrompt(`Scene ${scene.scene_no} at ${scene.location}: ${scene.description}. Shot: ${scene.shot_type}`)}
                      >
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-cyan-400/5 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700 ease-in-out" />
                        <div className="flex justify-between items-start mb-3 relative z-10">
                          <span className="text-xs font-bold px-2 py-1 bg-cyan-900/50 text-cyan-300 rounded-md border border-cyan-700/50">
                            SCENE {scene.scene_no}
                          </span>
                          <span className="text-[10px] tracking-wider uppercase bg-emerald-900/20 border border-emerald-500/30 text-emerald-400 px-2 py-1 rounded">
                            {scene.shot_type}
                          </span>
                        </div>
                        <p className="text-sm text-cyan-100 font-medium mb-1 relative z-10 flex items-center gap-2">
                          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span> {scene.location}
                        </p>
                        <p className="text-sm text-cyan-200/70 leading-relaxed font-light relative z-10">
                          {scene.description}
                        </p>
                        <p className="text-[10px] text-cyan-500/0 mt-2 group-hover:text-cyan-500/70 transition-colors uppercase tracking-widest font-semibold relative z-10">
                          Click to use as prompt
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center w-full mt-6">

                  {/* Welcome Card */}
                  <div className="bg-[#0A0F1E] p-6 rounded-2xl border border-cyan-800/40 shadow-glow-subtle relative overflow-hidden group w-full max-w-lg">
                    {/* Glowing Border effect */}
                    <div className="absolute inset-0 border-2 rounded-2xl border-transparent bg-gradient-to-br from-blue-600/30 via-cyan-400/20 to-emerald-400/10 pointer-events-none opacity-50 group-hover:opacity-100 transition-opacity" style={{ maskImage: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)', maskComposite: 'exclude', WebkitMaskComposite: 'xor', padding: '1px' }}></div>

                    <div className="flex items-start gap-4 relative z-10">
                      <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-900 to-cyan-900 flex items-center justify-center border border-cyan-500/30 shadow-[0_0_15px_rgba(0,255,255,0.2)]">
                        <Bot size={20} className="text-cyan-300" />
                      </div>
                      <div>
                        <h3 className="text-sm font-semibold text-cyan-50 mb-1">System Initialize</h3>
                        <p className="text-xs text-cyan-200/60 leading-relaxed font-light">
                          Welcome to the ITS TV Storyboard Generator. Agentic mode engaged.
                          Upload your script in the sources panel, or simply type a prompt below to initiate autonomous scene generation.
                        </p>
                      </div>
                    </div>
                  </div>

                </div>
              )}
            </div>

            {/* Input Area */}
            <div className="p-6 bg-[#050810] z-10 shrink-0">
              <div className="relative flex items-center bg-[#0A0F1E] border border-cyan-900/50 rounded-2xl p-2 shadow-[inset_0_2px_10px_rgba(0,255,255,0.02)] focus-within:border-cyan-400/50 focus-within:shadow-[0_0_15px_rgba(0,255,255,0.1),inset_0_2px_15px_rgba(0,255,255,0.05)] transition-all ease-out duration-300 ring-1 ring-black/50">
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Describe a scene to generate..."
                  className="w-full h-[40px] max-h-[160px] bg-transparent resize-none py-2.5 px-3 text-sm text-cyan-50 placeholder-cyan-700 outline-none leading-relaxed font-light"
                  rows={1}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleGenerate();
                    }
                  }}
                />
                <button
                  onClick={handleGenerate}
                  disabled={isGenerating || !prompt.trim()}
                  className={`p-2.5 rounded-xl flex items-center justify-center shrink-0 ml-2 transition-all duration-300
                    ${prompt.trim() && !isGenerating
                      ? 'bg-gradient-agentic text-slate-900 shadow-glow-emerald hover:brightness-110'
                      : 'bg-[#131b2c] text-cyan-900/50 cursor-not-allowed border border-cyan-900/20'}`}
                >
                  {isGenerating ? (
                    <div className="w-4 h-4 border-2 border-slate-900 border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <Send size={16} className={prompt.trim() ? 'translate-x-[1px] translate-y-[-1px]' : ''} />
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Canvas Area */}
          <div className="flex-1 bg-[#020409] flex flex-col relative overflow-hidden">
            {/* Background pattern */}
            <div className="absolute inset-0 opacity-[0.03] pointer-events-none" style={{ backgroundImage: 'radial-gradient(circle at 2px 2px, #00FFFF 1px, transparent 0)', backgroundSize: '32px 32px' }}></div>

            {/* Canvas Header */}
            <div className="absolute top-0 left-0 right-0 p-4 flex justify-between items-center z-10 bg-gradient-to-b from-[#020409] to-transparent pointer-events-none">
              <h3 className="text-xs font-semibold uppercase tracking-widest text-cyan-600/50 flex items-center gap-2 drop-shadow-md">
                <ImageIcon size={14} /> Output Viewer
              </h3>
            </div>

            {/* Visualizer Area */}
            <div className="flex-1 p-8 overflow-y-auto relative z-10 flex flex-col justify-center">

              {images.length === 0 && !isGenerating ? (
                // Agentic Floating Empty State Card
                <div className="mx-auto w-full max-w-md bg-gradient-to-b from-[#0A1A2F]/80 to-[#0A0F1E]/90 rounded-3xl border border-cyan-500/20 p-8 shadow-[0_20px_50px_rgba(0,0,0,0.5),auto_auto_auto_rgba(0,255,255,0.05)_inset] backdrop-blur-xl flex flex-col items-center justify-center relative overflow-hidden group">

                  {/* Internal Glow Effects */}
                  <div className="absolute top-0 left-1/2 -translate-x-1/2 w-32 h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent opacity-50"></div>
                  <div className="absolute -top-12 -left-12 w-32 h-32 bg-blue-500/20 rounded-full blur-2xl"></div>
                  <div className="absolute -bottom-12 -right-12 w-32 h-32 bg-emerald-500/10 rounded-full blur-2xl"></div>

                  {/* Icon Row */}
                  <div className="flex gap-6 mb-8 relative z-10">
                    <div className="w-12 h-12 rounded-2xl bg-[#050810] border border-cyan-900/50 flex items-center justify-center text-cyan-500/70 group-hover:text-cyan-400 group-hover:border-cyan-500/40 transition-all duration-300 shadow-inner">
                      <Upload size={20} />
                    </div>
                    <div className="w-12 h-12 rounded-2xl bg-[#0A1A2F] border border-cyan-400/40 flex items-center justify-center text-cyan-300 shadow-[0_0_15px_rgba(0,255,255,0.15)] group-hover:shadow-[0_0_20px_rgba(0,255,255,0.25)] transition-all duration-300 scale-110 z-10">
                      <Search size={24} />
                    </div>
                    <div className="w-12 h-12 rounded-2xl bg-[#050810] border border-cyan-900/50 flex items-center justify-center text-cyan-500/70 group-hover:text-cyan-400 group-hover:border-cyan-500/40 transition-all duration-300 shadow-inner">
                      <LinkIcon size={20} />
                    </div>
                  </div>

                  <h3 className="text-lg font-medium text-cyan-50 mb-2 relative z-10">Awaiting Commands</h3>
                  <p className="text-sm text-cyan-600/60 text-center relative z-10 font-light">
                    No storyboard frames generated yet. Provide a prompt to begin visual synthesis.
                  </p>
                </div>
              ) : (
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 h-full content-start">
                  {images.map((img) => (
                    <div key={img.id} className="group relative rounded-2xl overflow-hidden border border-[#0A1A2F] bg-[#050810] shadow-[0_10px_30px_rgba(0,0,0,0.5)] transition-all hover:border-cyan-800/60 hover:shadow-glow-subtle flex flex-col h-full">
                      <div className="aspect-video bg-[#020409] relative shrink-0">
                        <img src={img.url} alt="Generated frame" className="w-full h-full object-cover opacity-90 group-hover:opacity-100 transition-opacity" />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end p-5">
                          <p className="text-xs text-cyan-50 font-light leading-relaxed">{img.prompt}</p>
                        </div>
                      </div>

                      {img.rag_context && (
                        <div className="p-4 bg-[#0A1A2F]/30 border-t border-[#0A1A2F] flex-1">
                          <h4 className="text-[10px] text-emerald-500/80 uppercase tracking-widest font-semibold mb-1 flex items-center gap-1.5">
                            <Search size={10} /> Grounding Context (RAG)
                          </h4>
                          <p className="text-xs text-cyan-200/60 leading-relaxed font-light line-clamp-3">
                            {Array.isArray(img.rag_context)
                              ? img.rag_context.map((ctx, i) => `${ctx.source !== 'Unknown' ? `[${ctx.source}] ` : ''}${ctx.text}`).join(' • ')
                              : img.rag_context}
                          </p>
                        </div>
                      )}

                      <div className="p-4 bg-[#0A0F1E] border-t border-[#0A1A2F] flex justify-between items-center shrink-0">
                        <span className="text-xs font-semibold uppercase tracking-widest text-cyan-600/70">Frame {images.indexOf(img) + 1}</span>
                        <button
                          onClick={() => handleExportHighRes(img.id)}
                          disabled={img.isUpscaling}
                          className={`text-[11px] uppercase tracking-wider font-semibold flex items-center gap-2 transition-colors ${img.isUpscaling ? 'text-emerald-500/50 cursor-not-allowed' : 'text-emerald-400 hover:text-emerald-300'}`}
                        >
                          {img.isUpscaling ? <div className="w-3 h-3 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" /> : null}
                          {img.isUpscaling ? 'UPSCALING...' : 'EXPORT HIGH-RES'}
                        </button>
                      </div>
                    </div>
                  ))}

                  {isGenerating && (
                    <div className="rounded-2xl overflow-hidden border border-cyan-500/30 bg-[#0A0F1E] shadow-[0_0_20px_rgba(0,255,255,0.05)] relative aspect-video flex flex-col items-center justify-center group">
                      <div className="absolute inset-0 opacity-20" style={{ backgroundImage: 'linear-gradient(0deg, transparent 24%, rgba(0, 255, 255, .3) 25%, rgba(0, 255, 255, .3) 26%, transparent 27%, transparent 74%, rgba(0, 255, 255, .3) 75%, rgba(0, 255, 255, .3) 76%, transparent 77%, transparent), linear-gradient(90deg, transparent 24%, rgba(0, 255, 255, .3) 25%, rgba(0, 255, 255, .3) 26%, transparent 27%, transparent 74%, rgba(0, 255, 255, .3) 75%, rgba(0, 255, 255, .3) 76%, transparent 77%, transparent)', backgroundSize: '50px 50px' }}></div>

                      <div className="absolute inset-0">
                        {/* Scanner effect (Subtle Agentic) */}
                        <div className="w-full h-1 bg-cyan-400/80 shadow-[0_0_25px_rgba(0,255,255,0.8)] absolute top-0 animate-[scan_3s_ease-in-out_infinite]" />
                      </div>
                      <Sparkles size={28} className="text-cyan-400/60 mb-4 animate-pulse relative z-10" />
                      <p className="text-xs font-semibold uppercase tracking-widest text-cyan-300 animate-pulse relative z-10">Synthesizing Visuals</p>
                      <p className="text-[10px] text-emerald-500/60 mt-2 uppercase tracking-widest relative z-10">Neural Rendering Active</p>
                    </div>
                  )}
                </div>
              )}

            </div>

          </div>

        </div>
      </div>

      <style dangerouslySetInnerHTML={{
        __html: `
        @keyframes scan {
          0% { top: 0; opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { top: 100%; opacity: 0; }
        }
      `}} />
    </div>
  );
}

export default App;
