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
  const [currentVisualDesc, setCurrentVisualDesc] = useState('');
  const [currentScriptDialogue, setCurrentScriptDialogue] = useState('');
  const [orchestrationStatus, setOrchestrationStatus] = useState('');
  const [useRag, setUseRag] = useState(true);

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
    setOrchestrationStatus('Proses inisialisasi...');
    const currentPrompt = prompt;
    const currentVis = currentVisualDesc;
    const currentScript = currentScriptDialogue;
    setPrompt('');
    setCurrentVisualDesc('');
    setCurrentScriptDialogue('');

    try {
      // 1. Send generation request
      const startRes = await axios.post(`${API_URL}/api/generate`, { 
        prompt: currentPrompt,
        visual_description: currentVis || null,
        script_dialogue: currentScript || null,
        use_rag: useRag
      });
      const taskId = startRes.data.task_id;

      // 2. Poll for status
      const pollInterval = setInterval(async () => {
        try {
          const statusRes = await axios.get(`${API_URL}/api/task/${taskId}`);

          if (statusRes.data.status === 'completed') {
            clearInterval(pollInterval);
            setImages(prev => [...prev, {
              id: taskId,
              url: statusRes.data.result.image_url.startsWith('http') 
                    ? statusRes.data.result.image_url 
                    : `${API_URL}${statusRes.data.result.image_url.startsWith('/') ? '' : '/'}${statusRes.data.result.image_url}`,
              prompt: statusRes.data.result.enhanced_prompt,
              original_prompt: currentPrompt,
              visual_description: statusRes.data.result.visual_description,
              script_dialogue: statusRes.data.result.script_dialogue,
              rag_context: statusRes.data.result.rag_context,
              mode_ablasi: statusRes.data.result.mode_ablasi
            }]);
            setIsGenerating(false);
            setOrchestrationStatus('');
          } else if (statusRes.data.status === 'failed') {
            clearInterval(pollInterval);
            console.error("Generation failed:", statusRes.data.result.error);
            setIsGenerating(false);
            setOrchestrationStatus('');
            alert("Generation failed. Check console.");
          } else if (statusRes.data.status === 'rag_search') {
            setOrchestrationStatus('Mencari Referensi Latar (RAG)...');
          } else if (statusRes.data.status === 'skip_rag') {
            setOrchestrationStatus('Mode Ablasi (Tanpa Grounding)...');
          } else if (statusRes.data.status === 'diffusion') {
            setOrchestrationStatus(useRag ? 'Mensintesis Visual (Diffusion)...' : 'Mode Ablasi (Mensintesis Visual)...');
          } else {
            setOrchestrationStatus('Proses inisialisasi...');
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
            const absoluteUrl = newUrl.startsWith('http') 
              ? newUrl 
              : `${API_URL}${newUrl.startsWith('/') ? '' : '/'}${newUrl}`;

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
        className={`${sidebarOpen ? 'w-64' : 'w-0'} transition-all duration-300 ease-in-out border-r border-[#0A1A2F]/50 bg-gradient-to-b from-[#0A1A2F]/80 to-[#050810] flex flex-col relative overflow-hidden print:hidden`}
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
        <header className="h-14 border-b border-[#0A1A2F]/50 flex items-center justify-between px-4 bg-[#050810]/50 backdrop-blur-md z-20 shrink-0 print:hidden">
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
          <div className="w-[28%] min-w-[320px] max-w-[400px] flex flex-col border-r border-[#0A1A2F] bg-[#050810] relative print:hidden">

            {/* Background Ambient Glow */}
            <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-blue-900/10 rounded-full blur-3xl pointer-events-none" />

            {/* Messages/Analysis Output */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6 z-10 w-full relative">
              {isUploading ? (
                <div className="flex flex-col items-center justify-center h-full w-full mt-20">
                  <div className="w-10 h-10 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin mb-4" />
                  <p className="text-cyan-400 text-sm font-medium animate-pulse tracking-wide uppercase">Menganalisis Naskah...</p>
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
                        onClick={() => {
                          setPrompt(`Scene ${scene.scene_no} at ${scene.location}: ${scene.description}. Shot: ${scene.shot_type}`);
                          setCurrentVisualDesc(scene.visual_description || '');
                          setCurrentScriptDialogue(scene.script_dialogue || '');
                        }}
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
            <div className="p-6 bg-[#050810] z-10 shrink-0 border-t border-cyan-900/30 print:hidden">
              <div className="flex justify-between items-center mb-3 px-2">
                <span className="text-xs text-cyan-600 font-semibold uppercase tracking-widest">Generation Settings</span>
                <label className="flex items-center gap-3 cursor-pointer group">
                  <span className={`text-[10px] uppercase font-bold tracking-wider transition-colors ${useRag ? 'text-cyan-400/80 group-hover:text-cyan-300' : 'text-slate-500 group-hover:text-slate-400'}`}>
                    Gunakan Grounding RAG (Aset ITS)
                  </span>
                  <div className="relative">
                    <input 
                      type="checkbox" 
                      className="sr-only" 
                      checked={useRag} 
                      onChange={(e) => setUseRag(e.target.checked)} 
                      disabled={isGenerating}
                    />
                    <div className={`block w-8 h-4 rounded-full transition-colors duration-300 ${useRag ? 'bg-emerald-500/80' : 'bg-[#0A1A2F] border border-slate-700'}`}></div>
                    <div className={`dot absolute left-1 top-1 bg-white w-2 h-2 rounded-full transition-transform duration-300 ${useRag ? 'transform translate-x-4' : ''}`}></div>
                  </div>
                </label>
              </div>

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
          <div className="flex-1 bg-[#020409] flex flex-col relative overflow-hidden print:bg-white">
            {/* Background pattern */}
            <div className="absolute inset-0 opacity-[0.03] pointer-events-none print:hidden" style={{ backgroundImage: 'radial-gradient(circle at 2px 2px, #00FFFF 1px, transparent 0)', backgroundSize: '32px 32px' }}></div>

            {/* Canvas Header */}
            <div className="absolute top-0 left-0 right-0 p-4 flex justify-between items-center z-10 bg-gradient-to-b from-[#020409] to-transparent pointer-events-none print:hidden">
              <h3 className="text-xs font-semibold uppercase tracking-widest text-cyan-600/50 flex items-center gap-2 drop-shadow-md">
                <ImageIcon size={14} /> Output Viewer
              </h3>
              {images.length > 0 && (
                <button
                  onClick={() => window.print()}
                  className="pointer-events-auto flex items-center gap-2 bg-emerald-900/40 hover:bg-emerald-800/60 border border-emerald-500/50 text-emerald-400 px-3 py-1.5 rounded-lg text-xs font-semibold tracking-wide transition-colors"
                >
                  Export to PDF
                </button>
              )}
            </div>

            {/* Visualizer Area */}
            <div className="flex-1 p-8 overflow-y-auto relative z-10 flex flex-col pt-16 print:p-0">

              {images.length === 0 && !isGenerating ? (
                // Agentic Floating Empty State Card
                <div className="mx-auto w-full max-w-md bg-gradient-to-b from-[#0A1A2F]/80 to-[#0A0F1E]/90 rounded-3xl border border-cyan-500/20 p-8 shadow-[0_20px_50px_rgba(0,0,0,0.5),auto_auto_auto_rgba(0,255,255,0.05)_inset] backdrop-blur-xl flex flex-col items-center justify-center relative overflow-hidden group print:hidden">

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
                <div className="w-full bg-white border border-slate-800 overflow-x-auto shadow-2xl relative text-black">
                  <table className="w-full text-center text-[11px] whitespace-normal tabular-nums border-collapse border border-slate-800">
                    <thead className="bg-[#E2E2E2] font-extrabold text-black border-b border-slate-800">
                      <tr>
                        <th className="border border-slate-400 p-2 w-[4%]">SCENE<br/><span className="text-[9px] font-normal italic">*No. Scene*</span></th>
                        <th className="border border-slate-400 p-2 w-[4%]">SHOT<br/><span className="text-[9px] font-normal italic">*No. Shot*</span></th>
                        <th className="border border-slate-400 p-2 w-[14%]">DESKRIPSI ADEGAN<br/><span className="text-[9px] font-normal italic">*Deskripsi cerita/adegan, alur, mood*</span></th>
                        <th className="border border-slate-400 p-2 w-[10%]">SCRIPT<br/><span className="text-[9px] font-normal italic">*Dialog atau voice over*</span></th>
                        <th className="border border-slate-400 p-2 w-[22%]">VISUALISASI DAN REFERENSI<br/><span className="text-[9px] font-normal italic">*Gambaran visual dari adegan*</span></th>
                        <th className="border border-slate-400 p-2 w-[12%]">DESKRIPSI VISUAL<br/><span className="text-[9px] font-normal italic">*Jenis framing, angle, movement*</span></th>
                        <th className="border border-slate-400 p-2 w-[6%]">DURASI<br/><span className="text-[9px] font-normal italic">*detik*</span></th>
                        <th className="border border-slate-400 p-2 w-[8%]">TRANSISI<br/><span className="text-[9px] font-normal italic">*Jenis transisi*</span></th>
                        <th className="border border-slate-400 p-2 w-[10%]">AUDIO<br/><span className="text-[9px] font-normal italic">*SFX / BGM*</span></th>
                        <th className="border border-slate-400 p-2 w-[10%]">KETERANGAN<br/><span className="text-[9px] font-normal italic">*Lokasi*</span></th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                      {images.map((img, index) => {
                        // Extracting some logical chunks if possible, else falling back gracefully
                        const cleanedPrompt = img.prompt.replace(/masterpiece/gi, '')
                                                        .replace(/high resolution/gi, '')
                                                        .replace(/highly detailed/gi, '')
                                                        .replace(/8k/gi, '')
                                                        .replace(/cinematic lighting/gi, '')
                                                        .replace(/,/g, ' ')
                                                        .replace(/\s+/g, ' ').trim();
                                                        
                        // Convert integer index to alphabetical shot identifier (A, B, C...)
                        const shotLetter = String.fromCharCode(65 + (index % 26));

                        // Attempt to extract duration/transition/script details from prompt (Dummy placeholders for perfection)
                        const durationMatch = img.original_prompt?.match(/(\d+)s/i);
                        const duration = durationMatch ? durationMatch[0] : '3s';
                        const transitionMatch = img.original_prompt?.match(/cut to cut|fade|crossfade/i);
                        const transition = transitionMatch ? transitionMatch[0].toLowerCase() : 'cut to cut';

                        return (
                          <tr key={img.id} className="transition-colors hover:bg-slate-50">
                            <td className="px-2 py-4 align-middle border border-slate-400 font-bold text-sm">1</td>
                            <td className="px-2 py-4 align-middle border border-slate-400 font-bold text-sm">{shotLetter}</td>
                            <td className="px-3 py-4 align-top border border-slate-400 text-left font-medium">
                              {img.original_prompt ? img.original_prompt : cleanedPrompt}
                            </td>
                            <td className="px-2 py-4 align-top border border-slate-400 italic text-slate-800 font-medium">
                              {img.script_dialogue ? `VO: "${img.script_dialogue}"` : '-'}
                            </td>
                            <td className="px-3 py-4 align-top border border-slate-400">
                              <div className="relative rounded-sm overflow-hidden border border-slate-400 bg-black min-h-[120px] shadow-sm flex items-center justify-center">
                                <img src={img.url} alt={`Scene ${index + 1}`} className="w-full h-auto object-cover max-h-[160px]" onError={(e) => { e.target.onerror = null; e.target.src = 'https://via.placeholder.com/512x288?text=Image+Load+Error'; }} />
                                <button
                                  onClick={() => handleExportHighRes(img.id)}
                                  disabled={img.isUpscaling}
                                  className={`absolute bottom-1 right-1 px-1.5 py-0.5 bg-black/60 backdrop-blur-md rounded border border-emerald-500/30 text-[8px] uppercase font-bold transition-all print:hidden
                                    ${img.isUpscaling ? 'text-emerald-500/50 cursor-not-allowed' : 'text-emerald-400 hover:bg-emerald-900/80 hover:text-emerald-200'}`}
                                >
                                  {img.isUpscaling ? 'UPSCALING...' : 'HI-RES'}
                                </button>
                              </div>
                              {img.rag_context && !img.mode_ablasi && (
                                <div className="mt-2 text-[8px] text-left text-slate-700 leading-tight">
                                  <strong>RAG refs:</strong> {Array.isArray(img.rag_context) ? img.rag_context.map(ctx => ctx.source).join(', ') : 'Context Used'}
                                </div>
                              )}
                              {img.mode_ablasi && (
                                <div className="mt-2 text-[9px] text-rose-700 bg-rose-100 uppercase font-bold border border-rose-300 rounded px-1 py-0.5 inline-block tracking-widest text-center w-full">
                                  Ablation Mode (No RAG)
                                </div>
                              )}
                            </td>
                            <td className="px-2 py-4 align-middle border border-slate-400 font-medium">
                              {img.visual_description || 'N/A'}
                            </td>
                            <td className="px-2 py-4 align-middle border border-slate-400">
                              {duration}
                            </td>
                            <td className="px-2 py-4 align-middle border border-slate-400">
                              {transition}
                            </td>
                            <td className="px-2 py-4 align-top border border-slate-400 text-left">
                              -
                            </td>
                            <td className="px-2 py-4 align-middle border border-slate-400 font-medium">
                              Lokasi
                            </td>
                          </tr>
                        );
                      })}
                      {isGenerating && (
                        <tr>
                          <td colSpan="10" className="px-4 py-12 text-center bg-slate-50 border border-slate-400 print:hidden">
                            <div className="flex flex-col items-center justify-center space-y-3">
                               <Sparkles size={24} className="text-cyan-600 animate-pulse" />
                               <span className="text-xs font-bold uppercase tracking-widest text-slate-800 animate-pulse">{orchestrationStatus || 'Synthesizing Visuals'}</span>
                            </div>
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
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
