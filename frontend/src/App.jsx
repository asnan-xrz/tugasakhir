import React, { useState, useEffect, useRef } from 'react';
import { Bot, FileText, Image as ImageIcon, Send, Upload, Settings, PanelLeftClose, PanelLeftOpen, Search, Link as LinkIcon, Sparkles, Terminal, Wand2, BookOpen, Camera, Loader, Sun, Moon } from 'lucide-react';
import axios from 'axios';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
const API_URL = 'http://localhost:8000';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [theme, setTheme] = useState('dark');
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

  // Full Auto Storyboard States
  const [inputMode, setInputMode] = useState('manual'); // 'manual' | 'auto'
  const [concept, setConcept] = useState('');
  const [logStream, setLogStream] = useState([]);
  const [fullAutoProgress, setFullAutoProgress] = useState(null);
  const [isExportingPdf, setIsExportingPdf] = useState(false);

  // Resizeable Left Panel State
  const [leftPanelWidth, setLeftPanelWidth] = useState(380);
  const [isDraggingPanel, setIsDraggingPanel] = useState(false);

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isDraggingPanel) return;
      let newWidth = e.clientX;
      if (newWidth < 300) newWidth = 300;
      if (newWidth > 800) newWidth = 800; // max width
      setLeftPanelWidth(newWidth);
    };
    const handleMouseUp = () => setIsDraggingPanel(false);

    if (isDraggingPanel) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDraggingPanel]);

  // Auto-scroll ref for terminal
  const logEndRef = useRef(null);
  
  // Scroll to bottom whenever logStream updates
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logStream]);

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

  const handleGenerateFullAuto = async () => {
    if (!concept.trim()) return;

    setIsGenerating(true);
    setLogStream(['--- INIT: Full Auto Agentic Pipeline ---']);
    setFullAutoProgress({ current: 0, total: 0 });
    setImages([]); // Clear previous!
    
    try {
      const startRes = await axios.post(`${API_URL}/api/generate-full`, { 
        concept: concept.trim(),
        use_rag: useRag
      });
      const taskId = startRes.data.task_id;

      const pollInterval = setInterval(async () => {
        try {
          const statusRes = await axios.get(`${API_URL}/api/task/${taskId}`);

          if (statusRes.data.log_stream) {
            setLogStream(statusRes.data.log_stream);
          }
          if (statusRes.data.total_frames) {
              setFullAutoProgress(prev => ({ ...prev, total: statusRes.data.total_frames, current: statusRes.data.current_frame || 0 }));
          }

          if (statusRes.data.status === 'processing_frame' || statusRes.data.status === 'generating_script') {
            setOrchestrationStatus(statusRes.data.log_stream[statusRes.data.log_stream.length - 1] || 'Memproses Automasi...');
            
            // PRE-FLIGHT RENDER: React captures the scenes and mounts them to the DOM before generation concludes
            if (statusRes.data.result_scenes) {
                const mappedUrls = statusRes.data.result_scenes.map(s => ({
                    id: s.id || (Math.random().toString()),
                    prompt: s.prompt || s.enhanced_prompt || "",
                    original_prompt: s.original_prompt,
                    script_dialogue: s.script_dialogue,
                    visual_description: s.visual_description,
                    scene_no: s.scene_no,
                    shot_letter: s.shot_letter,
                    durasi: s.durasi,
                    transisi: s.transisi,
                    audio: s.audio,
                    keterangan: s.keterangan,
                    url: s.image_url ? (s.image_url.startsWith('http') ? s.image_url : `${API_URL}${s.image_url}`) : null,
                    isUpscaling: false,
                    isGeneratingStatus: s.is_generating !== undefined ? s.is_generating : false
                }));
                setImages(mappedUrls);
            }
          } else if (statusRes.data.status === 'completed') {
            clearInterval(pollInterval);
            setIsGenerating(false);
            setOrchestrationStatus('');
            // Final sync
            if (statusRes.data.result_scenes) {
                const mappedUrls = statusRes.data.result_scenes.map(s => ({
                    id: s.id, prompt: s.enhanced_prompt, original_prompt: s.original_prompt,
                    script_dialogue: s.script_dialogue, visual_description: s.visual_description,
                    scene_no: s.scene_no, 
                    shot_letter: s.shot_letter, durasi: s.durasi, transisi: s.transisi, audio: s.audio, keterangan: s.keterangan,
                    url: s.image_url ? (s.image_url.startsWith('http') ? s.image_url : `${API_URL}${s.image_url}`) : null,
                    rag_context: s.rag_context, mode_ablasi: s.mode_ablasi, isUpscaling: false, isGeneratingStatus: false
                }));
                setImages(mappedUrls);
            }
          } else if (statusRes.data.status === 'failed') {
            clearInterval(pollInterval);
            setIsGenerating(false);
            setOrchestrationStatus('');
            alert("Gagal melakukan full automation: " + (statusRes.data.result?.error || "Unknown"));
          }
        } catch (err) {
          console.error("Poll err", err);
          clearInterval(pollInterval);
          setIsGenerating(false);
        }
      }, 3000);
    } catch (e) {
      console.error("Full Auto error:", e);
      setIsGenerating(false);
      alert("Gagal menghubungi server");
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

  const getBase64ImageFromUrl = async (imageUrl) => {
    try {
      const res = await fetch(imageUrl, { mode: 'cors' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
    } catch (e) {
      console.warn(`PDF image fetch failed for ${imageUrl}:`, e);
      return null; // Gracefully skip image instead of crashing PDF export
    }
  };

  const handleExportToPdf = async () => {
      if (images.length === 0) return;
      try {
          setIsExportingPdf(true);
          const doc = new jsPDF('l', 'pt', 'a4');
          
          // Add Branding Header
          doc.setFontSize(18);
          doc.setTextColor(30, 60, 114); // Cyan/Blue tone matching ITS
          doc.text(`ITS TV STORYBOARD${concept ? (' - ' + concept.substring(0, 50).trim()) : ''}`, 40, 40);
          
          doc.setFontSize(10);
          doc.setTextColor(100, 100, 100);
          doc.text(`Tanggal Produksi: ${new Date().toLocaleDateString()}`, 40, 60);
          doc.text("Generated by ITS TV Storyboard AI (Agentic VRAM-Safe Build)", 40, 75);

          const tableBody = [];
          for (let i = 0; i < images.length; i++) {
              const imgState = images[i];
              let base64Img = null;
              
              if (imgState.url) {
                  const absoluteUrl = imgState.url.startsWith('http') ? imgState.url : `${API_URL}${imgState.url}`;
                  try {
                      base64Img = await getBase64ImageFromUrl(absoluteUrl);
                  } catch (e) {
                      console.error("PDF Fetch warn (CORS/404):", e);
                  }
              }
              
              const cleanedPrompt = imgState.prompt?.replace(/masterpiece/gi, '')
                                                  .replace(/high resolution/gi, '')
                                                  .replace(/highly detailed/gi, '')
                                                  .replace(/8k/gi, '')
                                                  .replace(/cinematic lighting/gi, '')
                                                  .replace(/,/g, ' ')
                                                  .replace(/\s+/g, ' ').trim() || '';
                                                  
              const deskripsiAdegan = imgState.original_prompt ? imgState.original_prompt : cleanedPrompt;
              const dialog = imgState.script_dialogue ? `VO: "${imgState.script_dialogue}"` : "-";
              const sceneNum = imgState.scene_no || (i + 1);
              const shotLetter = imgState.shot_letter || String.fromCharCode(65 + (i % 26));
              const duration = imgState.durasi || '3s';
              const transition = imgState.transisi || 'cut to cut';
              const audio = imgState.audio || (imgState.script_dialogue ? 'Voice dialogue, ambient background' : 'BGM Cinematic, ambient');
              const keterangan = imgState.keterangan || (imgState.original_prompt?.match(/di\s([^,]+)/i) ? imgState.original_prompt.match(/di\s([^,]+)/i)[1] : "Exterior / Interior");

              tableBody.push([
                  sceneNum, 
                  shotLetter,
                  deskripsiAdegan,
                  dialog,
                  "", // Placeholder untuk image render di didDrawCell
                  imgState.visual_description || '-',
                  duration,
                  transition,
                  audio,
                  keterangan,
                  base64Img // Index 10 rahasia untuk extract
              ]);
          }

          autoTable(doc, {
              startY: 90,
              head: [['SCENE', 'SHOT', 'DESKRIPSI ADEGAN', 'SCRIPT', 'VISUALISASI', 'DESKRIPSI VISUAL', 'DURASI', 'TRANSISI', 'AUDIO', 'KETERANGAN']],
              body: tableBody,
              styles: {
                  valign: 'middle',
                  cellPadding: 4,
                  fontSize: 7,
                  lineColor: [40, 40, 40],
                  lineWidth: 0.5,
                  overflow: 'linebreak'
              },
              headStyles: {
                  fillColor: [10, 26, 47], // ITS TV Dark Nav
                  textColor: 255,
                  fontStyle: 'bold',
                  halign: 'center',
                  fontSize: 7
              },
              columnStyles: {
                  0: { cellWidth: 30, halign: 'center', fontStyle: 'bold' },
                  1: { cellWidth: 30, halign: 'center', fontStyle: 'bold' },
                  2: { cellWidth: 100 },
                  3: { cellWidth: 100, fontStyle: 'italic' },
                  4: { cellWidth: 120 }, // Visual space
                  5: { cellWidth: 100 },
                  6: { cellWidth: 40, halign: 'center' },
                  7: { cellWidth: 50, halign: 'center' },
                  8: { cellWidth: 90 },
                  9: { cellWidth: 'auto' }
              },
              didDrawCell: function (data) {
                  // Jika ini sel visual dan dibagian body (Kolom ke-4 adalah indeks 4)
                  if (data.section === 'body' && data.column.index === 4) {
                      const base64Img = data.row.raw[10]; // Extract dari rahasia array 
                      if (base64Img) {
                          const dim = 100; // Smaller dim to fit 10 columns
                          const x = data.cell.x + 10;
                          // Center vertically
                          const y = data.cell.y + (data.cell.height - dim) / 2;
                          try {
                              doc.addImage(base64Img, 'PNG', x, y > data.cell.y ? y : data.cell.y + 5, dim, dim); 
                          } catch (e) { console.error("Error drawing image in PDF", e); }
                      }
                  }
              },
              margin: { top: 70, left: 20, right: 20, bottom: 40 },
              rowPageBreak: 'avoid',
              bodyStyles: { minCellHeight: 110 }, // Spacer tinggi minimum sel untuk gambar
          });

          doc.save(`ITS_TV_Storyboard_${new Date().getTime()}.pdf`);
      } catch (error) {
          console.error("PDF Export failed:", error);
          alert("Gagal mengekspor PDF.");
      } finally {
          setIsExportingPdf(false);
      }
  };
  const renderWithItalics = (text) => {
      if (!text) return text;
      // Split by *asterisks*
      const parts = text.split(/(\*[^*]+\*)/g);
      return parts.map((part, i) => 
        part.startsWith('*') && part.endsWith('*') 
          ? <i key={i} className="italic text-cyan-300 font-serif tracking-wide">{part.slice(1, -1)}</i> 
          : part
      );
  };

  return (
    <div className={`flex h-screen bg-[#050810] text-slate-200 overflow-hidden font-sans selection:bg-cyan-900 selection:text-cyan-100 transition-all duration-700 ease-in-out ${theme === 'light' ? 'theme-light' : ''}`}>

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
        <div className="p-4 border-t border-cyan-900/20 bg-[#050810]/80 z-10 transition-colors">
          <div className="flex justify-between items-center text-sm">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 text-slate-400 hover:text-cyan-300 cursor-pointer transition-colors group">
                <Settings size={16} className="group-hover:rotate-45 transition-transform duration-300" />
                <span className="font-medium text-xs hidden lg:inline">Settings</span>
              </div>
              <button 
                onClick={() => setTheme(prev => prev === 'dark' ? 'light' : 'dark')}
                className="flex items-center gap-1.5 text-cyan-400 hover:text-cyan-200 border border-cyan-900/40 bg-cyan-900/10 px-2 py-1 rounded transition-colors group"
                title="Toggle Theme"
              >
                {theme === 'dark' ? <Sun size={12} className="group-hover:rotate-45 transition-transform duration-500" /> : <Moon size={12} className="group-hover:-rotate-12 transition-transform duration-500" />}
                <span className="text-[9px] font-bold tracking-widest uppercase">{theme}</span>
              </button>
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
          <div className="min-w-[300px] flex flex-col border-r border-[#0A1A2F] bg-[#050810] relative print:hidden shrink-0" style={{ width: leftPanelWidth }}>
            
            {/* Drag Handle */}
            <div 
              className="absolute top-0 right-[-4px] w-2 h-full cursor-col-resize z-50 hover:bg-cyan-500/20 active:bg-cyan-500/50 transition-colors"
              onMouseDown={() => setIsDraggingPanel(true)}
            />
            {/* Background Ambient Glow */}
            <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-blue-900/10 rounded-full blur-3xl pointer-events-none" />

            {/* Mode Swapper */}
            <div className="flex border-b border-[#0A1A2F] shrink-0 z-20 bg-[#050810]">
              <button 
                onClick={() => setInputMode('manual')}
                className={`flex-1 py-3 text-[10px] font-bold uppercase tracking-widest transition-colors ${inputMode === 'manual' ? 'bg-cyan-900/20 text-cyan-400 border-b-2 border-cyan-400' : 'text-slate-500 hover:text-slate-300'}`}
              >
                Manual Scene
              </button>
              <button 
                onClick={() => setInputMode('auto')}
                className={`flex-1 py-3 text-[10px] font-bold uppercase tracking-widest transition-colors flex items-center justify-center gap-2 ${inputMode === 'auto' ? 'bg-emerald-900/20 text-emerald-400 border-b-2 border-emerald-400' : 'text-slate-500 hover:text-slate-300'}`}
              >
                <Sparkles size={12} /> Concept Auto Mode
              </button>
            </div>

            {inputMode === 'manual' ? (
              <>
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
            </>
            ) : (
            <>
              {/* Auto Input Area */}
              <div className="flex-1 overflow-y-auto p-6 space-y-6 z-10 w-full relative custom-scrollbar">
                <div className="space-y-3">
                  <label className="text-xs font-bold tracking-widest text-[#0A1A2F] uppercase drop-shadow-sm flex items-center gap-2">
                    <Wand2 size={14} className="text-emerald-600" /> Tulis Ide Konsep Kreatif
                  </label>
                  <p className="text-[10px] text-slate-500 mb-2 leading-relaxed font-mono">
                    LLM Llama 3 akan mengekstrak ide ini menjadi 5-8 adegan berurutan beserta Visual Grounding secara otonom...
                  </p>
                  <textarea
                    value={concept}
                    onChange={(e) => setConcept(e.target.value)}
                    placeholder="Contoh: Iklan layanan masyarakat sedih tentang Hari Ibu di kampus ITS..."
                    className="w-full h-[150px] bg-[#0A0F1E] border border-emerald-900/40 rounded-xl p-3 text-emerald-50 text-sm focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/20 transition-all placeholder:text-slate-700/80 placeholder:italic resize-none"
                  />
                </div>
              </div>

              {/* Generate Auto Action */}
              <div className="p-6 bg-[#050810] z-10 shrink-0 border-t border-cyan-900/30 print:hidden">
                <div className="flex justify-between items-center mb-3 px-2">
                  <span className="text-xs text-cyan-600 font-semibold uppercase tracking-widest">Generation Settings</span>
                  <label className="flex items-center gap-3 cursor-pointer group">
                    <span className={`text-[10px] uppercase font-bold tracking-wider transition-colors ${useRag ? 'text-cyan-400/80 group-hover:text-cyan-300' : 'text-slate-500 group-hover:text-slate-400'}`}>
                      Gunakan RAG
                    </span>
                    <div className="relative">
                      <input type="checkbox" className="sr-only" checked={useRag} onChange={(e) => setUseRag(e.target.checked)} disabled={isGenerating} />
                      <div className={`block w-8 h-4 rounded-full transition-colors duration-300 ${useRag ? 'bg-emerald-500/80' : 'bg-[#0A1A2F] border border-slate-700'}`}></div>
                      <div className={`dot absolute left-1 top-1 bg-white w-2 h-2 rounded-full transition-transform duration-300 ${useRag ? 'transform translate-x-4' : ''}`}></div>
                    </div>
                  </label>
                </div>
                
                <button
                    onClick={handleGenerateFullAuto}
                    disabled={isGenerating || !concept.trim()}
                    className={`w-full py-3.5 rounded-lg font-extrabold uppercase tracking-[0.1em] text-xs transition-all relative overflow-hidden group
                      ${isGenerating || !concept.trim() ? 'bg-[#0A1A2F] border border-[#0A1A2F]/50 text-slate-500 cursor-not-allowed' : 'bg-emerald-600 text-white hover:bg-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.3)]'}`}
                  >
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700" />
                    <span className="relative flex items-center justify-center gap-2">
                       <Sparkles size={14}/> {isGenerating ? 'Auto Orchestrating...' : 'Generate Full Sequence'} 
                    </span>
                </button>
              </div>
            </>
            )}
          </div>

          {/* Canvas Area */}
          <div className="flex-1 bg-[#020409] flex flex-col relative overflow-hidden print:bg-white">
            {/* Background pattern */}
            <div className="absolute inset-0 opacity-[0.03] pointer-events-none print:hidden" style={{ backgroundImage: 'radial-gradient(circle at 2px 2px, #00FFFF 1px, transparent 0)', backgroundSize: '32px 32px' }}></div>

            {/* Canvas Header */}
            <div className="absolute top-0 left-0 right-0 p-4 flex justify-between items-center z-50 bg-gradient-to-b from-[#020409] to-transparent print:hidden">
              <h3 className="text-xs font-semibold uppercase tracking-widest text-cyan-600/50 flex items-center gap-2 drop-shadow-md">
                <ImageIcon size={14} /> Output Viewer
              </h3>
              {images.length > 0 && (
                <button
                  onClick={handleExportToPdf}
                  disabled={isExportingPdf || isGenerating}
                  className={`flex items-center gap-2 border px-3 py-1.5 rounded-lg text-xs font-semibold tracking-wide transition-colors shadow-lg
                    ${isExportingPdf || isGenerating ? 'border-emerald-900/50 text-emerald-700 bg-emerald-900/20 cursor-not-allowed opacity-70' : 'border-emerald-500/50 text-emerald-400 bg-emerald-900/40 hover:bg-emerald-500 hover:text-white'}`}
                >
                  {isExportingPdf ? <Loader size={12} className="animate-spin" /> : <FileText size={14} />}
                  {isExportingPdf ? 'Menyusun PDF...' : 'Export Professional PDF'}
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
                <div className="w-full bg-[#050810] border-t border-slate-800 overflow-x-auto shadow-2xl relative text-slate-200 print-container z-10 flex-1">
                  
                  {/* Live Stream Terminal Area */}
                  {(inputMode === 'auto' && logStream.length > 0) && (
                    <div className="w-full h-40 bg-[#020409] border-b-4 border-emerald-600/50 p-3 font-mono text-[10px] overflow-y-auto print:hidden shadow-[inset_0px_10px_20px_#00000080]">
                      <div className="text-emerald-400 mb-2 font-bold tracking-widest uppercase flex items-center gap-2">
                        <Terminal size={12} /> Director AI Orchestrator Link
                        {fullAutoProgress?.total > 0 && <span className="text-cyan-400 ml-auto">PROGRESS: {fullAutoProgress.current}/{fullAutoProgress.total}</span>}
                      </div>
                      <ul className="space-y-1">
                        {logStream.map((logLine, idx) => (
                           <li key={idx} className={String(logLine).includes("ERROR") ? "text-rose-500 font-bold" : String(logLine).includes("Flushed") ? "text-slate-500" : String(logLine).includes("--- INIT") ? "text-yellow-400 mt-2" : "text-emerald-500/80"}>
                              <span className="text-slate-600 select-none">{`[${String(idx).padStart(4, '0')}] `}</span>{String(logLine)}
                           </li>
                        ))}
                        <div ref={logEndRef} />
                      </ul>
                    </div>
                  )}

                  <table className="w-full text-center text-[11px] whitespace-normal tabular-nums border-collapse border-slate-800">
                    <thead className="bg-[#0A1A2F] font-extrabold text-cyan-50 border-b border-slate-800">
                      <tr>
                        <th className="border border-slate-700 p-2 w-[4%]">SCENE<br/><span className="text-[9px] font-normal italic text-cyan-600/60">*No. Scene*</span></th>
                        <th className="border border-slate-700 p-2 w-[4%]">SHOT<br/><span className="text-[9px] font-normal italic text-cyan-600/60">*No. Shot*</span></th>
                        <th className="border border-slate-700 p-2 w-[14%]">DESKRIPSI ADEGAN<br/><span className="text-[9px] font-normal italic text-cyan-600/60">*Deskripsi cerita/adegan, alur, mood*</span></th>
                        <th className="border border-slate-700 p-2 w-[10%]">SCRIPT<br/><span className="text-[9px] font-normal italic text-cyan-600/60">*Dialog atau voice over*</span></th>
                        <th className="border border-slate-700 p-2 w-[22%]">VISUALISASI DAN REFERENSI<br/><span className="text-[9px] font-normal italic text-cyan-600/60">*Gambaran visual dari adegan*</span></th>
                        <th className="border border-slate-700 p-2 w-[12%]">DESKRIPSI VISUAL<br/><span className="text-[9px] font-normal italic text-cyan-600/60">*Jenis framing, angle, movement*</span></th>
                        <th className="border border-slate-700 p-2 w-[6%]">DURASI<br/><span className="text-[9px] font-normal italic text-cyan-600/60">*detik*</span></th>
                        <th className="border border-slate-700 p-2 w-[8%]">TRANSISI<br/><span className="text-[9px] font-normal italic text-cyan-600/60">*Jenis transisi*</span></th>
                        <th className="border border-slate-700 p-2 w-[10%]">AUDIO<br/><span className="text-[9px] font-normal italic text-cyan-600/60">*SFX / BGM*</span></th>
                        <th className="border border-slate-700 p-2 w-[10%]">KETERANGAN<br/><span className="text-[9px] font-normal italic text-cyan-600/60">*Lokasi*</span></th>
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
                                                        
                        const shotLetter = img.shot_letter || String.fromCharCode(65 + (index % 26));
                        const duration = img.durasi || '3s';
                        const transition = img.transisi || 'cut to cut';
                        const audio = img.audio || (img.script_dialogue ? 'Voice dialogue, ambient background' : 'BGM Cinematic, ambient');
                        const keterangan = img.keterangan || (img.original_prompt?.match(/di\s([^,]+)/i) ? img.original_prompt.match(/di\s([^,]+)/i)[1] : "Exterior / Interior");

                        return (
                          <tr key={img.id} className="transition-colors hover:bg-slate-800/40">
                            <td className="px-2 py-4 align-middle border border-slate-700 font-bold text-sm text-cyan-100">{img.scene_no || index + 1}</td>
                            <td className="px-2 py-4 align-middle border border-slate-700 font-bold text-sm text-cyan-200">{shotLetter}</td>
                            <td className="px-3 py-4 align-top border border-slate-700 text-left font-medium text-slate-300">
                              {renderWithItalics(img.original_prompt ? img.original_prompt : cleanedPrompt)}
                            </td>
                            <td className="px-2 py-4 align-top border border-slate-700 font-medium">
                              {img.script_dialogue ? <span className="italic text-cyan-400">VO: "{renderWithItalics(img.script_dialogue)}"</span> : '-'}
                            </td>
                            <td className="px-3 py-4 align-top border border-slate-700">
                              <div className="relative rounded-sm overflow-hidden border border-slate-400 bg-black min-h-[120px] shadow-sm flex items-center justify-center">
                                {img.isGeneratingStatus ? (
                                    <div className="flex flex-col items-center justify-center p-4 bg-slate-900/50 w-full h-full">
                                        <div className="w-6 h-6 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin mb-2"></div>
                                        <span className="text-[9px] text-emerald-400 tracking-widest uppercase font-bold text-center">Mensintesis<br/>Visual...</span>
                                    </div>
                                ) : img.url ? (
                                    <>
                                        <img src={img.url} alt={`Scene ${index + 1}`} className="w-full h-auto object-cover max-h-[160px]" onError={(e) => { e.target.onerror = null; e.target.src = 'https://via.placeholder.com/512x288?text=Image+Load+Error'; }} />
                                        <button
                                          onClick={() => handleExportHighRes(img.id)}
                                          disabled={img.isUpscaling}
                                          className={`absolute bottom-1 right-1 px-1.5 py-0.5 bg-black/60 backdrop-blur-md rounded border border-emerald-500/30 text-[8px] uppercase font-bold transition-all print:hidden
                                            ${img.isUpscaling ? 'text-emerald-500/50 cursor-not-allowed' : 'text-emerald-400 hover:bg-emerald-900/80 hover:text-emerald-200'}`}
                                        >
                                          {img.isUpscaling ? 'UPSCALING...' : 'HI-RES'}
                                        </button>
                                    </>
                                ) : (
                                    <span className="text-xs text-slate-600 uppercase font-bold">No Image Rendered</span>
                                )}
                              </div>
                              {img.rag_context && !img.mode_ablasi && img.rag_context.length > 0 && (
                                <div className="mt-3 flex flex-col gap-1.5 print:hidden bg-slate-900/40 p-2 rounded border border-slate-700/50 shadow-inner">
                                  <span className="text-[9px] text-cyan-500/80 uppercase font-bold tracking-widest border-b border-slate-700/50 pb-1 mb-1">
                                    Vector Similarity (RAG)
                                  </span>
                                  {img.rag_context.map((ctx, cIdx) => {
                                      // ChromaDB L2 distance -> simple percentage score approx
                                      const simScore = Math.max(0, 100 - (ctx.distance * 50)).toFixed(1);
                                      return (
                                          <div key={cIdx} className="flex justify-between items-center text-[10px] bg-black/40 px-2 py-1 rounded border border-slate-800/60">
                                              <span className="text-slate-400 truncate w-2/3" title={ctx.source}>{ctx.source}</span>
                                              <span className={`font-mono font-semibold ${simScore > 75 ? 'text-emerald-400' : simScore > 50 ? 'text-amber-400' : 'text-rose-400'}`}>
                                                {simScore}%
                                              </span>
                                          </div>
                                      );
                                  })}
                                </div>
                              )}
                              {img.mode_ablasi && (
                                <div className="mt-2 text-[9px] text-rose-700 bg-rose-100 uppercase font-bold border border-rose-300 rounded px-1 py-0.5 inline-block tracking-widest text-center w-full">
                                  Ablation Mode (No RAG)
                                </div>
                              )}
                            </td>
                            <td className="px-2 py-4 align-middle border border-slate-700 font-medium text-slate-300">
                              {renderWithItalics(img.visual_description || 'N/A')}
                            </td>
                            <td className="px-2 py-4 align-middle border border-slate-700 text-cyan-300 font-mono">
                              {renderWithItalics(duration)}
                            </td>
                            <td className="px-2 py-4 align-middle border border-slate-700 uppercase text-slate-400">
                              {renderWithItalics(transition)}
                            </td>
                            <td className="px-2 py-4 align-top border border-slate-700 text-left text-slate-500">
                              {renderWithItalics(audio)}
                            </td>
                            <td className="px-2 py-4 align-middle border border-slate-700 font-medium text-slate-300">
                              {renderWithItalics(keterangan)}
                            </td>
                          </tr>
                        );
                      })}
                      {isGenerating && (
                        <tr>
                          <td colSpan="10" className="px-4 py-12 text-center bg-[#050810] border border-slate-700 print:hidden">
                            <div className="flex flex-col items-center justify-center space-y-3">
                               <Sparkles size={24} className="text-cyan-600 animate-pulse" />
                               <span className="text-xs font-bold uppercase tracking-widest text-cyan-400 animate-pulse">{orchestrationStatus || 'Synthesizing Visuals'}</span>
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
