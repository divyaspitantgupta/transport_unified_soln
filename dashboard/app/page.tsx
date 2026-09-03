'use client';

import { useState } from 'react';
import Map, { Marker } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';

export default function SmartCityDashboard() {
  // Wapas 3D perspective le aaye (pitch: 60, zoom thoda paas)
  const [viewState, setViewState] = useState({
    longitude: 77.2090, 
    latitude: 28.6139,
    zoom: 14,
    pitch: 60, 
    bearing: -15
  });

  return (
    <main className="flex flex-col h-screen w-full bg-[#050914] text-slate-400 font-mono text-[10px] overflow-hidden">
      
      {/* ==================== TOP NAV BAR ==================== */}
      <header className="flex items-center justify-between h-14 border-b border-slate-800 px-4 bg-[#0a0f1a] z-20 shadow-md">
        <div className="flex items-center gap-3">
          <div className="w-6 h-6 bg-cyan-900/50 border border-cyan-500 rounded flex items-center justify-center text-cyan-400">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="square" strokeLinejoin="miter" strokeWidth="2" d="M4 4h16v16H4z"></path><path strokeLinecap="square" strokeLinejoin="miter" strokeWidth="2" d="M9 9h6v6H9z"></path></svg>
          </div>
          <div>
            <h1 className="text-xs font-bold text-slate-200 tracking-widest">ROADSENSE ICCC</h1>
            <p className="text-[9px] text-slate-500 uppercase tracking-widest">ANPR & Road Defect Geo Intelligence</p>
          </div>
        </div>
        
        <div className="flex items-center gap-6 text-[10px]">
          <div className="flex flex-col items-end"><span className="text-slate-500">EDGE UNITS</span><span className="text-slate-200">6 online</span></div>
          <div className="flex flex-col items-end"><span className="text-slate-500">DEFECTS MAPPED</span><span className="text-slate-200">94</span></div>
          <div className="flex flex-col items-end"><span className="text-slate-500">PLATES READ</span><span className="text-slate-200">14</span></div>
          <button className="flex items-center gap-2 border border-emerald-500/50 bg-emerald-900/20 text-emerald-400 px-3 py-1 rounded hover:bg-emerald-900/40 transition-colors">
            <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></span>
            LIVE STREAM
          </button>
        </div>
      </header>

      {/* ==================== MAIN WORKSPACE ==================== */}
      <div className="flex flex-1 overflow-hidden">
        
        {/* ======== LEFT SIDEBAR ======== */}
        <div className="w-64 border-r border-slate-800 bg-[#070b14]/90 backdrop-blur-md flex flex-col overflow-y-auto z-20">
          <div className="p-4 border-b border-slate-800">
            <h2 className="text-xs text-slate-300 mb-3 tracking-widest">LIVE INFERENCE PIPELINE</h2>
            <div className="bg-[#0a101d] border border-slate-800 p-3 rounded mb-3">
              <p className="text-slate-500 mb-3">Attach a camera or upload road footage to run real detection + OCR.</p>
              <div className="flex gap-2">
                <button className="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-300 py-1.5 rounded flex items-center justify-center gap-2 border border-slate-700 transition-colors">
                  CAMERA
                </button>
                <button className="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-300 py-1.5 rounded flex items-center justify-center gap-2 border border-slate-700 transition-colors">
                  UPLOAD
                </button>
              </div>
            </div>
            <div className="space-y-1 text-slate-500">
              <p>▶ Pipeline idle</p>
              <p>⌖ 28.6139, 77.2090 - default</p>
              <p>frames analysed: 0</p>
            </div>
          </div>

          <div className="p-4 border-b border-slate-800">
            <h2 className="text-xs text-slate-300 mb-3 tracking-widest">MAP LAYERS</h2>
            <div className="space-y-3">
              {['Defect markers', 'Severity heatmap', 'Fleet positions'].map(layer => (
                <div key={layer} className="flex justify-between items-center">
                  <span>{layer}</span>
                  <div className="w-8 h-4 bg-cyan-900 border border-cyan-500 rounded-full flex items-center p-0.5 justify-end">
                    <div className="w-3 h-3 bg-cyan-400 rounded-full"></div>
                  </div>
                </div>
              ))}
              <div className="flex justify-between items-center opacity-50">
                <span>ANPR captures</span>
                <div className="w-8 h-4 bg-slate-800 rounded-full flex items-center p-0.5 justify-start">
                  <div className="w-3 h-3 bg-slate-500 rounded-full"></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ======== CENTER: 3D MAP & CHART ======== */}
        <div className="flex-1 relative flex flex-col bg-black">
          <div className="absolute top-2 left-2 right-2 z-10 flex gap-2 pointer-events-none">
            <div className="bg-black/80 border border-slate-800 px-2 py-1 rounded text-slate-400 shadow-[0_0_10px_rgba(0,0,0,0.5)]">
              5 FPS · edge inference · TensorRT
            </div>
          </div>

          <div className="flex-1 relative invert hue-rotate-180 brightness-90 contrast-125">
            <Map
              {...viewState}
              onMove={evt => setViewState(evt.viewState)}
              maxZoom={18}
              mapStyle={{
                version: 8,
                sources: { 'osm': { type: 'raster', tiles: ['https://a.tile.openstreetmap.org/{z}/{x}/{y}.png'], tileSize: 256 } },
                layers: [{ id: 'osm-layer', type: 'raster', source: 'osm', minzoom: 0, maxzoom: 18 }]
              }}
              style={{ width: '100%', height: '100%' }}
              attributionControl={false}
            >
              {/* === RED 3D HUD MARKER (Pothole) === */}
              <Marker longitude={77.2090} latitude={28.6139} anchor="bottom">
                <div className="relative flex flex-col items-center justify-end h-28">
                  {/* Floating Data Tag */}
                  <div className="bg-rose-950/80 border border-rose-500/50 text-rose-400 text-[10px] font-mono px-2 py-1 rounded shadow-[0_0_10px_rgba(225,29,72,0.4)] mb-1 backdrop-blur-md z-30 flex gap-2 items-center">
                    <span className="w-1.5 h-1.5 bg-rose-500 rounded-full animate-pulse"></span>
                    POTHOLE_DETECTED
                  </div>
                  {/* Laser Connector Line */}
                  <div className="w-[1px] h-12 bg-gradient-to-t from-rose-500 to-transparent z-20"></div>
                  {/* Ground Target Lock */}
                  <div className="absolute bottom-0 flex justify-center items-center" style={{ transform: 'rotateX(60deg)' }}>
                    <div className="w-10 h-10 border border-rose-500/40 rounded-full absolute animate-[spin_3s_linear_infinite]">
                       <div className="absolute top-0 left-1/2 w-[1px] h-2 bg-rose-500"></div>
                       <div className="absolute bottom-0 left-1/2 w-[1px] h-2 bg-rose-500"></div>
                       <div className="absolute left-0 top-1/2 w-2 h-[1px] bg-rose-500"></div>
                       <div className="absolute right-0 top-1/2 w-2 h-[1px] bg-rose-500"></div>
                    </div>
                    <div className="w-6 h-6 border-2 border-rose-500 rounded-full animate-ping opacity-50 absolute"></div>
                    <div className="w-2 h-2 bg-rose-500 rounded-full shadow-[0_0_10px_#e11d48]"></div>
                  </div>
                </div>
              </Marker>

              {/* === CYAN 3D HUD MARKER (Crack) === */}
              <Marker longitude={77.2150} latitude={28.6180} anchor="bottom">
                <div className="relative flex flex-col items-center justify-end h-32">
                  {/* Floating Data Tag */}
                  <div className="bg-cyan-950/80 border border-cyan-400/50 text-cyan-300 text-[10px] font-mono px-2 py-1 rounded shadow-[0_0_10px_rgba(34,211,238,0.4)] mb-1 backdrop-blur-md z-30 flex gap-2 items-center">
                    <span className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-pulse"></span>
                    CRACK_DETECTED
                  </div>
                  {/* Laser Connector Line */}
                  <div className="w-[1px] h-16 bg-gradient-to-t from-cyan-400 to-transparent z-20"></div>
                  {/* Ground Target Lock */}
                  <div className="absolute bottom-0 flex justify-center items-center" style={{ transform: 'rotateX(60deg)' }}>
                    <div className="w-10 h-10 border border-cyan-400/40 rounded-full absolute animate-[spin_3s_linear_infinite_reverse]">
                       <div className="absolute top-0 left-1/2 w-[1px] h-2 bg-cyan-400"></div>
                       <div className="absolute bottom-0 left-1/2 w-[1px] h-2 bg-cyan-400"></div>
                       <div className="absolute left-0 top-1/2 w-2 h-[1px] bg-cyan-400"></div>
                       <div className="absolute right-0 top-1/2 w-2 h-[1px] bg-cyan-400"></div>
                    </div>
                    <div className="w-6 h-6 border-2 border-cyan-400 rounded-full animate-ping opacity-50 absolute"></div>
                    <div className="w-2 h-2 bg-cyan-400 rounded-full shadow-[0_0_10px_#22d3ee]"></div>
                  </div>
                </div>
              </Marker>
            </Map>
          </div>

          <div className="h-40 bg-[#070b14]/95 border-t border-slate-800 p-4 relative z-20 shadow-[0_-10px_30px_rgba(0,0,0,0.5)]">
            <h3 className="text-slate-400 mb-2 tracking-widest flex items-center gap-2">
              <svg className="w-3 h-3 text-cyan-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path></svg>
              TRAFFIC DENSITY - DETECTIONS PER 30 MIN
            </h3>
            <div className="absolute left-4 bottom-4 top-10 flex flex-col justify-between text-slate-600 text-[8px]">
              <span>100</span><span>75</span><span>25</span><span>0</span>
            </div>
            <div className="ml-6 h-full border-b border-l border-slate-700/50 relative">
              <svg className="w-full h-full absolute inset-0" preserveAspectRatio="none" viewBox="0 0 100 100">
                <path d="M0,80 Q25,30 50,70 T100,50 L100,100 L0,100 Z" fill="rgba(6, 182, 212, 0.1)" />
                <path d="M0,80 Q25,30 50,70 T100,50" fill="none" stroke="#06b6d4" strokeWidth="2" />
                <path d="M0,95 Q30,90 60,98 T100,90" fill="none" stroke="#f43f5e" strokeWidth="1" />
                <path d="M0,90 Q40,80 70,95 T100,85" fill="none" stroke="#f59e0b" strokeWidth="1" />
              </svg>
            </div>
          </div>
        </div>

        {/* ======== RIGHT SIDEBAR ======== */}
        <div className="w-80 border-l border-slate-800 bg-[#070b14]/90 backdrop-blur-md flex flex-col z-20">
          <div className="flex justify-between items-center p-4 border-b border-slate-800">
            <h2 className="text-xs text-slate-300 tracking-widest">LIVE INCIDENT FEED</h2>
            <span className="text-slate-200 bg-slate-800 px-2 py-0.5 rounded">48</span>
          </div>
          
          <div className="flex-1 overflow-y-auto">
            <div className="p-3 border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors cursor-pointer">
              <div className="flex justify-between mb-1">
                <span className="text-cyan-400 font-bold flex items-center gap-2">
                  <span className="text-[8px] animate-pulse">●</span> CRACK DETECTED
                </span>
                <span className="text-slate-600">1s ago</span>
              </div>
              <div className="flex gap-3 text-slate-400 mb-1">
                <span className="text-slate-300">D10</span>
                <span>conf <span className="text-slate-200">89%</span></span>
              </div>
              <div className="text-slate-500">28.6180, 77.2150</div>
            </div>

             <div className="p-3 border-b border-slate-800/50 bg-rose-950/20 hover:bg-rose-900/20 transition-colors cursor-pointer border-l-2 border-l-rose-500">
              <div className="flex justify-between mb-1">
                <span className="text-rose-500 font-bold flex items-center gap-2">
                  <span className="text-[8px] animate-pulse">⚠️</span> POTHOLE
                </span>
                <span className="text-slate-600">2s ago</span>
              </div>
              <div className="flex gap-3 text-slate-400 mb-1">
                <span className="text-slate-300">D40</span>
                <span>conf <span className="text-slate-200">94%</span></span>
                <span className="text-rose-500">sev 4</span>
              </div>
              <div className="text-slate-500">28.6139, 77.2090</div>
            </div>
          </div>
        </div>

      </div>
    </main>
  );
}