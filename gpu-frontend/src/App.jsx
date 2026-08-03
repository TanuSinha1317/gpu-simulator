import { useState, useEffect, useRef } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from "recharts";

function App() {
  const [isCompareMode, setIsCompareMode] = useState(false);
  const [modelSize, setModelSize] = useState(30);
  const [activationCheckpointing, setActivationCheckpointing] = useState(false);
  const [interconnect, setInterconnect] = useState("NVLink");
  const [topologyType, setTopologyType] = useState("Ring");
  const [straggler, setStraggler] = useState(false);

  // --- ACCELERATOR SCALE CONTROL ---
  const [gpus, setGpus] = useState(4);

  // --- CONFIG A INPUT STATE ---
  const [dpA, setDpA] = useState(2);
  const [ppA, setPpA] = useState(2);
  const [tpA, setTpA] = useState(1);
  const [mbA, setMbA] = useState(4);
  const [resultA, setResultA] = useState(null);

  // --- CONFIG B INPUT STATE ---
  const [dpB, setDpB] = useState(1);
  const [ppB, setPpB] = useState(2);
  const [tpB, setTpB] = useState(2);
  const [mbB, setMbB] = useState(4);
  const [resultB, setResultB] = useState(null);

  // --- DISCRETE HARDWARE TIME ENGINE ---
  const [isSimulating, setIsSimulating] = useState(false);
  const [currentPhase, setCurrentPhase] = useState("IDLE"); // IDLE, CPU_TO_GPU, DISTRIBUTE, COMPUTE, SYNC, GATHER, GPU_TO_CPU, DONE
  const [simTimeMs, setSimTimeMs] = useState(0);
  const [maxDuration, setMaxDuration] = useState(100);
  const timerRef = useRef(null);

  // Phase Definitions & Runtimes (Expanded intentionally for crisp readability)
  const PHASE_TIMINGS = [
    { name: "CPU_TO_GPU", duration: 2500, label: "Phase 1: Loading Data from Host CPU to Cluster Core via PCIe" },
    { name: "DISTRIBUTE", duration: 2000, label: "Phase 2: Scatter-Distributing Batches Across Accel Mesh" },
    { name: "COMPUTE",    duration: 4000, label: "Phase 3: Deep Compute Execution (Forward/Backward Passes)" },
    { name: "SYNC",       duration: 3500, label: "Phase 4: Multi-Dimensional Gradient Synchronization Cross-Exchange" },
    { name: "GATHER",     duration: 1500, label: "Phase 5: Gathering Inter-Node Output States" },
    { name: "GPU_TO_CPU", duration: 2000, label: "Phase 6: Offloading Weights & Checkpoints back to Host CPU System Memory" }
  ];

  const totalSimDuration = PHASE_TIMINGS.reduce((sum, p) => sum + p.duration, 0);

  const runSimulation = async () => {
    setIsSimulating(false);
    setCurrentPhase("IDLE");
    setSimTimeMs(0);
    if (timerRef.current) clearInterval(timerRef.current);

    const payloadA = {
      gpus: intVal(gpus),
      dp_degree: intVal(dpA),
      pp_degree: intVal(ppA),
      tp_degree: intVal(tpA),
      micro_batches: intVal(mbA),
      model_size: parseFloat(modelSize),
      interconnect,
      topology_type: topologyType,
      straggler,
      activation_checkpointing: activationCheckpointing,
    };

    try {
      const resA = await fetch("http://127.0.0.1:8000/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payloadA),
      });
      const dataA = await resA.json();
      setResultA(dataA);

      if (isCompareMode) {
        const payloadB = {
          gpus: intVal(gpus),
          dp_degree: intVal(dpB),
          pp_degree: intVal(ppB),
          tp_degree: intVal(tpB),
          micro_batches: intVal(mbB),
          model_size: parseFloat(modelSize),
          interconnect,
          topology_type: topologyType,
          straggler,
          activation_checkpointing: activationCheckpointing,
        };
        const resB = await fetch("http://127.0.0.1:8000/simulate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payloadB),
        });
        const dataB = await resB.json();
        setResultB(dataB);
      }

      // Read backend execution times or establish fallbacks if OOM triggered
      const maxTime = Math.max(dataA?.total_time || 0, (isCompareMode && dataB?.total_time) || 0, 100);
      setMaxDuration(maxTime);

      // Begin Step-by-Step Visible Execution Animation Run Loop
      if (!dataA?.is_oom) {
        setIsSimulating(true);
        startHardwareClock();
      }
    } catch (err) {
      console.error("Simulation dispatch network failure:", err);
    }
  };

  const startHardwareClock = () => {
    let startTimestamp = performance.now();
    
    timerRef.current = setInterval(() => {
      const elapsed = performance.now() - startTimestamp;
      if (elapsed >= totalSimDuration) {
        setSimTimeMs(totalSimDuration);
        setCurrentPhase("DONE");
        setIsSimulating(false);
        clearInterval(timerRef.current);
      } else {
        setSimTimeMs(elapsed);
        
        // Track down the matching discrete runtime execution block phase
        let accumulatedTime = 0;
        let foundPhase = "IDLE";
        for (const p of PHASE_TIMINGS) {
          accumulatedTime += p.duration;
          if (elapsed < accumulatedTime) {
            foundPhase = p.name;
            break;
          }
        }
        setCurrentPhase(foundPhase);
      }
    }, 30);
  };

  const intVal = (val) => parseInt(val, 10) || 1;

  useEffect(() => {
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, []);

  // Compute scale multipliers to move the Gantt bars synchronously with Phase 3 & 4
  const computeGanttProgressPct = () => {
    if (currentPhase === "IDLE" || currentPhase === "CPU_TO_GPU" || currentPhase === "DISTRIBUTE") return 0;
    if (currentPhase === "GATHER" || currentPhase === "GPU_TO_CPU" || currentPhase === "DONE") return 100;
    
    // Calculate elapsed window relative strictly to compute + sync windows
    const elapsedComputeSync = simTimeMs - (2500 + 2000);
    const combinedDuration = 4000 + 3500;
    return Math.min(100, Math.max(0, (elapsedComputeSync / combinedDuration) * 100));
  };

  const currentGanttPct = computeGanttProgressPct();

  // --- DYNAMIC SELECTION-DRIVEN TOPOLOGY NETWORK GRAPH ARCHITECTURE ---
  const renderLiveHardwareMap = (dpVal, ppVal, tpVal, title) => {
    const dp = intVal(dpVal);
    const pp = intVal(ppVal);
    const tp = intVal(tpVal);
    const count = Math.min(gpus, 16);

    const cx = 200;
    const cy = 110;
    const rx = 130;
    const ry = 55;

    // Build Host CPU Location Coordinate at Top
    const cpuNode = { x: cx, y: 25, label: "HOST CPU" };

    // Build Accelerator Coordinate Matrix Layout dynamically mapping selections
    const gpuNodes = Array.from({ length: count }, (_, i) => {
      if (topologyType === "Ring") {
        const angle = (i * 2 * Math.PI) / count - Math.PI / 2;
        return {
          x: cx + rx * Math.cos(angle),
          y: cy + ry * Math.sin(angle) + 15,
          id: i + 1,
          dpRank: i % dp,
          ppRank: Math.floor(i / dp) % pp,
          tpRank: Math.floor(i / (dp * pp)) % tp
        };
      } else {
        // Tree Architecture Spans outward symmetrically around a Central Fabric switch hub
        const spacing = 320 / (count + 1);
        return {
          x: 40 + spacing * (i + 1),
          y: 155,
          id: i + 1,
          dpRank: i % dp,
          ppRank: Math.floor(i / dp) % pp,
          tpRank: Math.floor(i / (dp * pp)) % tp
        };
      }
    });

    // Extract dynamic labeling strings based on Selected Interconnect Speeds
    const activeBusLabel = interconnect === "NVLink" ? "NVLink (450 GB/s)" : "PCIe Gen4 P2P (32 GB/s)";

    return (
      <div style={{ background: "#0b0d19", padding: "12px", borderRadius: "6px", border: "1px solid #1e264a" }}>
        <div style={{ display: "flex", justifyContent: "between", alignItems: "center", marginBottom: "6px" }}>
          <div style={{ fontSize: "11px", color: "#60a5fa", fontWeight: "bold" }}>{title}</div>
          <div style={{ fontSize: "9px", color: "#10b981", textTransform: "uppercase" }}>System State: <strong>{currentPhase}</strong></div>
        </div>
        
        <div style={{ background: "#020306", borderRadius: "4px", position: "relative" }}>
          <svg viewBox="0 0 400 210" style={{ width: "100%", height: "auto" }}>
            <defs>
              <marker id="cpuArrow" viewBox="0 0 10 10" refX="15" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                <path d="M 0 2 L 8 5 L 0 8 z" fill="#a855f7" />
              </marker>
              <marker id="busArrow" viewBox="0 0 10 10" refX="16" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                <path d="M 0 2 L 8 5 L 0 8 z" fill="#22c55e" />
              </marker>
            </defs>

            {/* ── PHASE 1 & 6: HOST CPU-TO-ACCELERATOR BUS LINK CHANNELS ── */}
            {gpuNodes.map((node, i) => {
              const isFirstOrRoot = topologyType === "Ring" ? i === 0 : true;
              if (!isFirstOrRoot) return null;
              
              const isBusActive = currentPhase === "CPU_TO_GPU" || currentPhase === "GPU_TO_CPU";
              return (
                <g key={`cpu-link-${i}`}>
                  <line 
                    x1={cpuNode.x} y1={cpuNode.y} x2={topologyType === "Tree" ? cx : node.x} y2={topologyType === "Tree" ? 95 : node.y}
                    stroke={isBusActive ? "#a855f7" : "#1f293d"} strokeWidth={isBusActive ? "2" : "1"}
                    strokeDasharray={isBusActive ? "4 2" : "0"}
                  />
                  <text x={(cpuNode.x + node.x)/2 - 15} y={(cpuNode.y + node.y)/2 - 5} fill="#a855f7" fontSize="6" fontWeight="bold">PCIe System Bus</text>
                  
                  {currentPhase === "CPU_TO_GPU" && (
                    <circle r="3.5" fill="#c084fc">
                      <animateMotion dur="1.8s" repeatCount="indefinite" path={`M ${cpuNode.x} ${cpuNode.y} L ${topologyType === "Tree" ? cx : node.x} ${topologyType === "Tree" ? 95 : node.y}`} />
                    </circle>
                  )}
                  {currentPhase === "GPU_TO_CPU" && (
                    <circle r="3.5" fill="#f43f5e">
                      <animateMotion dur="1.8s" repeatCount="indefinite" path={`M ${topologyType === "Tree" ? cx : node.x} ${topologyType === "Tree" ? 95 : node.y} L ${cpuNode.x} ${cpuNode.y}`} />
                    </circle>
                  )}
                </g>
              );
            })}

            {/* ── INTRA-CLUSTER MESH TOPOLOGY INTERCONNECTS ── */}
            {topologyType === "Tree" && (
              /* Core Switching Hub for Central Fabric configuration mappings */
              <g>
                <circle cx={cx} cy={95} r="10" fill="#1e293b" stroke="#3b82f6" strokeWidth="1.5" />
                <text x={cx} y={98} fill="#3b82f6" fontSize="6" fontWeight="bold" textAnchor="middle">SWITCH</text>
              </g>
            )}

            {gpuNodes.map((node, idx) => {
              if (topologyType === "Ring") {
                const nextNode = gpuNodes[(idx + 1) % count];
                const isLinkActive = currentPhase === "DISTRIBUTE" || currentPhase === "SYNC" || currentPhase === "GATHER";
                const midX = (node.x + nextNode.x) / 2;
                const midY = (node.y + nextNode.y) / 2;

                return (
                  <g key={`ring-link-${idx}`}>
                    <line 
                      x1={node.x} y1={node.y} x2={nextNode.x} y2={nextNode.y} 
                      stroke={isLinkActive ? "#10b981" : "#1e293b"} 
                      strokeWidth={isLinkActive ? "1.75" : "1.25"} 
                      strokeDasharray={isLinkActive ? "5 3" : "0"}
                    />
                    {/* Interconnect speed classification metric overlay */}
                    {idx === 0 && (
                      <text x={midX} y={midY - 4} fill="#94a3b8" fontSize="5.5" textAnchor="middle" backgroundColor="#020306">{activeBusLabel}</text>
                    )}
                    
                    {(currentPhase === "DISTRIBUTE" || currentPhase === "SYNC") && (
                      <circle r="3" fill="#34d399">
                        <animateMotion dur="2.5s" repeatCount="indefinite" path={`M ${node.x} ${node.y} L ${nextNode.x} ${nextNode.y}`} />
                      </circle>
                    )}
                  </g>
                );
              } else {
                const isLinkActive = currentPhase === "DISTRIBUTE" || currentPhase === "SYNC" || currentPhase === "GATHER";
                return (
                  <g key={`tree-link-${idx}`}>
                    <line 
                      x1={cx} y1={95} x2={node.x} y2={node.y} 
                      stroke={isLinkActive ? "#3b82f6" : "#1e293b"} 
                      strokeWidth={isLinkActive ? "1.5" : "1"} 
                    />
                    {idx === Math.floor(count/2) && (
                      <text x={(cx + node.x)/2 + 20} y={(95 + node.y)/2} fill="#94a3b8" fontSize="5.5" textAnchor="middle">{activeBusLabel}</text>
                    )}
                    {currentPhase === "DISTRIBUTE" && (
                      <circle r="2.5" fill="#60a5fa">
                        <animateMotion dur="2.2s" repeatCount="indefinite" path={`M ${cx} 95 L ${node.x} ${node.y}`} />
                      </circle>
                    )}
                    {currentPhase === "SYNC" && (
                      <circle r="2.5" fill="#fbbf24">
                        <animateMotion dur="2.2s" repeatCount="indefinite" path={`M ${node.x} ${node.y} L ${cx} 95`} />
                      </circle>
                    )}
                  </g>
                );
              }
            })}

            {/* HOST PHYSICAL CPU MODULE BOX */}
            <g transform={`translate(${cpuNode.x - 28}, ${cpuNode.y - 12})`}>
              <rect width="56" height="20" rx="3" fill="#2d1b4e" stroke="#a855f7" strokeWidth="1.5" />
              <text x="28" y="12" fill="#fff" fontSize="7" fontWeight="bold" textAnchor="middle">SYSTEM CPU</text>
            </g>

            {/* DYNAMIC PROCESSING ACCELERATOR GPU NODE GRAPHICS */}
            {gpuNodes.map((gpu) => {
              // Node operational state mapping layout
              let targetStroke = "#4f46e5";
              let targetBg = "#111422";
              let statusLabel = "STANDBY";

              if (currentPhase === "CPU_TO_GPU" || currentPhase === "DISTRIBUTE") {
                targetStroke = "#a855f7";
                statusLabel = "RECEIVING";
              } else if (currentPhase === "COMPUTE") {
                targetStroke = "#ec4899";
                targetBg = "#31102f";
                statusLabel = "COMPUTING";
              } else if (currentPhase === "SYNC") {
                targetStroke = "#eab308";
                statusLabel = "SYNC LOOP";
              } else if (currentPhase === "GATHER" || currentPhase === "GPU_TO_CPU") {
                targetStroke = "#f43f5e";
                statusLabel = "OFFLOADING";
              } else if (currentPhase === "DONE") {
                targetStroke = "#22c55e";
                statusLabel = "FINISHED";
              }

              return (
                <g key={gpu.id} transform={`translate(${gpu.x - 24}, ${gpu.y - 19})`}>
                  <rect width="48" height="36" rx="4" fill={targetBg} stroke={targetStroke} strokeWidth="1.75" />
                  <text x="24" y="11" fill="#fff" fontSize="7" fontWeight="bold" textAnchor="middle">GPU-{gpu.id}</text>
                  <text x="24" y="21" fill={targetStroke} fontSize="5.5" fontWeight="bold" textAnchor="middle">{statusLabel}</text>
                  <text x="24" y="30" fill="#64748b" fontSize="5" textAnchor="middle">D:{gpu.dpRank} P:{gpu.ppRank} T:{gpu.tpRank}</text>
                </g>
              );
            })}
          </svg>
        </div>
      </div>
    );
  };

  // --- RE-ENGINEERED SYNCHRONIZED INCREMENTAL GANTT CHART CONTAINER ---
  const renderInteractiveGanttTimeline = (resultPayload) => {
    if (!resultPayload || resultPayload.is_oom) return null;

    return resultPayload.timeline.map((track) => (
      <div key={track.gpu} style={{ display: "flex", alignItems: "center", marginBottom: "6px" }}>
        <div style={{ width: "48px", fontSize: "9px", color: "#a1a1aa", flexShrink: 0 }}>{track.gpu}</div>
        <div style={{
          flex: 1, display: "block", position: "relative", background: "#090b11",
          height: "18px", borderRadius: "3px", border: "1px solid #141722", overflow: "hidden",
        }}>
          {track.events?.map((event, i) => {
            const startPct = (event.start / maxDuration) * 100;
            const endPct   = (event.end   / maxDuration) * 100;
            const widthPct = endPct - startPct;

            // Enforce clock limits to dynamically lock bars behind current phase milestones
            if (startPct > currentGanttPct) return null;
            const visibleWidthPct = Math.min(widthPct, currentGanttPct - startPct);

            const bg =
              event.state === "forward"       ? "#a855f7" : 
              event.state === "recompute"     ? "#3b82f6" : 
              event.state === "backward"      ? "#ec4899" : 
              event.state === "tp_sync"       ? "#f59e0b" : 
              event.state === "reduce_all"    ? "#10b981" : 
              event.state === "optimizer"     ? "#ef4444" : 
              event.state === "communication" ? "#d97706" : "#11131e";

            return (
              <div
                key={i}
                style={{
                  position: "absolute", left: `${startPct}%`, width: `${visibleWidthPct}%`, height: "100%", background: bg,
                  borderRight: "1px solid #020305", display: "flex", alignItems: "center", justifyContent: "left",
                  fontSize: "6.5px", color: "#fff", paddingLeft: "2px", boxSizing: "border-box"
                }}
              >
                {visibleWidthPct > 8 && event.state[0].toUpperCase()}
              </div>
            );
          })}
        </div>
        <div style={{ fontSize: "8px", color: "#71717a", marginLeft: "8px", width: "95px", textAlign: "right", flexShrink: 0 }}>
          W:{track.static_weight_gb}G | A:{track.peak_activation_gb}G
        </div>
      </div>
    ));
  };

  const inputStyle = { width: "100%", background: "#06070a", border: "1px solid #27272a", color: "#fff", fontSize: "11px", padding: "4px", boxSizing: "border-box" };
  const selectStyle = { ...inputStyle, padding: "6px", borderRadius: "4px" };
  const metricBox = (label, value) => (
    <div style={{ padding: "6px", background: "#06070a", fontSize: "11px", border: "1px solid #1a1f36" }}>{label}: <strong>{value}</strong></div>
  );

  return (
    <div style={{ display: "flex", height: "100vh", background: "#06070a", color: "#e4e4e7", fontFamily: "monospace" }}>

      {/* ── SIDEBAR CONTROL INTERACTION RADAR PANEL ── */}
      <div style={{ width: "350px", background: "#0e1017", padding: "18px", borderRight: "1px solid #1c1e29", boxSizing: "border-box", overflowY: "auto", flexShrink: 0 }}>
        <h2 style={{ margin: "0 0 5px 0", color: "#fff", fontSize: "14px", letterSpacing: "1px" }}>CLUSTER CONSOLE</h2>
        <p style={{ color: "#71717a", fontSize: "10px", margin: "0 0 15px 0" }}>Hardware Cluster Engine Tuning Layout</p>

        {/* STEP 1: SELECT NUMBER OF ACCELERATOR CORES */}
        <div style={{ marginBottom: "15px", border: "1px solid #3b82f6", padding: "8px", borderRadius: "4px", background: "rgba(59,130,246,0.03)" }}>
          <label style={{ display: "block", fontSize: "10px", color: "#60a5fa", fontWeight: "bold", marginBottom: "4px" }}>[STEP 1] ACCELERATOR SCALE VOLUME</label>
          <select value={gpus} onChange={(e) => setGpus(Number(e.target.value))} style={selectStyle}>
            <option value={2}>2 GPU Compact Chassis</option>
            <option value={4}>4 GPU Standard Node (Intra-Node)</option>
            <option value={8}>8 GPU High-Density Mainframe (Intra-Node)</option>
            <option value={16}>16 GPU Distributed Cluster Block (Cross-Node)</option>
          </select>
        </div>

        {/* STEP 2: CHOOSE SYSTEM FABRIC LAYOUT TO ARRANGE GEOMETRY */}
        <div style={{ marginBottom: "12px" }}>
          <label style={{ display: "block", fontSize: "10px", color: "#a1a1aa", marginBottom: "4px" }}>[STEP 2] INTERCONNECT TOPOLOGY ENGAGED</label>
          <select value={topologyType} onChange={(e) => setTopologyType(e.target.value)} style={selectStyle}>
            <option value="Ring">High-Speed Ring All-Reduce Loop Mesh</option>
            <option value="Tree">Centralized Core Spine Switch Infrastructure</option>
          </select>
        </div>

        <div style={{ marginBottom: "12px" }}>
          <label style={{ display: "block", fontSize: "10px", color: "#a1a1aa", marginBottom: "4px" }}>HARDWARE DATA TRANSFER CHANNEL TIER</label>
          <select value={interconnect} onChange={(e) => setInterconnect(e.target.value)} style={selectStyle}>
            <option value="NVLink">NVIDIA NVLink Fabric Channel (450 GB/s)</option>
            <option value="PCIe">PCIe Gen 4 Standard Shared Bus Link (32 GB/s)</option>
          </select>
        </div>

        <div style={{ marginBottom: "15px", background: "#11141e", padding: "10px", borderRadius: "6px", border: "1px solid #1c2135" }}>
          <label style={{ display: "block", fontSize: "9px", color: "#a1a1aa", margin: "2px 0 2px" }}>PARAMETER SCALE FOOTPRINT (BILLIONS)</label>
          <input type="number" value={modelSize} onChange={(e) => setModelSize(parseFloat(e.target.value) || 1)} style={{ ...inputStyle, padding: "6px", borderRadius: "4px" }} />
        </div>

        <hr style={{ border: "0", borderTop: "1px solid #1c1e29", margin: "12px 0" }} />

        {/* Distributed 3D Tensor Parallel Sharding A Controls */}
        <div style={{ background: "#11141e", padding: "10px", borderRadius: "6px", border: "1px solid #1e2235", marginBottom: "12px" }}>
          <span style={{ fontSize: "10px", color: "#a855f7", fontWeight: "bold", display: "block", marginBottom: "8px" }}> 3D SHARDING GEOMETRY — CONFIG A</span>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "6px", marginBottom: "6px" }}>
            <div><label style={{ fontSize: "8px", color: "#a1a1aa" }}>DP DEG</label><input type="number" min="1" value={dpA} onChange={(e) => setDpA(e.target.value)} style={inputStyle} /></div>
            <div><label style={{ fontSize: "8px", color: "#a1a1aa" }}>PP STG</label><input type="number" min="1" value={ppA} onChange={(e) => setPpA(e.target.value)} style={inputStyle} /></div>
            <div><label style={{ fontSize: "8px", color: "#a1a1aa" }}>TP TSN</label><input type="number" min="1" value={tpA} onChange={(e) => setTpA(e.target.value)} style={inputStyle} /></div>
          </div>
          <label style={{ fontSize: "8px", color: "#a1a1aa" }}>MICRO-BATCHES COUNT (m)</label>
          <input type="number" min="1" value={mbA} onChange={(e) => setMbA(e.target.value)} style={inputStyle} />
        </div>

        {/* Distributed 3D Tensor Parallel Sharding B Controls */}
        {isCompareMode && (
          <div style={{ background: "#16121e", padding: "10px", borderRadius: "6px", border: "1px solid #2d1e3d", marginBottom: "12px" }}>
            <span style={{ fontSize: "10px", color: "#ec4899", fontWeight: "bold", display: "block", marginBottom: "8px" }}> 3D SHARDING GEOMETRY — CONFIG B</span>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "6px", marginBottom: "6px" }}>
              <div><label style={{ fontSize: "8px", color: "#a1a1aa" }}>DP DEG</label><input type="number" min="1" value={dpB} onChange={(e) => setDpB(e.target.value)} style={inputStyle} /></div>
              <div><label style={{ fontSize: "8px", color: "#a1a1aa" }}>PP STG</label><input type="number" min="1" value={ppB} onChange={(e) => setPpB(e.target.value)} style={inputStyle} /></div>
              <div><label style={{ fontSize: "8px", color: "#a1a1aa" }}>TP TSN</label><input type="number" min="1" value={tpB} onChange={(e) => setTpB(e.target.value)} style={inputStyle} /></div>
            </div>
            <label style={{ fontSize: "8px", color: "#a1a1aa" }}>MICRO-BATCHES COUNT (m)</label>
            <input type="number" min="1" value={mbB} onChange={(e) => setMbB(e.target.value)} style={inputStyle} />
          </div>
        )}

        <div style={{ marginBottom: "10px", display: "flex", alignItems: "center", gap: "6px" }}>
          <input type="checkbox" id="chkpt" checked={activationCheckpointing} onChange={(e) => setActivationCheckpointing(e.target.checked)} />
          <label htmlFor="chkpt" style={{ fontSize: "10px", color: "#c084fc", cursor: "pointer", fontWeight: "bold" }}>ACTIVATION CHECKPOINTING</label>
        </div>
        <div style={{ marginBottom: "15px", display: "flex", alignItems: "center", gap: "6px" }}>
          <input type="checkbox" id="strag" checked={straggler} onChange={(e) => setStraggler(e.target.checked)} />
          <label htmlFor="strag" style={{ fontSize: "10px", color: "#e4e4e7", cursor: "pointer" }}>INJECT HARDWARE STRAGGLER STALL</label>
        </div>

        <button onClick={runSimulation} style={{ width: "100%", padding: "11px", background: "#6d28d9", color: "#fff", border: "none", borderRadius: "4px", fontWeight: "bold", cursor: "pointer", fontSize: "11px", letterSpacing: "0.2px" }}>
          ▶ RUN DISCRETE HARDWARE SIMULATION
        </button>
      </div>

      {/* ── MAIN TELESCOPIC DASHBOARD CANVAS FRAME ── */}
      <div style={{ flex: 1, padding: "20px", overflowY: "auto", boxSizing: "border-box" }}>
        <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid #1f202e", paddingBottom: "10px", marginBottom: "15px" }}>
          <div>
            <h1 style={{ fontSize: "15px", margin: "0", color: "#fff" }}>DISTRIBUTED CLUSTER DISCRETE HARDWARE CORE RUNNER</h1>
            <p style={{ color: "#71717a", fontSize: "10px", margin: "2px 0 0 0" }}>Observe Host CPU Offloading, Mesh Communication Waves, and Gantt Trace Logs Locks</p>
          </div>
          <button onClick={() => { setIsCompareMode(!isCompareMode); setResultB(null); }} style={{ background: isCompareMode ? "#ec4899" : "#27272a", color: "#fff", border: "none", borderRadius: "4px", padding: "8px 12px", fontSize: "11px", fontWeight: "bold", cursor: "pointer" }}>
            {isCompareMode ? "✕ Close Side Comparison" : "🔄 Enable Twin Side-by-Side Compare"}
          </button>
        </header>

        {/* HUD PHASE LOGS RUNNER BAR BANNER */}
        {currentPhase !== "IDLE" && (
          <div style={{ background: "#0c1221", border: "1px solid #3b82f6", borderRadius: "4px", padding: "8px 12px", marginBottom: "15px", fontSize: "11px" }}>
            <span style={{ color: "#60a5fa", fontWeight: "bold" }}>⚡ CURRENT RUN LOOP STEP: </span> 
            <span style={{ color: "#fff" }}>
              {PHASE_TIMINGS.find(p => p.name === currentPhase)?.label || "Simulation complete. System in Standby State."}
            </span>
          </div>
        )}

        {/* TOPOLOGY HARNESS MAPPING GRAPHICS LAYER */}
        <div style={{ display: "grid", gridTemplateColumns: isCompareMode ? "1fr 1fr" : "1fr", gap: "16px", marginBottom: "16px" }}>
          {renderLiveHardwareMap(dpA, ppA, tpA, "CONFIG A — ACCELERATOR MESH GRAPH LAYOUT")}
          {isCompareMode && renderLiveHardwareMap(dpB, ppB, tpB, "CONFIG B — ACCELERATOR MESH GRAPH LAYOUT")}
        </div>

        {/* HORIZONTAL GANTT EXECUTION PIPELINE MARKS LOG SYSTEM */}
        <div style={{ display: "grid", gridTemplateColumns: isCompareMode ? "1fr 1fr" : "1fr", gap: "16px" }}>
          
          {/* Config A Gantt Panel */}
          <div style={{ background: "#0e1017", padding: "15px", borderRadius: "6px", border: "1px solid #1e2235" }}>
            <h3 style={{ margin: "0 0 10px 0", fontSize: "12px", color: "#a855f7" }}>CONFIG A — SYNCHRONIZED EXECUTION TIME LOG GANTT</h3>
            
            <div style={{ display: "flex", gap: "8px", fontSize: "8.5px", marginBottom: "12px", flexWrap: "wrap", background: "#05060a", padding: "6px", borderRadius: "4px" }}>
              {[
                ["#a855f7","Forward (F)"], ["#3b82f6","Activation Remat (R)"], ["#ec4899","Backward Gradient (B)"], 
                ["#f59e0b", "TP Sync (T)"], ["#10b981", "Reduce-All (G)"], ["#ef4444", "Weights Update (O)"],
                ["#d97706", "Pipeline Bubble (C)"]
              ].map(([colorCode, name]) => (
                <span key={name} style={{ display: "flex", alignItems: "center", gap: "3px" }}>
                  <span style={{ display: "inline-block", width: "7px", height: "7px", background: colorCode, borderRadius: "1px" }}></span>
                  {name}
                </span>
              ))}
            </div>

            {resultA?.is_oom ? (
              <div style={{ color: "#ef4444", fontSize: "11px", padding: "10px", background: "#450a0a", borderRadius: "4px" }}>
                 OOM LIMIT EXCEEDED: Requested parameters crash physical hardware buffers.
              </div>
            ) : resultA ? (
              <>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px", marginBottom: "12px" }}>
                  {metricBox("Theoretical iteration step time", `${resultA.total_time} ms`)}
                  {metricBox("Cluster hardware compute efficiency", `${resultA.efficiency}%`)}
                </div>
                <div style={{ background: "#020305", padding: "8px", borderRadius: "4px" }}>{renderInteractiveGanttTimeline(resultA)}</div>
              </>
            ) : <p style={{ fontSize: "11px", color: "#52525b", textAlign: "center", padding: "15px 0" }}>Configure matrix limits and click launch above.</p>}
          </div>

          {/* Config B Gantt Panel */}
          {isCompareMode && (
            <div style={{ background: "#0e1017", padding: "15px", borderRadius: "6px", border: "1px solid #2d1e3d" }}>
              <h3 style={{ margin: "0 0 10px 0", fontSize: "12px", color: "#ec4899" }}>CONFIG B — SYNCHRONIZED EXECUTION TIME LOG GANTT</h3>
              
              <div style={{ display: "flex", gap: "8px", fontSize: "8.5px", marginBottom: "12px", flexWrap: "wrap", background: "#05060a", padding: "6px", borderRadius: "4px" }}>
                {[
                  ["#a855f7","Forward (F)"], ["#3b82f6","Activation Remat (R)"], ["#ec4899","Backward Gradient (B)"], 
                  ["#f59e0b", "TP Sync (T)"], ["#10b981", "Reduce-All (G)"], ["#ef4444", "Weights Update (O)"],
                  ["#d97706", "Pipeline Bubble (C)"]
                ].map(([colorCode, name]) => (
                  <span key={name} style={{ display: "flex", alignItems: "center", gap: "3px" }}>
                    <span style={{ display: "inline-block", width: "7px", height: "7px", background: colorCode, borderRadius: "1px" }}></span>
                    {name}
                  </span>
                ))}
              </div>

              {resultB?.is_oom ? (
                <div style={{ color: "#ef4444", fontSize: "11px", padding: "10px", background: "#450a0a", borderRadius: "4px" }}>
                   OOM LIMIT EXCEEDED: Requested parameters crash alternative config buffers.
                </div>
              ) : resultB ? (
                <>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px", marginBottom: "12px" }}>
                    {metricBox("Theoretical iteration step time", `${resultB.total_time} ms`)}
                    {metricBox("Cluster hardware compute efficiency", `${resultB.efficiency}%`)}
                  </div>
                  <div style={{ background: "#020305", padding: "8px", borderRadius: "4px" }}>{renderInteractiveGanttTimeline(resultB)}</div>
                </>
              ) : <p style={{ fontSize: "11px", color: "#52525b", textAlign: "center", padding: "15px 0" }}>Initialize alternative values to perform variance profiling.</p>}
            </div>
          )}

        </div>
      </div>
    </div>
  );
}

export default App;