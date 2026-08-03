from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import math
 
app = FastAPI()
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
class SimulationInput(BaseModel):
    dp_degree: int
    pp_degree: int
    tp_degree: int
    micro_batches: int
    model_size: float
    interconnect: str
    topology_type: str
    straggler: bool
    activation_checkpointing: bool
 
GPU_MEMORY_CAPACITY_GB = 80.0
GPU_PEAK_TFLOAT32_TFLOPS = 156.0
 
NVLINK_SPEED_GBS = 450.0
PCIE_SPEED_GBS = 32.0
INFINIBAND_SPEED_GBS = 50.0
 
def execute_simulator_kernel(cfg: SimulationInput, override_dp_degree: int = None):
    dp = override_dp_degree if override_dp_degree is not None else cfg.dp_degree
    pp = cfg.pp_degree
    tp = cfg.tp_degree
 
    total_gpus = dp * pp * tp
    m = cfg.micro_batches
 
    param_bytes = cfg.model_size * 2 * (10**9)
 
    sharded_params = param_bytes / (tp * pp)
    weight_mem_gb = sharded_params / (10**9)
    grad_mem_gb = sharded_params / (10**9)
    optimizer_state_gb = (sharded_params * 4) / (10**9)
 
    base_activation_per_microbatch = (sharded_params * 1.5) / (10**9)
    if cfg.activation_checkpointing:
        peak_activation_gb = base_activation_per_microbatch * 0.25 * math.log2(m + 1)
    else:
        peak_activation_gb = base_activation_per_microbatch * min(m, pp)
 
    total_memory_per_gpu = weight_mem_gb + grad_mem_gb + optimizer_state_gb + peak_activation_gb
    is_oom = total_memory_per_gpu > GPU_MEMORY_CAPACITY_GB
 
    if is_oom:
        return {"is_oom": True, "total_memory_requested": round(total_memory_per_gpu, 2)}
 
    intra_node_speed = NVLINK_SPEED_GBS if cfg.interconnect.upper() == "NVLINK" else PCIE_SPEED_GBS
    tp_bandwidth = intra_node_speed
    pp_bandwidth = intra_node_speed if total_gpus <= 8 else INFINIBAND_SPEED_GBS
    dp_bandwidth = intra_node_speed if total_gpus <= 8 else INFINIBAND_SPEED_GBS
 
    if cfg.topology_type == "Tree":
        dp_bandwidth *= 0.75
        pp_bandwidth *= 0.85
 
    fw_flops = (2 * param_bytes) / (tp * pp)
    bw_flops = (4 * param_bytes) / (tp * pp)
 
    t_fw_micro = (fw_flops / (GPU_PEAK_TFLOAT32_TFLOPS * (10**12))) * 1000
    t_bw_micro = (bw_flops / (GPU_PEAK_TFLOAT32_TFLOPS * (10**12))) * 1000
    t_recompute = t_fw_micro * 0.33 if cfg.activation_checkpointing else 0.0
 
    if cfg.straggler:
        t_fw_micro *= 1.8
        t_bw_micro *= 1.8
 
    t_comm_tp = 2 * (2 * (tp - 1) / tp) * (weight_mem_gb / tp_bandwidth) * 1000 if tp > 1 else 0.0
    t_comm_pp = (peak_activation_gb / pp_bandwidth) * 1000 if pp > 1 else 0.0
    t_comm_dp = 2 * ((dp - 1) / dp) * (grad_mem_gb / dp_bandwidth) * 1000 if dp > 1 else 0.0
 
    overlapped_bw_capacity = t_bw_micro * 0.70
    effective_dp_overhead = max(0.0, t_comm_dp - overlapped_bw_capacity)
 
    # Allocate explicit room for optimizer engine computations
    t_optimizer = (sharded_params * 16) / (GPU_PEAK_TFLOAT32_TFLOPS * (10**12)) * 1000
 
    total_compute_steps_time = (m * (t_fw_micro + t_recompute + t_bw_micro)) + (t_comm_tp * m * 2)
    pipeline_bubble_overhead = (pp - 1) * (t_fw_micro + t_bw_micro) if pp > 1 else 0.0
 
    total_step_execution_time = total_compute_steps_time + pipeline_bubble_overhead + t_comm_pp + effective_dp_overhead + t_optimizer
 
    ideal_unscaled_time = (t_fw_micro + t_bw_micro) * m
    efficiency_percentage = (ideal_unscaled_time / total_step_execution_time) * 100
 
    comm_overhead_pct = round(((t_comm_pp + effective_dp_overhead + (t_comm_tp * m * 2)) / total_step_execution_time) * 100, 2)
    bubble_fraction_pct = round((pipeline_bubble_overhead / total_step_execution_time) * 100, 2) if pp > 1 else 0.0
 
    return {
        "is_oom": False,
        "max_capacity_allowed": GPU_MEMORY_CAPACITY_GB,
        "total_memory_requested": round(total_memory_per_gpu, 2),
        "static_weight_gb": round(weight_mem_gb + optimizer_state_gb, 2),
        "peak_activation_gb": round(peak_activation_gb, 2),
        "gradient_gb": round(grad_mem_gb, 2),
        "total_time": round(total_step_execution_time, 2),
        "efficiency": round(min(100.0, max(5.0, efficiency_percentage)), 2),
        "comm_overhead_pct": comm_overhead_pct,
        "bubble_fraction_pct": bubble_fraction_pct,
        "t_fw": round(t_fw_micro, 2),
        "t_bw": round(t_bw_micro, 2),
        "t_recompute": round(t_recompute, 2),
        "t_comm_pp": round(t_comm_pp, 2),
        "t_comm_dp": round(effective_dp_overhead, 2),
        "t_comm_tp": round(t_comm_tp, 2),
        "t_optimizer": round(t_optimizer, 2)
    }
 
def build_1f1b_timeline(pp: int, total_gpus: int, m: int, metrics: dict, activation_checkpointing: bool):
    timeline = []
 
    t_fw = metrics["t_fw"]
    t_bw = metrics["t_bw"]
    t_recompute = metrics["t_recompute"]
    t_comm_pp = metrics["t_comm_pp"]
    t_comm_tp = metrics["t_comm_tp"]
    t_comm_dp = metrics["t_comm_dp"]
    t_optimizer = metrics["t_optimizer"]
 
    t_pp_slot = (t_comm_pp / m) if m > 0 else 0.0
 
    for g in range(total_gpus):
        pp_stage = g % pp
        events = []
        ptr = 0.0
 
        warmup_count = pp - pp_stage - 1
        initial_idle = pp_stage * (t_fw + t_pp_slot)
        if initial_idle > 0.0:
            events.append({"state": "idle", "start": round(0.0, 2), "end": round(initial_idle, 2)})
        ptr = initial_idle
 
        for _ in range(warmup_count):
            fw_end = ptr + t_fw
            events.append({"state": "forward", "start": round(ptr, 2), "end": round(fw_end, 2)})
            ptr = fw_end
            if t_comm_tp > 0:
                tp_end = ptr + (t_comm_tp / 2)
                events.append({"state": "tp_sync", "start": round(ptr, 2), "end": round(tp_end, 2)})
                ptr = tp_end
            if t_pp_slot > 0:
                comm_end = ptr + t_pp_slot
                events.append({"state": "communication", "start": round(ptr, 2), "end": round(comm_end, 2)})
                ptr = comm_end
 
        steady_count = m - warmup_count
        for step in range(steady_count):
            fw_end = ptr + t_fw
            events.append({"state": "forward", "start": round(ptr, 2), "end": round(fw_end, 2)})
            ptr = fw_end
 
            if t_comm_tp > 0:
                tp_end = ptr + (t_comm_tp / 2)
                events.append({"state": "tp_sync", "start": round(ptr, 2), "end": round(tp_end, 2)})
                ptr = tp_end
 
            if t_pp_slot > 0 and pp_stage < pp - 1:
                comm_end = ptr + t_pp_slot
                events.append({"state": "communication", "start": round(ptr, 2), "end": round(comm_end, 2)})
                ptr = comm_end
 
            if activation_checkpointing and t_recompute > 0:
                remat_end = ptr + t_recompute
                events.append({"state": "recompute", "start": round(ptr, 2), "end": round(remat_end, 2)})
                ptr = remat_end
 
            bw_end = ptr + t_bw
            events.append({"state": "backward", "start": round(ptr, 2), "end": round(bw_end, 2)})
            ptr = bw_end
 
            if t_comm_tp > 0:
                tp_end = ptr + (t_comm_tp / 2)
                events.append({"state": "tp_sync", "start": round(ptr, 2), "end": round(tp_end, 2)})
                ptr = tp_end
 
            if t_pp_slot > 0 and pp_stage > 0:
                comm_end = ptr + t_pp_slot
                events.append({"state": "communication", "start": round(ptr, 2), "end": round(comm_end, 2)})
                ptr = comm_end
 
        for _ in range(warmup_count):
            if activation_checkpointing and t_recompute > 0:
                remat_end = ptr + t_recompute
                events.append({"state": "recompute", "start": round(ptr, 2), "end": round(remat_end, 2)})
                ptr = remat_end
 
            bw_end = ptr + t_bw
            events.append({"state": "backward", "start": round(ptr, 2), "end": round(bw_end, 2)})
            ptr = bw_end
 
        if t_comm_dp > 0:
            sync_end = ptr + t_comm_dp
            events.append({"state": "reduce_all", "start": round(ptr, 2), "end": round(sync_end, 2)})
            ptr = sync_end
 
        if t_optimizer > 0:
            opt_end = ptr + t_optimizer
            events.append({"state": "optimizer", "start": round(ptr, 2), "end": round(opt_end, 2)})
            ptr = opt_end
 
        events.sort(key=lambda x: x["start"])
        timeline.append({
            "gpu": f"GPU-{g+1}",
            "pp_stage": pp_stage,
            "static_weight_gb": metrics["static_weight_gb"],
            "peak_activation_gb": metrics["peak_activation_gb"],
            "gradient_gb": metrics["gradient_gb"],
            "events": events,
        })
 
    return timeline
 
@app.post("/simulate")
def simulate(data: SimulationInput):
    total_gpus_requested = data.dp_degree * data.pp_degree * data.tp_degree
    if total_gpus_requested <= 0:
        raise HTTPException(status_code=400, detail="Distributed core dimensions must evaluate to positive integers")
 
    run_metrics = execute_simulator_kernel(data)
    if run_metrics["is_oom"]:
        return run_metrics
 
    scaling_curve_logs = []
    for dp_step in [1, 2, 4, 8, 16, 32]:
        step_result = execute_simulator_kernel(data, override_dp_degree=dp_step)
        calculated_total_gpus = dp_step * data.pp_degree * data.tp_degree
        scaling_curve_logs.append({
            "gpus": calculated_total_gpus,
            "ideal": 100.0,
            "actual": round(step_result["efficiency"], 2) if not step_result["is_oom"] else 0.0,
        })
 
    timeline = build_1f1b_timeline(
        pp=data.pp_degree,
        total_gpus=total_gpus_requested,
        m=data.micro_batches,
        metrics=run_metrics,
        activation_checkpointing=data.activation_checkpointing,
    )
 
    diagnostics = []
    if run_metrics["bubble_fraction_pct"] > 25.0:
        diagnostics.append({
            "type": "Pipeline Bubble Stall",
            "impact": f"{run_metrics['bubble_fraction_pct']}% of runtime idling",
            "explanation": f"Your pipeline depth (PP={data.pp_degree}) creates stalls.",
            "remediation": "Increase micro-batches (m) or reduce PP stages."
        })
 
    if not diagnostics:
        diagnostics.append({
            "type": "Optimized Compute Geometry",
            "impact": f"Cluster efficiency: {run_metrics['efficiency']}%",
            "explanation": "Compute parameters are highly balanced.",
            "remediation": "Configuration is stable."
        })
 
    return {
        "is_oom": False,
        "max_capacity_allowed": run_metrics["max_capacity_allowed"],
        "total_memory_requested": run_metrics["total_memory_requested"],
        "total_time": run_metrics["total_time"],
        "efficiency": run_metrics["efficiency"],
        "bubble_fraction_pct": run_metrics["bubble_fraction_pct"],
        "communication_overhead": run_metrics["comm_overhead_pct"],
        "network_topology": {
            "type": data.interconnect,
            "bandwidth": NVLINK_SPEED_GBS if data.interconnect.upper() == "NVLINK" else PCIE_SPEED_GBS,
            "layout": data.topology_type,
        },
        "timeline": timeline,
        "real_scaling_data": scaling_curve_logs,
        "diagnostics": diagnostics,
    }