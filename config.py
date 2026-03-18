"""Model configurations for Nemotron family on Apple Silicon."""

from dataclasses import dataclass, field

MODELS = {
    # Nemotron-3 Nano — 30B total, 3B active (MoE), hybrid Mamba-Transformer
    # Best perf/memory ratio: 83 tok/s on M4 Pro 48GB
    "nano": {
        "hf_id": "lmstudio-community/NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-8bit",
        "name": "Nemotron-3 Nano 30B-A3B",
        "total_params": "30B",
        "active_params": "3B",
        "arch": "hybrid-mamba-transformer-moe",
        "quant": "8-bit",
        "est_memory_gb": 18.0,
        "fits_48gb": True,
    },
    "nano-4bit": {
        "hf_id": "mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit",
        "name": "Nemotron-3 Nano 30B-A3B (4-bit)",
        "total_params": "30B",
        "active_params": "3B",
        "arch": "hybrid-mamba-transformer-moe",
        "quant": "4-bit",
        "est_memory_gb": 10.0,
        "fits_48gb": True,
    },
    # Llama-3.3-Nemotron-Super-49B — 49B dense, Llama-3.3 backbone + NAS
    "super-49b": {
        "hf_id": "mlx-community/Llama-3_3-Nemotron-Super-49B-v1-mlx-4bit",
        "name": "Nemotron-Super 49B v1 (4-bit)",
        "total_params": "49B",
        "active_params": "49B",
        "arch": "llama-3.3-hybrid",
        "quant": "4-bit",
        "est_memory_gb": 27.0,
        "fits_48gb": True,
    },
    # Nemotron-3 Super 120B-A12B — 120B total, 12B active (MoE)
    # EXPERIMENTAL: requires mmap offloading, exceeds 48GB at any useful quant
    "super-120b": {
        "hf_id": "unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF",
        "name": "Nemotron-3 Super 120B-A12B",
        "total_params": "120B",
        "active_params": "12B",
        "arch": "hybrid-mamba-transformer-moe",
        "quant": "Q4_K_M (GGUF)",
        "est_memory_gb": 65.0,
        "fits_48gb": False,  # needs mmap/swap
    },
}

DEFAULT_MODEL = "nano"

# M4 Pro hardware profile
HARDWARE = {
    "chip": "Apple M4 Pro",
    "memory_gb": 48,
    "memory_bandwidth_gbs": 273,
    "gpu_cores": 16,
    "nvme_read_gbs": 7.4,
    "usable_memory_gb": 38.5,  # after OS + runtime overhead
}


@dataclass
class InferenceConfig:
    model_key: str = DEFAULT_MODEL
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    context_length: int = 8192
    use_mmap: bool = True  # LLM-in-a-Flash: memory-map weights
    prompt: str = "Explain the key innovations in Apple's M4 chip architecture."
    benchmark: bool = False
    benchmark_prompts: int = 5
