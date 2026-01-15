from dataclasses import dataclass

@dataclass
class Config:
    hf_repo: str = "unsloth/Qwen3-8B-GGUF"
    hf_file: str = "*Q4_K_M.gguf"
    hf_tokenizer_override: str = "Qwen/Qwen3-8B-FP8"
    
    llm_max_tokens_gen: int = -1
    llm_n_gpu_layers: int = -1
    llm_n_ctx: int = 40960
    llm_flash_attn: bool = True
    
    llm_temperature: float = 0.6
    llm_top_p: float = 0.95
    llm_top_k: int = 40
    