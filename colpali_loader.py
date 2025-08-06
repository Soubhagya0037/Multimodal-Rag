import os
import time
import torch
import gc
from transformers import AutoModel, AutoProcessor
from colpali_engine import ColPaliProcessor, ColPali
import logging
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ColPaliLoader:
    def __init__(self, 
                 model_name: str = "vidore/colpali-v1.2", 
                 device: str = "cuda",
                 use_flash_attention: bool = True,
                 use_mixed_precision: bool = True):
        self.model_name = model_name
        self.device = device
        self.use_flash_attention = use_flash_attention
        self.use_mixed_precision = use_mixed_precision
        self.model = None
        self.processor = None
        
    def clear_memory(self):
        """Clear GPU memory aggressively"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
    def load_with_retry(self, max_retries: int = 3, base_delay: int = 5) -> Tuple[ColPali, ColPaliProcessor]:
        """Load model with exponential backoff retry logic"""
        
        # Set environment variables to handle rate limiting
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Loading ColPali model (attempt {attempt}/{max_retries})")
                
                # Clear memory before each attempt
                self.clear_memory()
                
                # Configure model loading arguments
                model_kwargs = {
                    "device_map": "auto",
                    "torch_dtype": torch.bfloat16 if self.use_mixed_precision else torch.float32,
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": True,
                }
                
                # Add flash attention if available
                if self.use_flash_attention:
                    try:
                        model_kwargs["attn_implementation"] = "flash_attention_2"
                    except Exception:
                        logger.warning("Flash attention not available, using default attention")
                
                # Load processor first (lighter weight)
                logger.info("Loading processor...")
                self.processor = ColPaliProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    local_files_only=False
                )
                
                # Load model with memory optimization
                logger.info("Loading model...")
                self.model = ColPali.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                
                # Move to eval mode and optimize
                self.model.eval()
                
                # Compile model for better performance (PyTorch 2.0+)
                if hasattr(torch, 'compile'):
                    try:
                        logger.info("Compiling model for optimization...")
                        self.model = torch.compile(self.model, mode="reduce-overhead")
                    except Exception as e:
                        logger.warning(f"Model compilation failed: {e}")
                
                logger.info("Model loaded successfully!")
                return self.model, self.processor
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Attempt {attempt} failed: {error_msg}")
                
                # Handle specific error types
                if "429" in error_msg or "rate" in error_msg.lower():
                    delay = base_delay * (2 ** (attempt - 1))  # Exponential backoff
                    logger.info(f"Rate limited. Waiting {delay}s before retry...")
                    time.sleep(delay)
                elif "CUDA out of memory" in error_msg:
                    logger.error("GPU OOM error. Try reducing batch size or using CPU offloading")
                    self.clear_memory()
                    time.sleep(2)
                elif "does not appear to have" in error_msg:
                    logger.info("Model files not found, clearing cache and retrying...")
                    self._clear_hf_cache()
                    time.sleep(1)
                else:
                    logger.error(f"Unexpected error: {error_msg}")
                    time.sleep(base_delay)
                
                if attempt == max_retries:
                    raise Exception(f"Failed to load model after {max_retries} attempts. Last error: {error_msg}")
    
    def _clear_hf_cache(self):
        """Clear HuggingFace cache for the model"""
        try:
            from huggingface_hub import scan_cache_dir
            cache_info = scan_cache_dir()
            for repo in cache_info.repos:
                if self.model_name in repo.repo_id:
                    logger.info(f"Clearing cache for {repo.repo_id}")
                    repo.delete_revisions().reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Could not clear HF cache: {e}")

# Usage example
if __name__ == "__main__":
    loader = ColPaliLoader(
        use_flash_attention=True,
        use_mixed_precision=True
    )
    
    try:
        model, processor = loader.load_with_retry(max_retries=5)
        print("Model loaded successfully!")
        
        # Test the model
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        
    except Exception as e:
        print(f"Failed to load model: {e}")