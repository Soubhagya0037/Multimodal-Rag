import torch
import gc
import time
import base64
import logging
from io import BytesIO
from typing import List, Any, Generator, Callable
from PIL import Image

logger = logging.getLogger(__name__)

def batch_iterate(lst: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string"""
    try:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to convert image to base64: {e}")
        raise

def clear_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def get_device_map() -> str:
    """Determines the appropriate device for model loading."""
    if torch.cuda.is_available():
        return "cuda"
    # You could add other accelerators like 'mps' for Mac here
    return "cpu"

def load_model_with_retry(model_class: Callable, model_name: str, max_retries: int = 3, **kwargs) -> Any:
    """Load model with retries and proper device handling"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Loading {model_class.__name__} (attempt {attempt + 1}/{max_retries})")

            # Prepare keyword arguments for model loading
            load_kwargs = kwargs.copy()

            # Set device_map for optimal loading
            device = get_device_map()
            load_kwargs['device_map'] = device

            if device == 'cuda':
                load_kwargs.setdefault('torch_dtype', torch.bfloat16)
                logger.info("CUDA available, using device_map='cuda' and bfloat16")
            else:
                # This branch handles the 'cpu' case from get_device_map()
                load_kwargs.setdefault('torch_dtype', torch.float32)
                logger.info("CUDA not available, loading on CPU with float32")
                # low_cpu_mem_usage can be problematic on CPU, safer to remove
                if 'low_cpu_mem_usage' in load_kwargs:
                    load_kwargs.pop('low_cpu_mem_usage')
                    logger.warning("Removed 'low_cpu_mem_usage' for CPU-only loading.")

            # Load the model using from_pretrained with robust device handling
            model = model_class.from_pretrained(model_name, **load_kwargs)
            
            # Set to eval mode for inference
            model.eval()
            
            final_device = next(model.parameters()).device
            logger.info(f"Successfully loaded model on {final_device}")
            return model
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logger.error(f"All {max_retries} attempts failed")
                raise
            time.sleep(2 ** attempt)

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('multimodal_rag.log'),
            logging.StreamHandler()
        ]
    )