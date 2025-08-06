import torch
import logging
import time
import gc
from typing import List, Union
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor
from tqdm import tqdm
from src.utils import batch_iterate, clear_memory, load_model_with_retry
from src.hpc_colpali import HPC
import numpy as np

logger = logging.getLogger(__name__)

class ColPaliEmbedder:
    """Optimized ColPali embedder with L4 GPU optimizations"""
    
    def __init__(self, 
                 model_name: str = "vidore/colpali-v1.2",
                 batch_size: int = 8,
                 use_hpc: bool = False,
                 use_flash_attention: bool = True,
                 hpc_kwargs: dict = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_flash_attention = use_flash_attention
        self.model = None
        self.processor = None
        self.use_hpc = use_hpc
        self.codebook_built = False  # Track if HPC codebook is ready
        
        # Initialize HPC if requested
        if self.use_hpc:
            self.hpc = HPC(**(hpc_kwargs or {}))
            logger.info(f"HPC initialized with k={self.hpc.k}, prune_ratio={1-self.hpc.p}")
        else:
            self.hpc = None
            logger.info("HPC disabled - using mean pooling")
            
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model with proper device handling"""
        try:
            torch.set_float32_matmul_precision('high')
            
            logger.info(f"Loading ColPali model: {self.model_name}")
            
            # Load processor first
            self.processor = ColPaliProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            logger.info("Processor loaded successfully") 

            # Load model with retry mechanism
            model_kwargs = {
                "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                "trust_remote_code": True,
                "low_cpu_mem_usage": False
            }
            
            self.model = load_model_with_retry(
                ColPali,
                self.model_name,
                **model_kwargs
            )

            # patch for vision model's position_ids buffer
            if torch.cuda.is_available() and hasattr(self.model, 'vision_model'):
                self.model.vision_model.embeddings.position_ids = self.model.vision_model.embeddings.position_ids.to(torch.long)
            
            logger.info("Model loaded and ready for inference")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def build_hpc_codebook(self, sample_embeddings: np.ndarray):
        """Builds the HPC codebook from a sample of embeddings."""
        if not self.use_hpc or self.hpc is None:
            logger.warning("HPC is not enabled, skipping codebook build.")
            return
        
        try:
            logger.info(f"Building HPC codebook with {sample_embeddings.shape[0]} patches...")
            
            # Ensure 2D shape [N_patches, dim]
            if len(sample_embeddings.shape) > 2:
                sample_embeddings = sample_embeddings.reshape(-1, sample_embeddings.shape[-1])
            
            self.hpc.fit_codebook(sample_embeddings.astype(np.float32))
            self.codebook_built = True
            logger.info(f"HPC codebook built successfully with {self.hpc.k} centroids")
            
        except Exception as e:
            logger.error(f"Failed to build HPC codebook: {e}")
            self.use_hpc = False
            self.codebook_built = False

    @torch.inference_mode()
    def embed_images(self, images: List[Image.Image]) -> List[List[float]]:
        """Generate image embeddings - returns flat vectors suitable for Qdrant"""
        if not self.model or not self.processor:
            raise ValueError("Model not initialized")
        
        if self.use_hpc and not self.codebook_built:
            logger.error("HPC enabled but codebook not built!")
            raise ValueError("HPC enabled but codebook not built. Call build_hpc_codebook() first.")
        
        all_embeddings = []
        
        try:
            for batch in tqdm(batch_iterate(images, self.batch_size), desc="Embedding images"):
                inputs = self.processor.process_images(batch)
                
                device = next(self.model.parameters()).device
                dtype = next(self.model.parameters()).dtype
                
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        if value.is_floating_point():
                            inputs[key] = value.to(device=device, dtype=dtype)
                        else:
                            inputs[key] = value.to(device=device)
                
                if torch.cuda.is_available():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        embeddings = self.model(**inputs)
                else:
                    embeddings = self.model(**inputs)
                
                # Process each image embedding
                for emb in embeddings.cpu().float().numpy():
                    # emb shape: [num_patches, 128]
                    
                    if self.use_hpc and self.codebook_built:
                        # Compress patches to codes
                        codes = self.hpc.compress(emb)  # uint8 codes [num_patches]
                        
                        # For Qdrant storage, we need a consistent vector size
                        # Pad or truncate to fixed size
                        fixed_size = 128  # Fixed vector size for HPC
                        if len(codes) < fixed_size:
                            # Pad with zeros
                            padded_codes = np.zeros(fixed_size, dtype=np.uint8)
                            padded_codes[:len(codes)] = codes
                            final_vector = padded_codes.astype(float).tolist()
                        else:
                            # Truncate
                            final_vector = codes[:fixed_size].astype(float).tolist()
                        
                        all_embeddings.append(final_vector)
                        
                    else:
                        # Mean pooling for non-HPC: [128]
                        mean_emb = emb.mean(axis=0)  # [128]
                        all_embeddings.append(mean_emb.tolist())
                
                clear_memory()
                
        except Exception as e:
            logger.error(f"Error embedding images: {e}")
            raise
        
        logger.info(f"Generated {len(all_embeddings)} image embeddings")
        logger.info(f"HPC status: {self.use_hpc}, Codebook built: {self.codebook_built}")
        logger.info(f"Vector dimensions: {len(all_embeddings[0]) if all_embeddings else 'None'}")
        
        return all_embeddings
    
    @torch.inference_mode()
    def embed_query(self, query: str) -> List[float]:
        """Generate query embedding - returns flat vector suitable for Qdrant"""
        if not self.model or not self.processor:
            raise ValueError("Model not initialized")
        
        if self.use_hpc and not self.codebook_built:
            logger.error("HPC enabled but codebook not built!")
            raise ValueError("HPC enabled but codebook not built.")
        
        try:
            inputs = self.processor.process_queries([query])
            
            device = next(self.model.parameters()).device
            dtype = next(self.model.parameters()).dtype
            
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if value.is_floating_point():
                        inputs[key] = value.to(device=device, dtype=dtype)
                    else:
                        inputs[key] = value.to(device=device)
            
            if torch.cuda.is_available():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    embedding_patches = self.model(**inputs)
            else:
                embedding_patches = self.model(**inputs)
            
            emb = embedding_patches[0].cpu().float().numpy()  # [num_patches, 128]

            if self.use_hpc and self.codebook_built:
                # Compress to codes
                codes = self.hpc.compress(emb)  # uint8 codes [num_patches]
                
                # Match the same fixed size as images
                fixed_size = 128
                if len(codes) < fixed_size:
                    padded_codes = np.zeros(fixed_size, dtype=np.uint8)
                    padded_codes[:len(codes)] = codes
                    embedding = padded_codes.astype(float).tolist()
                else:
                    embedding = codes[:fixed_size].astype(float).tolist()
            else:
                # Mean pooling for non-HPC
                embedding = emb.mean(axis=0).tolist()  # [128]

            clear_memory()
            
            logger.debug(f"Generated query embedding, HPC: {self.use_hpc}, dim: {len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise

    @torch.inference_mode()
    def get_raw_patch_embeddings_for_codebook(self, images: List[Image.Image]) -> np.ndarray:
        """
        Generates raw, uncompressed patch embeddings for a sample of images.
        This is used exclusively to gather data for training the HPC codebook.
        """
        if not self.model or not self.processor:
            raise ValueError("Model not initialized")
        
        all_patch_embs = []
        desc = "Generating raw embeddings for codebook"
        try:
            for batch in tqdm(batch_iterate(images, self.batch_size), desc=desc):
                # process batch
                inputs = self.processor.process_images(batch)
                
                # move inputs to model device with correct dtype
                device = next(self.model.parameters()).device
                dtype = next(self.model.parameters()).dtype
                
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        if value.is_floating_point():
                            inputs[key] = value.to(device=device, dtype=dtype)
                        else:
                            inputs[key] = value.to(device=device)
                
                # forward pass
                if torch.cuda.is_available():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        embeddings = self.model(**inputs)
                else:
                    embeddings = self.model(**inputs)
                
                # collect all patch embeddings from the batch
                for emb in embeddings.cpu().float().numpy():
                    all_patch_embs.append(emb)
            
            clear_memory()

        except Exception as e:
            logger.error(f"Error generating raw embeddings for codebook: {e}")
            raise

        if not all_patch_embs:
            return np.array([])
            
        # concatenate all patch embeddings into a single [N, D] numpy array
        return np.vstack(all_patch_embs)

class EmbedData:
    """Compatibility wrapper for existing code"""
    
    def __init__(self, embed_model_name: str = "vidore/colpali-v1.2", batch_size: int = 8, hpc_kwargs: dict = None):
        self.embedder = ColPaliEmbedder(embed_model_name, batch_size, use_flash_attention=True, hpc_kwargs=hpc_kwargs)
        self.embeddings = []
        self.images = []
    
    def embed(self, images: List[Image.Image]):
        """Embed images and store results"""
        self.images = images
        self.embeddings = self.embedder.embed_images(images)
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding"""
        return self.embedder.embed_query(query)