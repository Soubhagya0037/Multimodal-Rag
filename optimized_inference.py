import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import List, Union
import logging
from colpali_loader import ColPaliLoader

logger = logging.getLogger(__name__)

class OptimizedColPaliInference:
    def __init__(self, model_name: str = "vidore/colpali-v1.2"):
        self.loader = ColPaliLoader(model_name)
        self.model = None
        self.processor = None
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
    def initialize(self):
        """Initialize model and processor"""
        self.model, self.processor = self.loader.load_with_retry()
        
    @torch.no_grad()
    def encode_images(self, 
                     images: List[Image.Image], 
                     batch_size: int = 1,
                     use_amp: bool = True) -> torch.Tensor:
        """Encode images with memory optimization"""
        
        if not self.model or not self.processor:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        all_embeddings = []
        
        # Process in batches to manage memory
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Preprocess images
            batch_inputs = self.processor.process_images(batch_images)
            
            # Move to device
            if torch.cuda.is_available():
                batch_inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                              for k, v in batch_inputs.items()}
            
            # Forward pass with automatic mixed precision
            if use_amp and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    embeddings = self.model(**batch_inputs)
            else:
                embeddings = self.model(**batch_inputs)
            
            # Move to CPU to save GPU memory
            embeddings = embeddings.cpu()
            all_embeddings.append(embeddings)
            
            # Clear GPU cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return torch.cat(all_embeddings, dim=0)
    
    @torch.no_grad()
    def encode_queries(self, 
                      queries: List[str], 
                      batch_size: int = 4,
                      use_amp: bool = True) -> torch.Tensor:
        """Encode text queries with optimization"""
        
        if not self.model or not self.processor:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            
            # Preprocess queries
            batch_inputs = self.processor.process_queries(batch_queries)
            
            # Move to device
            if torch.cuda.is_available():
                batch_inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                              for k, v in batch_inputs.items()}
            
            # Forward pass with AMP
            if use_amp and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    embeddings = self.model(**batch_inputs)
            else:
                embeddings = self.model(**batch_inputs)
            
            embeddings = embeddings.cpu()
            all_embeddings.append(embeddings)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return torch.cat(all_embeddings, dim=0)
    
    def compute_similarity(self, 
                          image_embeddings: torch.Tensor, 
                          query_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute similarity scores efficiently"""
        
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.mm(query_embeddings, image_embeddings.T)
        
        return similarity

# Example usage
if __name__ == "__main__":
    # Initialize inference engine
    inference = OptimizedColPaliInference()
    
    try:
        print("Initializing model...")
        inference.initialize()
        print("Model initialized successfully!")
        
        # Example with dummy data
        from PIL import Image
        import numpy as np
        
        # Create dummy images
        dummy_images = [
            Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            for _ in range(3)
        ]
        
        queries = ["Find documents about AI", "Show me charts and graphs"]
        
        print("Encoding images...")
        image_embeddings = inference.encode_images(dummy_images, batch_size=1)
        
        print("Encoding queries...")
        query_embeddings = inference.encode_queries(queries, batch_size=2)
        
        print("Computing similarities...")
        similarities = inference.compute_similarity(image_embeddings, query_embeddings)
        
        print(f"Similarity matrix shape: {similarities.shape}")
        print(f"Similarities:\n{similarities}")
        
    except Exception as e:
        print(f"Error during inference: {e}")