import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient, models
from tqdm import tqdm
from src.utils import batch_iterate, image_to_base64

# Initialize logger
logger = logging.getLogger(__name__)

class QdrantVectorStore:
    """Production-ready Qdrant vector store"""
    
    def __init__(self, 
                 collection_name: str,
                 vector_dim: int = 128,
                 batch_size: int = 4,
                 qdrant_url: str = "http://localhost:6334",
                 use_hpc: bool = False):  # Changed default to False
        self.collection_name = collection_name
        self.vector_dim = vector_dim  # Always 128 for Janus compatibility
        self.batch_size = batch_size
        self.qdrant_url = qdrant_url
        self.use_hpc = use_hpc
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Qdrant client"""
        try:
            self.client = QdrantClient(url=self.qdrant_url, prefer_grpc=True)
            logger.info(f"Connected to Qdrant at {self.qdrant_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def create_collection(self):
        """Create collection - always 128 dimensions for Janus compatibility"""
        try:
            if not self.client.collection_exists(self.collection_name):
                
                # Always use 128 dimensions regardless of HPC
                # HPC compression/decompression happens during processing, not storage
                vector_params = models.VectorParams(
                    size=128,  # Fixed 128 for Janus model compatibility
                    distance=models.Distance.COSINE,
                    on_disk=True,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                )

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vector_params,
                    optimizers_config=models.OptimizersConfigDiff(
                        default_segment_number=2,
                        max_segment_size=None,
                        memmap_threshold=None,
                        indexing_threshold=10000,
                        flush_interval_sec=2,
                        max_optimization_threads=2
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        m=32,
                        ef_construct=200,
                        full_scan_threshold=20000,
                        max_indexing_threads=2
                    ),
                    on_disk_payload=True
                )
                logger.info(f"Created collection: {self.collection_name} with 128 dimensions")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def ingest_data(self, embeddings: List[List[float]], images: List, metadata: List[Dict] = None, image_paths: List[str] = None):
        """Ingest embeddings with metadata and proper image path mapping"""
        try:
            total_points = 0
            
            # Validate embedding dimensions
            if embeddings and len(embeddings[0]) != 128:
                logger.warning(f"Expected 128-dim embeddings, got {len(embeddings[0])}-dim. Collection expects 128.")
            
            for i, batch_embeddings in tqdm(
                enumerate(batch_iterate(embeddings, self.batch_size)), 
                desc="Ingesting data"
            ):
                points = []
                
                for j, embedding in enumerate(batch_embeddings):
                    point_id = i * self.batch_size + j
                    
                    # Validate embedding is 128-dimensional
                    if len(embedding) != 128:
                        logger.error(f"Embedding {point_id} has {len(embedding)} dims, expected 128")
                        continue
                    
                    # Prepare payload with proper image path mapping
                    payload = {
                        "page_id": point_id,
                        "image_index": point_id  # Add explicit image index
                    }
                    
                    if image_paths and point_id < len(image_paths):
                        payload["image_path"] = image_paths[point_id]
                    
                    if images and point_id < len(images):
                        payload["image"] = image_to_base64(images[point_id])
                    
                    if metadata and point_id < len(metadata):
                        payload.update(metadata[point_id])
                    
                    # Create point - embedding should be List[float] with 128 elements
                    point = models.PointStruct(
                        id=point_id,
                        vector=embedding,  # Already a flat list of 128 floats
                        payload=payload
                    )
                    points.append(point)
                
                if points:  # Only upsert if we have valid points
                    # Upsert batch
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                        wait=True
                    )
                    total_points += len(points)
            
            logger.info(f"Successfully ingested {total_points} points to {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to ingest data: {e}")
            raise
    
    def search(self, 
               query_vector: List[float], 
               limit: int = 8,
               score_threshold: float = 0.0):
        """Search with optimized parameters"""
        try:
            # Validate query vector dimension
            if len(query_vector) != 128:
                logger.error(f"Query vector has {len(query_vector)} dims, expected 128")
                raise ValueError(f"Query vector must be 128-dimensional, got {len(query_vector)}")
            
            # Adjust search parameters based on HPC usage
            if self.use_hpc:
                search_params = models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        ignore=False,
                        rescore=True,
                        oversampling=3.0  # Higher oversampling for HPC
                    ),
                    hnsw_ef=256,  # Higher ef for better recall with compressed vectors
                    exact=False
                )
            else:
                search_params = models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        ignore=False,
                        rescore=True,
                        oversampling=2.0
                    ),
                    hnsw_ef=128,
                    exact=False
                )
            
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                search_params=search_params,
                with_payload=True,
                with_vectors=False
            )
            
            logger.debug(f"Found {len(results.points)} results with scores: {[p.score for p in results.points]}")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def collection_info(self):
        """Get collection information for debugging"""
        try:
            info = self.client.get_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name}: {info.vectors_count} vectors, config: {info.config}")
            return info
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None
    
    def delete_collection(self):
        """Delete collection"""
        try:
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(self.collection_name)
                logger.info(f"Deleted collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} does not exist")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise

# Compatibility wrapper
class QdrantVDB_QB(QdrantVectorStore):
    """Backward compatibility wrapper"""
    
    def __init__(self, collection_name: str, vector_dim: int = 128, batch_size: int = 4):
        super().__init__(collection_name, vector_dim, batch_size, use_hpc=False)
    
    def define_client(self):
        """Legacy method - client is auto-initialized"""
        pass
    
    def ingest_data(self, embeddata):
        """Legacy interface"""
        super().ingest_data(embeddata.embeddings, embeddata.images)