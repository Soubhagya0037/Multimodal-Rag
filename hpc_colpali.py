import numpy as np
import torch
from sklearn.cluster import KMeans
from typing import List

class HPC:
    def __init__(self, k_centroids: int = 256, prune_ratio: float = 0.6):
        self.k = k_centroids
        self.p = 1 - prune_ratio          # keep top-p %
        self.km = None
        self.centroids_ = None            # float32 centroids

    # ---------- 1.  OFF-LINE: learn codebook ----------
    def fit_codebook(self, embeddings: np.ndarray):
        """embeddings: shape [N_patches, dim]"""
        self.km = KMeans(n_clusters=self.k, n_init=1, random_state=42, max_iter=10)
        self.km.fit(embeddings.astype(np.float32))
        self.centroids_ = self.km.cluster_centers_.astype(np.float32)

    # ---------- 2.  COMPRESS ----------
    def compress(self, patch_embs: np.ndarray) -> np.ndarray:
        """Return uint8 codes for each patch."""
        codes = self.km.predict(patch_embs.astype(np.float32)).astype(np.uint8)
        return codes

    # ---------- 3.  PRUNE via attention ----------
    @torch.inference_mode()
    def prune(self, patch_embs: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        """
        patch_embs: [L, dim]  (float32)
        attn      : [L]        (attention scores from VLM)
        returns   : pruned [L', dim]
        """
        k_keep = max(1, int(attn.size(0) * self.p))
        _, top_idx = torch.topk(attn, k_keep, largest=True, sorted=False)
        return patch_embs[top_idx]

    # ---------- 4.  DECOMPRESS ----------
    def decompress(self, codes: np.ndarray) -> np.ndarray:
        """codes: [N] uint8  -> float32 centroids"""
        return self.centroids_[codes]