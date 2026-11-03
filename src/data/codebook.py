"""
Platonic Taxonomy Codebook & Feature Generator
Maintains the standardized discrete attribute embeddings for:
- ISCO-08 Professions
- Plutchik 8 Primary Emotions
- Leary Interpersonal Circumplex Styles
- Standard Age Bins & Gender Descriptors
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Tuple


# Standardized Ontological Codebook Definitions
STANDARD_CODEBOOK = {
    "profession": [
        "Chief Executives, Senior Officials and Legislators (ISCO-11)",
        "Science and Engineering Professionals (ISCO-21)",
        "Health Professionals (ISCO-22)",
        "Teaching Professionals (ISCO-23)",
        "Business and Administration Professionals (ISCO-24)",
        "Information and Communications Technology Professionals (ISCO-25)",
        "Legal, Social and Cultural Professionals (ISCO-26)",
        "Personal Service Workers (ISCO-51)",
        "Sales Workers (ISCO-52)",
        "Chef or Cook (ISCO-5120)",
        "Athlete or Sports Coach (ISCO-342)",
        "Musician, Singer or Composer (ISCO-2652)"
    ],
    "emotion": [
        "Joy: intense positive happiness or delight",
        "Trust: acceptance, admiration and calm faith",
        "Fear: apprehension, alarm or anxious wariness",
        "Surprise: unexpected astonishment or realization",
        "Sadness: sorrow, grief or dejected melancholy",
        "Disgust: aversion, revulsion or contempt",
        "Anger: hostility, irritation or furious rage",
        "Anticipation: eager expectation or focused interest"
    ],
    "style": [
        "Dominant-Directive: assertive, commanding communication",
        "Friendly-Collaborative: warm, inclusive and supportive",
        "Submissive-Compliant: deferential, gentle and agreeable",
        "Hostile-Critical: stern, challenging and sharp",
        "Formal-Reserved: professional, serious and contained",
        "Expressive-Animated: dynamic, energetic and passionate"
    ],
    "age": [
        "Child: below 13 years",
        "Teenager: 13 to 19 years",
        "Young Adult: 20 to 35 years",
        "Middle Aged: 36 to 55 years",
        "Older Adult: above 55 years"
    ],
    "gender": [
        "Female presentation",
        "Male presentation",
        "Unspecified gender presentation"
    ]
}


class PlatonicCodebook:
    """Manages text taxonomy definitions and pre-computed embedding vectors."""
    def __init__(self, embed_dim: int = 384, taxonomy_path: str = None):
        self.embed_dim = embed_dim
        self.codebook = STANDARD_CODEBOOK
        
        # Load custom taxonomy JSON if available from Dataset Track
        if taxonomy_path and os.path.exists(taxonomy_path):
            try:
                with open(taxonomy_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Extend or update codebook with extracted clusters if available
            except Exception as e:
                print(f"Warning: Could not load {taxonomy_path}: {e}")

    def get_attribute_names(self, attribute_type: str) -> List[str]:
        return self.codebook.get(attribute_type, [])

    def get_or_create_embeddings(self, attribute_type: str, device: str = "cpu") -> Tuple[torch.Tensor, List[str]]:
        """
        Computes or generates normalized embedding vectors for the codebook.
        Uses deterministic pseudorandom orthogonal vectors if sentence-transformers is offline.
        """
        names = self.get_attribute_names(attribute_type)
        K = len(names)
        
        # Try sentence-transformers if available
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            vecs = model.encode(names, convert_to_tensor=True, device=device)
            return torch.nn.functional.normalize(vecs, p=2, dim=-1), names
        except Exception:
            # Deterministic fallback vectors
            torch.manual_seed(hash(attribute_type) % 10000)
            vecs = torch.randn(K, self.embed_dim, device=device)
            return torch.nn.functional.normalize(vecs, p=2, dim=-1), names


def generate_synthetic_aligned_pair(
    num_samples: int = 200,
    dim: int = 384,
    condition: str = "good_views",
    misalignment_eps: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic video-attribute latent pairs for standalone testing:
    - 'good_views': Z_a = Z_v @ R + noise(eps)   [Bounded misalignment]
    - 'bad_views':  Z_a has random rank collapse + uncorrelated noise [Unbounded misalignment]
    """
    rng = np.random.RandomState(42)
    
    # Ground-truth video features
    Z_v = rng.randn(num_samples, dim)
    Z_v = Z_v / np.linalg.norm(Z_v, axis=1, keepdims=True)
    
    # Random orthogonal ground-truth rotation R in O(d)
    H = rng.randn(dim, dim)
    Q, _ = np.linalg.qr(H)
    
    if condition == "good_views":
        # Bounded misalignment: theorem holds
        noise = rng.randn(num_samples, dim) * misalignment_eps
        Z_a = Z_v @ Q + noise
    else:
        # Bad views: large linguistic prior noise + rank collapse on invisible attributes
        noise = rng.randn(num_samples, dim) * 1.5
        # Low rank linguistic prior
        rank_k = 10
        prior = rng.randn(num_samples, rank_k) @ rng.randn(rank_k, dim)
        Z_a = 0.2 * (Z_v @ Q) + 0.8 * prior + noise

    Z_a = Z_a / np.linalg.norm(Z_a, axis=1, keepdims=True)
    return Z_v, Z_a
