# ABOUTME: Loads a Gemma Scope SAE plus Neuronpedia densities/explanations and scores features with
# ABOUTME: the paper's TF-IDF metric (mean activation x log(1/density)) at selected token positions.
import json
import threading
from typing import Any, Dict, List, Optional

import torch
from sae_lens import SAE


class SAEBackend:
    """SAE readout used by the white-box tools, mirroring `elicitation_methods/sae.py`."""

    def __init__(
        self,
        release: str,
        sae_id: str,
        densities_file: str,
        explanations_file: Optional[str],
        device: str,
        tokenizer,
        embedding_matrix: Optional[torch.Tensor] = None,
    ):
        """Load the SAE and the Neuronpedia metadata needed for scoring and descriptions.

        Args:
            release: SAE-Lens release name (e.g. "gemma-scope-9b-pt-res-canonical").
            sae_id: SAE id inside the release (e.g. "layer_34/width_131k/canonical").
            densities_file: JSONL of Neuronpedia feature records with `index` and `frac_nonzero`.
            explanations_file: JSON mapping feature index -> description (Neuronpedia explanations).
            device: Device for the SAE weights.
            tokenizer: Tokenizer used to decode most-similar tokens.
            embedding_matrix: Token embedding matrix of the audited model (for most-similar tokens).
        """
        loaded = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
        self.sae = loaded[0] if isinstance(loaded, tuple) else loaded
        assert hasattr(self.sae, "encode"), f"unexpected SAE object: {type(self.sae)}"
        self.device = device
        self.tokenizer = tokenizer
        self.embedding_matrix = embedding_matrix
        self.n_features = self.sae.W_dec.shape[0]

        densities = {}
        with open(densities_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                densities[int(record["index"])] = float(record["frac_nonzero"])
        self.available_features = sorted(i for i in densities if i < self.n_features)
        assert self.available_features, f"no feature densities found in {densities_file}"
        self.density_tensor = torch.tensor(
            [densities[i] for i in self.available_features],
            dtype=torch.float32,
            device=device,
        )

        self.gpu_lock = threading.Lock()
        self.explanations: Dict[str, str] = {}
        if explanations_file:
            with open(explanations_file, "r", encoding="utf-8") as f:
                self.explanations = json.load(f)["explanations"]

    def describe(self, feature_index: int) -> str:
        """Return the Neuronpedia description of a feature, or a placeholder if missing."""
        return self.explanations.get(str(feature_index), "No description available")

    def _most_similar_tokens(self, feature_index: int, top_k_tokens: int) -> List[str]:
        assert self.embedding_matrix is not None, "embedding matrix required for SAE tokens"
        with self.gpu_lock, torch.no_grad():
            feature_vector = self.sae.W_dec[feature_index].float()
            similarities = feature_vector @ self.embedding_matrix.float().T
            top_ids = torch.argsort(similarities, descending=True)[:top_k_tokens]
        return [self.tokenizer.decode([int(i)]) for i in top_ids]

    def top_features(
        self, activations: torch.Tensor, top_k: int, top_k_tokens: int = 0
    ) -> List[Dict[str, Any]]:
        """Score features by TF-IDF over the given positions and return the top-k.

        Args:
            activations: [n_positions, d_model] residual stream activations.
            top_k: Number of features to return.
            top_k_tokens: If > 0, also return that many tokens most similar to each feature vector.
        """
        assert activations.dim() == 2, f"expected [n_pos, d_model], got {activations.shape}"
        with self.gpu_lock, torch.no_grad():
            sae_acts = self.sae.encode(activations.to(self.device).float())
        assert sae_acts.shape == (activations.shape[0], self.n_features), (
            f"unexpected SAE activation shape {sae_acts.shape}"
        )
        available = sae_acts[:, self.available_features]
        mean_acts = available.mean(dim=0)
        scores = mean_acts * torch.log(1 / (self.density_tensor + 1e-8))
        top_idx = torch.argsort(scores, descending=True)[:top_k]

        features = []
        for rank, idx in enumerate(top_idx.tolist(), start=1):
            feature_index = self.available_features[idx]
            entry = {
                "rank": rank,
                "feature_index": feature_index,
                "activation_value": float(mean_acts[idx]),
                "tfidf_score": float(scores[idx]),
                "description": self.describe(feature_index),
            }
            if top_k_tokens > 0:
                entry["most_similar_tokens"] = self._most_similar_tokens(
                    feature_index, top_k_tokens
                )
            features.append(entry)
        return features
