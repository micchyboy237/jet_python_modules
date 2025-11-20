#!/usr/bin/env python3
"""
Multimodal Processors - Cross-Modal Processing Components
========================================================

Production-ready multimodal processing implementations.
Minimal code, maximal signal ratio.

Usage:
    from multimodal_processors import TextEncoder, ImageEncoder, CrossModalFusion
    
    text_encoder = TextEncoder(d_model=512)
    image_encoder = ImageEncoder(d_model=512)
    fusion = CrossModalFusion(d_model=512, num_modalities=2)
    
    unified_embedding = fusion.fuse([text_features, image_features])
"""

import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

__all__ = [
    'Modality', 'ModalityEncoder', 'TextEncoder', 'ImageEncoder', 'AudioEncoder',
    'CrossModalFusion', 'MultimodalProcessor', 'ModalityAlignment', 'MultimodalRAG'
]

# ============================================================================
# OUTPUT & LOGGING SETUP
# ============================================================================

from jet.logger import CustomLogger
import os
import shutil

BASE_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)
shutil.rmtree(BASE_OUTPUT_DIR, ignore_errors=True)
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

main_logger = CustomLogger(
    name="long_context_evaluation",
    filename=os.path.join(BASE_OUTPUT_DIR, "main.log"),
    console_level="INFO",
    level="DEBUG",
    overwrite=True
)
main_logger.info("=" * 80)
main_logger.info("LONG CONTEXT EVALUATION LAB STARTED")
main_logger.info("=" * 80)


def create_example_dir(example_name: str) -> str:
    example_dir = os.path.join(BASE_OUTPUT_DIR, example_name)
    os.makedirs(example_dir, exist_ok=True)
    return example_dir


def get_example_logger(example_name: str, example_dir: str) -> CustomLogger:
    log_file = os.path.join(example_dir, "run.log")
    log = CustomLogger(
        name=example_name,
        filename=log_file,
        console_level="INFO",
        level="DEBUG",
        fmt="%(asctime)s | %(message)s",
        overwrite=True
    )
    log.info("")
    log.info("=" * 80)
    log.info(f"EXAMPLE: {example_name}")
    log.info("=" * 80)
    return log

import json
import matplotlib.pyplot as plt
from pathlib import Path

# Helper utilities used by every example
def save_json(data: Any, path: str):
    Path(path).write_text(json.dumps(data, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))

def save_npy(arr: np.ndarray, path: str):
    np.save(path, arr)

def plot_similarity_matrix(matrix: np.ndarray, labels: List[str], path: str):
    plt.figure(figsize=(6, 5))
    im = plt.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    plt.title("Modality Similarity Matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                     color="white" if matrix[i, j] < 0.5 else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ============================================================================
# CORE INTERFACES & DATA STRUCTURES
# ============================================================================

class Modality(Enum):
    """Supported modality types."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"

@dataclass
class ModalityData:
    """Container for modality-specific data."""
    modality: Modality
    data: np.ndarray
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ModalityEncoder(ABC):
    """Base interface for modality encoders."""
    
    def __init__(self, d_model: int = 512):
        self.d_model = d_model
    
    @abstractmethod
    def encode(self, data: Any) -> np.ndarray:
        """Encode raw data into unified embedding space."""
        pass
    
    @property
    def output_dim(self) -> int:
        return self.d_model

# ============================================================================
# MODALITY ENCODERS
# ============================================================================

class TextEncoder(ModalityEncoder):
    """Production-ready text encoder with self-attention."""
    
    def __init__(self, d_model: int = 512, vocab_size: int = 32000, max_seq_len: int = 512):
        super().__init__(d_model)
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Embedding layers
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_embedding = self._create_positional_encoding()
        
        # Self-attention parameters
        self.num_heads = 8
        self.d_k = d_model // self.num_heads
        self.W_qkv = np.random.randn(d_model, 3 * d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02
        
        # Feed-forward network
        self.W_ff1 = np.random.randn(d_model, d_model * 4) * 0.02
        self.W_ff2 = np.random.randn(d_model * 4, d_model) * 0.02
    
    def encode(self, tokens: np.ndarray) -> np.ndarray:
        """Encode token sequence to unified embedding."""
        seq_len = min(len(tokens), self.max_seq_len)
        tokens = tokens[:seq_len].astype(int) % self.vocab_size
        
        # Embeddings
        token_embeds = self.token_embedding[tokens]
        pos_embeds = self.pos_embedding[:seq_len]
        x = token_embeds + pos_embeds
        
        # Self-attention + feed-forward
        x = self._self_attention(x)
        x = self._feed_forward(x)
        
        # Global pooling
        return np.mean(x, axis=0)
    
    def _create_positional_encoding(self) -> np.ndarray:
        """Sinusoidal positional encodings."""
        pos_enc = np.zeros((self.max_seq_len, self.d_model))
        position = np.arange(self.max_seq_len)[:, None]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pos_enc[:, 0::2] = np.sin(position * div_term)
        pos_enc[:, 1::2] = np.cos(position * div_term)
        return pos_enc
    
    def _self_attention(self, x: np.ndarray) -> np.ndarray:
        """Multi-head self-attention."""
        seq_len = x.shape[0]
        
        # Project to Q, K, V
        qkv = x @ self.W_qkv
        qkv = qkv.reshape(seq_len, 3, self.num_heads, self.d_k)
        q, k, v = qkv.transpose(1, 2, 0, 3)  # (3, num_heads, seq_len, d_k)
        
        # Scaled attention
        scores = np.matmul(q, k.transpose(0, 2, 1)) / math.sqrt(self.d_k)
        attn_weights = self._softmax(scores)
        attended = np.matmul(attn_weights, v)
        
        # Output projection
        attended = attended.transpose(1, 0, 2).reshape(seq_len, self.d_model)
        return x + (attended @ self.W_o)  # Residual connection
    
    def _feed_forward(self, x: np.ndarray) -> np.ndarray:
        """Feed-forward network with residual connection."""
        ff_out = np.maximum(0, x @ self.W_ff1) @ self.W_ff2  # ReLU
        return x + ff_out
    
    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        max_vals = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - max_vals)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class ImageEncoder(ModalityEncoder):
    """Vision transformer-style image encoder."""
    
    def __init__(self, d_model: int = 512, patch_size: int = 16, image_size: int = 224):
        super().__init__(d_model)
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        patch_dim = 3 * patch_size * patch_size  # RGB patches
        self.patch_projection = np.random.randn(patch_dim, d_model) * 0.02
        self.pos_embedding = np.random.randn(self.num_patches + 1, d_model) * 0.02
        self.cls_token = np.random.randn(1, d_model) * 0.02
        
        # Attention parameters
        self.num_heads = 8
        self.d_k = d_model // self.num_heads
        self.W_qkv = np.random.randn(d_model, 3 * d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02
    
    def encode(self, image: np.ndarray) -> np.ndarray:
        """Encode image to unified embedding."""
        # Handle different input formats
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Standard RGB image
            patches = self._extract_patches(image)
        elif len(image.shape) == 2:
            # Pre-computed features
            patches = image[:self.num_patches] if len(image) >= self.num_patches else image
            if patches.shape[1] != self.d_model:
                # Project to correct dimension
                if patches.shape[1] > self.d_model:
                    patches = patches[:, :self.d_model]
                else:
                    pad_size = self.d_model - patches.shape[1]
                    patches = np.pad(patches, ((0, 0), (0, pad_size)), 'constant')
        else:
            # Flatten and use as features
            flattened = image.flatten()
            feature_len = min(len(flattened), self.num_patches * self.d_model)
            patches = flattened[:feature_len].reshape(-1, self.d_model)
            if patches.shape[0] < self.num_patches:
                pad_patches = self.num_patches - patches.shape[0]
                patches = np.pad(patches, ((0, pad_patches), (0, 0)), 'constant')
        
        # Add CLS token and positional embeddings
        cls_tokens = np.repeat(self.cls_token, 1, axis=0)
        x = np.concatenate([cls_tokens, patches], axis=0)
        
        # Add positional embeddings
        seq_len = min(x.shape[0], self.pos_embedding.shape[0])
        x = x[:seq_len] + self.pos_embedding[:seq_len]
        
        # Self-attention
        x = self._image_attention(x)
        
        # Return CLS token representation
        return x[0]
    
    def _extract_patches(self, image: np.ndarray) -> np.ndarray:
        """Extract patches from image."""
        height, width = image.shape[:2]
        
        # Simple patch extraction
        patches = []
        for i in range(0, height, self.patch_size):
            for j in range(0, width, self.patch_size):
                patch = image[i:i+self.patch_size, j:j+self.patch_size]
                if patch.shape[0] == self.patch_size and patch.shape[1] == self.patch_size:
                    patch_flat = patch.flatten()
                    # Project to embedding dimension
                    if len(patch_flat) <= self.patch_projection.shape[0]:
                        projected = np.zeros(self.d_model)
                        projected[:len(patch_flat)] = patch_flat @ self.patch_projection[:len(patch_flat)]
                    else:
                        projected = patch_flat[:self.patch_projection.shape[0]] @ self.patch_projection
                    patches.append(projected)
        
        # Ensure we have the right number of patches
        patches = patches[:self.num_patches]
        while len(patches) < self.num_patches:
            patches.append(np.zeros(self.d_model))
        
        return np.array(patches)
    
    def _image_attention(self, x: np.ndarray) -> np.ndarray:
        """Self-attention for image patches."""
        seq_len = x.shape[0]
        
        # Multi-head attention
        qkv = x @ self.W_qkv
        qkv = qkv.reshape(seq_len, 3, self.num_heads, self.d_k)
        q, k, v = qkv.transpose(1, 2, 0, 3)
        
        scores = np.matmul(q, k.transpose(0, 2, 1)) / math.sqrt(self.d_k)
        attn_weights = TextEncoder._softmax(scores)
        attended = np.matmul(attn_weights, v)
        
        attended = attended.transpose(1, 0, 2).reshape(seq_len, self.d_model)
        return x + (attended @ self.W_o)

class AudioEncoder(ModalityEncoder):
    """Spectral audio encoder with temporal modeling."""
    
    def __init__(self, d_model: int = 512, sample_rate: int = 16000, n_fft: int = 512):
        super().__init__(d_model)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = n_fft // 4
        
        # Mel filterbank
        self.n_mels = 80
        self.mel_filters = self._create_mel_filterbank()
        
        # Temporal modeling
        self.temporal_conv = np.random.randn(self.n_mels, d_model) * 0.02
        self.temporal_attention = np.random.randn(d_model, 1) * 0.02
        
        # Output projection
        self.output_proj = np.random.randn(d_model, d_model) * 0.02
    
    def encode(self, audio: np.ndarray) -> np.ndarray:
        """Encode audio to unified embedding."""
        # Handle different input formats
        if len(audio.shape) == 1:
            # Raw waveform
            mel_spec = self._compute_mel_spectrogram(audio)
        elif len(audio.shape) == 2:
            # Pre-computed spectral features
            mel_spec = audio
        else:
            # Flatten and process as waveform
            audio_flat = audio.flatten()
            mel_spec = self._compute_mel_spectrogram(audio_flat)
        
        # Temporal modeling
        temporal_features = self._temporal_modeling(mel_spec)
        
        # Temporal attention pooling
        attended = self._attention_pooling(temporal_features)
        
        return attended @ self.output_proj
    
    def _compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel-scale spectrogram."""
        # Ensure minimum length
        if len(audio) < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - len(audio)))
        
        # Simple STFT simulation
        num_frames = max(1, (len(audio) - self.n_fft) // self.hop_length + 1)
        stft_matrix = np.zeros((self.n_fft // 2 + 1, num_frames))
        
        # Hanning window
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(self.n_fft) / (self.n_fft - 1)))
        
        for i in range(num_frames):
            start = i * self.hop_length
            end = min(start + self.n_fft, len(audio))
            frame = np.zeros(self.n_fft)
            frame[:end-start] = audio[start:end]
            frame = frame * window
            
            # Simple DFT
            fft_frame = np.fft.fft(frame)[:self.n_fft // 2 + 1]
            stft_matrix[:, i] = np.abs(fft_frame)
        
        # Apply mel filterbank
        mel_spec = self.mel_filters @ stft_matrix
        return np.log(mel_spec + 1e-8)
    
    def _create_mel_filterbank(self) -> np.ndarray:
        """Create mel-scale filterbank."""
        n_fft_bins = self.n_fft // 2 + 1
        
        # Mel scale functions
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595) - 1)
        
        # Create mel filterbank
        mel_points = np.linspace(hz_to_mel(0), hz_to_mel(self.sample_rate / 2), self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((n_fft_bins - 1) * hz_points / (self.sample_rate / 2)).astype(int)
        
        filterbank = np.zeros((self.n_mels, n_fft_bins))
        
        for i in range(1, self.n_mels + 1):
            left = bin_points[i - 1]
            center = bin_points[i]
            right = bin_points[i + 1]
            
            # Triangular filters
            for j in range(left, center):
                if center > left:
                    filterbank[i - 1, j] = (j - left) / (center - left)
            
            for j in range(center, right):
                if right > center:
                    filterbank[i - 1, j] = (right - j) / (right - center)
        
        return filterbank
    
    def _temporal_modeling(self, mel_spec: np.ndarray) -> np.ndarray:
        """Model temporal dependencies."""
        # Simple temporal convolution
        n_mels, n_frames = mel_spec.shape
        
        temporal_features = []
        for t in range(n_frames):
            frame_features = mel_spec[:, t] @ self.temporal_conv
            temporal_features.append(frame_features)
        
        return np.array(temporal_features)
    
    def _attention_pooling(self, temporal_features: np.ndarray) -> np.ndarray:
        """Attention-based temporal pooling."""
        if len(temporal_features) == 0:
            return np.zeros(self.d_model)
        
        # Compute attention weights
        attention_scores = temporal_features @ self.temporal_attention
        attention_weights = TextEncoder._softmax(attention_scores.flatten())
        
        # Weighted sum
        return np.sum(temporal_features * attention_weights[:, None], axis=0)

# ============================================================================
# CROSS-MODAL FUSION
# ============================================================================

class CrossModalFusion(ABC):
    """Base interface for multimodal fusion strategies."""
    
    def __init__(self, d_model: int = 512):
        self.d_model = d_model
    
    @abstractmethod
    def fuse(self, modality_embeddings: List[np.ndarray], 
            modality_masks: Optional[List[bool]] = None) -> np.ndarray:
        """Fuse embeddings from multiple modalities."""
        pass

class AttentionFusion(CrossModalFusion):
    """Cross-modal attention fusion."""
    
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super().__init__(d_model)
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Cross-modal attention
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02
        
        # Final fusion
        self.fusion_layer = np.random.randn(d_model, d_model) * 0.02
    
    def fuse(self, modality_embeddings: List[np.ndarray], 
            modality_masks: Optional[List[bool]] = None) -> np.ndarray:
        """Cross-modal attention fusion."""
        
        if not modality_embeddings:
            return np.zeros(self.d_model)
        
        # Handle masks
        if modality_masks is None:
            modality_masks = [True] * len(modality_embeddings)
        
        # Stack valid modalities
        valid_embeddings = []
        for emb, mask in zip(modality_embeddings, modality_masks):
            if mask and emb is not None:
                valid_embeddings.append(emb)
        
        if not valid_embeddings:
            return np.zeros(self.d_model)
        
        if len(valid_embeddings) == 1:
            return valid_embeddings[0] @ self.fusion_layer
        
        # Stack embeddings for attention
        stacked = np.stack(valid_embeddings)  # (num_modalities, d_model)
        n_modalities = stacked.shape[0]
        
        # Multi-head cross-modal attention
        Q = stacked @ self.W_q  # (n_modalities, d_model)
        K = stacked @ self.W_k
        V = stacked @ self.W_v
        
        # Reshape for multi-head
        Q = Q.reshape(n_modalities, self.num_heads, self.d_k).transpose(1, 0, 2)
        K = K.reshape(n_modalities, self.num_heads, self.d_k).transpose(1, 0, 2)
        V = V.reshape(n_modalities, self.num_heads, self.d_k).transpose(1, 0, 2)
        
        # Attention computation
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / math.sqrt(self.d_k)
        attn_weights = TextEncoder._softmax(scores)
        attended = np.matmul(attn_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 0, 2).reshape(n_modalities, self.d_model)
        
        # Output projection and fusion
        output = attended @ self.W_o
        fused = np.mean(output, axis=0)  # Simple aggregation
        
        return fused @ self.fusion_layer

class ConcatenationFusion(CrossModalFusion):
    """Simple concatenation-based fusion."""
    
    def __init__(self, d_model: int = 512, num_modalities: int = 3):
        super().__init__(d_model)
        self.num_modalities = num_modalities
        self.projection = np.random.randn(d_model * num_modalities, d_model) * 0.02
    
    def fuse(self, modality_embeddings: List[np.ndarray], 
            modality_masks: Optional[List[bool]] = None) -> np.ndarray:
        """Concatenation-based fusion."""
        
        # Prepare embeddings with masks
        processed_embeddings = []
        
        for i in range(self.num_modalities):
            if i < len(modality_embeddings) and modality_embeddings[i] is not None:
                if modality_masks is None or modality_masks[i]:
                    processed_embeddings.append(modality_embeddings[i])
                else:
                    processed_embeddings.append(np.zeros(self.d_model))
            else:
                processed_embeddings.append(np.zeros(self.d_model))
        
        # Concatenate and project
        concatenated = np.concatenate(processed_embeddings)
        return concatenated @ self.projection

class GatedFusion(CrossModalFusion):
    """Gated fusion with learnable modality weights."""
    
    def __init__(self, d_model: int = 512, num_modalities: int = 3):
        super().__init__(d_model)
        self.num_modalities = num_modalities
        
        # Gating networks
        self.gate_networks = [
            np.random.randn(d_model, 1) * 0.02 
            for _ in range(num_modalities)
        ]
        
        # Modality projections
        self.modality_projections = [
            np.random.randn(d_model, d_model) * 0.02
            for _ in range(num_modalities)
        ]
        
        # Final fusion
        self.fusion_projection = np.random.randn(d_model, d_model) * 0.02
    
    def fuse(self, modality_embeddings: List[np.ndarray], 
            modality_masks: Optional[List[bool]] = None) -> np.ndarray:
        """Gated multimodal fusion."""
        
        processed_embeddings = []
        gate_scores = []
        
        for i in range(min(len(modality_embeddings), self.num_modalities)):
            embedding = modality_embeddings[i]
            mask = modality_masks[i] if modality_masks else True
            
            if embedding is not None and mask:
                # Project embedding
                projected = embedding @ self.modality_projections[i]
                processed_embeddings.append(projected)
                
                # Compute gate score
                gate_score = float(np.sigmoid(embedding @ self.gate_networks[i]))
                gate_scores.append(gate_score)
            else:
                processed_embeddings.append(np.zeros(self.d_model))
                gate_scores.append(0.0)
        
        # Weighted fusion
        if not processed_embeddings:
            return np.zeros(self.d_model)
        
        # Normalize gate scores
        total_weight = sum(gate_scores) + 1e-8
        normalized_weights = [score / total_weight for score in gate_scores]
        
        # Weighted combination
        fused = sum(weight * embedding for weight, embedding 
                   in zip(normalized_weights, processed_embeddings))
        
        return fused @ self.fusion_projection

def np_sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

np.sigmoid = np_sigmoid

# ============================================================================
# MULTIMODAL PROCESSOR
# ============================================================================

class MultimodalProcessor:
    """Complete multimodal processing pipeline."""
    
    def __init__(self, d_model: int = 512, fusion_strategy: str = "attention"):
        self.d_model = d_model
        
        # Initialize encoders
        self.encoders = {
            Modality.TEXT: TextEncoder(d_model),
            Modality.IMAGE: ImageEncoder(d_model),
            Modality.AUDIO: AudioEncoder(d_model)
        }
        
        # Initialize fusion
        fusion_strategies = {
            "attention": AttentionFusion,
            "concatenation": ConcatenationFusion,
            "gated": GatedFusion
        }
        
        if fusion_strategy not in fusion_strategies:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        self.fusion = fusion_strategies[fusion_strategy](d_model)
        self.fusion_strategy = fusion_strategy
        
        # Processing cache
        self.cache = {}
    
    def process(self, modality_data: Dict[Modality, Any], 
               cache_key: Optional[str] = None) -> Dict[str, Any]:
        """Process multimodal data into unified representation."""
        
        # Check cache
        if cache_key and cache_key in self.cache:
            return self.cache[cache_key]
        
        embeddings = []
        masks = []
        processed_modalities = []
        
        # Process each modality
        for modality in [Modality.TEXT, Modality.IMAGE, Modality.AUDIO]:
            if modality in modality_data and modality_data[modality] is not None:
                try:
                    encoder = self.encoders[modality]
                    embedding = encoder.encode(modality_data[modality])
                    embeddings.append(embedding)
                    masks.append(True)
                    processed_modalities.append(modality.value)
                except Exception as e:
                    embeddings.append(np.zeros(self.d_model))
                    masks.append(False)
            else:
                embeddings.append(np.zeros(self.d_model))
                masks.append(False)
        
        # Fuse modalities
        if any(masks):
            fused_embedding = self.fusion.fuse(embeddings, masks)
        else:
            fused_embedding = np.zeros(self.d_model)
        
        result = {
            'fused_embedding': fused_embedding,
            'individual_embeddings': {
                mod.value: emb for mod, emb in zip([Modality.TEXT, Modality.IMAGE, Modality.AUDIO], embeddings)
            },
            'modality_masks': {
                mod.value: mask for mod, mask in zip([Modality.TEXT, Modality.IMAGE, Modality.AUDIO], masks)
            },
            'processed_modalities': processed_modalities,
            'fusion_strategy': self.fusion_strategy
        }
        
        # Cache result
        if cache_key:
            self.cache[cache_key] = result
        
        return result

# ============================================================================
# MODALITY ALIGNMENT
# ============================================================================

class ModalityAlignment:
    """Align embeddings across modalities using contrastive learning."""
    
    def __init__(self, d_model: int = 512):
        self.d_model = d_model
        
        # Alignment projections
        self.text_proj = np.random.randn(d_model, d_model) * 0.02
        self.image_proj = np.random.randn(d_model, d_model) * 0.02
        self.audio_proj = np.random.randn(d_model, d_model) * 0.02
        
        self.projections = {
            Modality.TEXT: self.text_proj,
            Modality.IMAGE: self.image_proj,
            Modality.AUDIO: self.audio_proj
        }
    
    def align_embeddings(self, embeddings: Dict[Modality, np.ndarray]) -> Dict[Modality, np.ndarray]:
        """Project embeddings to aligned space."""
        aligned = {}
        
        for modality, embedding in embeddings.items():
            if modality in self.projections:
                aligned[modality] = embedding @ self.projections[modality]
                # L2 normalize
                norm = np.linalg.norm(aligned[modality])
                if norm > 0:
                    aligned[modality] = aligned[modality] / norm
            else:
                aligned[modality] = embedding
        
        return aligned
    
    def compute_similarity_matrix(self, aligned_embeddings: Dict[Modality, np.ndarray]) -> np.ndarray:
        """Compute pairwise similarities between modalities."""
        modalities = list(aligned_embeddings.keys())
        n_modalities = len(modalities)
        
        similarity_matrix = np.zeros((n_modalities, n_modalities))
        
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities):
                emb1 = aligned_embeddings[mod1]
                emb2 = aligned_embeddings[mod2]
                similarity = np.dot(emb1, emb2)  # Cosine similarity (already normalized)
                similarity_matrix[i, j] = similarity
        
        return similarity_matrix

# ============================================================================
# MULTIMODAL RAG
# ============================================================================

class MultimodalRAG:
    """Multimodal retrieval-augmented generation system."""
    
    def __init__(self, d_model: int = 512):
        self.d_model = d_model
        self.processor = MultimodalProcessor(d_model, fusion_strategy="attention")
        self.alignment = ModalityAlignment(d_model)
        
        # Knowledge base
        self.knowledge_base = []
        self.embeddings_cache = []
        
    def add_multimodal_document(self, doc_id: str, 
                               text: Optional[Any] = None,
                               image: Optional[Any] = None,
                               audio: Optional[Any] = None,
                               metadata: Optional[Dict] = None):
        """Add multimodal document to knowledge base."""
        
        # Process multimodal content
        modality_data = {}
        if text is not None:
            modality_data[Modality.TEXT] = text
        if image is not None:
            modality_data[Modality.IMAGE] = image
        if audio is not None:
            modality_data[Modality.AUDIO] = audio
        
        if not modality_data:
            return None
        
        processed = self.processor.process(modality_data, cache_key=doc_id)
        
        document = {
            'doc_id': doc_id,
            'modality_data': modality_data,
            'processed': processed,
            'metadata': metadata or {}
        }
        
        self.knowledge_base.append(document)
        self.embeddings_cache.append(processed['fused_embedding'])
        
        return len(self.knowledge_base) - 1
    
    def retrieve(self, query_text: Optional[Any] = None,
                query_image: Optional[Any] = None,
                query_audio: Optional[Any] = None,
                top_k: int = 5) -> List[Dict]:
        """Retrieve relevant multimodal documents."""
        
        # Process query
        query_data = {}
        if query_text is not None:
            query_data[Modality.TEXT] = query_text
        if query_image is not None:
            query_data[Modality.IMAGE] = query_image
        if query_audio is not None:
            query_data[Modality.AUDIO] = query_audio
        
        if not query_data:
            return []
        
        query_processed = self.processor.process(query_data)
        query_embedding = query_processed['fused_embedding']
        
        # Compute similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings_cache):
            similarity = np.dot(query_embedding, doc_embedding)
            similarity /= (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-8)
            similarities.append((i, float(similarity)))
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_idx, similarity in similarities[:top_k]:
            doc = self.knowledge_base[doc_idx]
            results.append({
                'document': doc,
                'similarity': similarity,
                'doc_id': doc['doc_id']
            })
        
        return results

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_sample_data(d_model: int = 512) -> Dict[Modality, Any]:
    """Create sample multimodal data for testing."""
    return {
        Modality.TEXT: np.random.randint(0, 1000, 50),  # Token IDs
        Modality.IMAGE: np.random.rand(224, 224, 3),    # RGB image
        Modality.AUDIO: np.random.randn(16000) * 0.1    # Audio waveform
    }

def benchmark_multimodal_processing(processor_class, num_trials: int = 10) -> Dict[str, Any]:
    """Benchmark multimodal processing performance."""
    import time
    
    results = []
    
    for trial in range(num_trials):
        # Create sample data
        data = create_sample_data()
        
        processor = processor_class()
        
        start_time = time.time()
        result = processor.process(data)
        processing_time = time.time() - start_time
        
        results.append({
            'processing_time': processing_time,
            'modalities_processed': len(result['processed_modalities']),
            'fusion_success': np.linalg.norm(result['fused_embedding']) > 0
        })
    
    return {
        'mean_processing_time': np.mean([r['processing_time'] for r in results]),
        'mean_modalities_processed': np.mean([r['modalities_processed'] for r in results]),
        'success_rate': np.mean([r['fusion_success'] for r in results])
    }

def visualize_modality_similarities(similarity_matrix: np.ndarray, 
                                  modality_names: List[str]) -> str:
    """Create text visualization of modality similarity matrix."""
    
    result = "\nModality Similarity Matrix:\n"
    result += "=" * 40 + "\n"
    
    # Header
    result += "        "
    for name in modality_names:
        result += f"{name[:8]:>8s} "
    result += "\n"
    
    # Matrix rows
    for i, name in enumerate(modality_names):
        result += f"{name[:8]:>8s} "
        for j in range(len(modality_names)):
            result += f"{similarity_matrix[i, j]:8.3f} "
        result += "\n"
    
    return result

# ----------------------------------------------------------------------
# INDIVIDUAL EXAMPLE FUNCTIONS
# ----------------------------------------------------------------------

def example_01_individual_encoders():
    example_name = "example_01_individual_encoders"
    ex_dir = create_example_dir(example_name)
    log = get_example_logger(example_name, ex_dir)

    sample = create_sample_data(d_model=512)
    encoders = {
        "text":  (TextEncoder(d_model=512),  sample[Modality.TEXT]),
        "image": (ImageEncoder(d_model=512), sample[Modality.IMAGE]),
        "audio": (AudioEncoder(d_model=512), sample[Modality.AUDIO]),
    }

    results = {}
    for name, (enc, data) in encoders.items():
        import time
        t0 = time.time()
        emb = enc.encode(data)
        elapsed = (time.time() - t0) * 1000
        results[name] = {
            "embedding": emb,
            "norm": float(np.linalg.norm(emb)),
            "processing_time_ms": elapsed
        }
        log.info(f"{name.capitalize()} encoder → norm {results[name]['norm']:.3f} | time {elapsed:.2f}ms")

    # Save everything
    save_json({"d_model": 512}, os.path.join(ex_dir, "config.json"))
    for name, info in results.items():
        save_npy(info["embedding"], os.path.join(ex_dir, f"{name}_embedding.npy"))
    save_json({k: {"norm": v["norm"], "time_ms": v["processing_time_ms"]} for k, v in results.items()},
              os.path.join(ex_dir, "summary.json"))

    # Markdown report
    Path(os.path.join(ex_dir, "report.md")).write_text(
f"""# Example 01 – Individual Modality Encoders

All encoders produce 512-dim embeddings.

| Modality | Norm    | Time (ms) |
|----------|---------|-----------|
| Text     | {results['text']['norm']:.3f} | {results['text']['processing_time_ms']:.2f} |
| Image    | {results['image']['norm']:.3f} | {results['image']['processing_time_ms']:.2f} |
| Audio    | {results['audio']['norm']:.3f} | {results['audio']['processing_time_ms']:.2f} |

Raw embeddings saved as `*_embedding.npy`.
""")

def example_02_fusion_strategies_comparison():
    example_name = "example_02_fusion_strategies_comparison"
    ex_dir = create_example_dir(example_name)
    log = get_example_logger(example_name, ex_dir)

    # Encode once
    sample = create_sample_data()
    text_emb  = TextEncoder(d_model=512).encode(sample[Modality.TEXT])
    img_emb   = ImageEncoder(d_model=512).encode(sample[Modality.IMAGE])
    audio_emb = AudioEncoder(d_model=512).encode(sample[Modality.AUDIO])
    embeddings = [text_emb, img_emb, audio_emb]

    strategies = {
        "attention":      AttentionFusion(d_model=512),
        "concatenation": ConcatenationFusion(d_model=512, num_modalities=3),
        "gated":          GatedFusion(d_model=512, num_modalities=3),
    }

    results = {}
    for name, fusion in strategies.items():
        import time
        t0 = time.time()
        fused = fusion.fuse(embeddings)
        elapsed = (time.time() - t0) * 1000
        results[name] = {"fused": fused, "norm": float(np.linalg.norm(fused)), "time_ms": elapsed}
        log.info(f"{name.capitalize()} fusion → norm {results[name]['norm']:.3f} | time {elapsed:.2f}ms")

    save_json({"strategies": list(strategies.keys())}, os.path.join(ex_dir, "config.json"))
    for name, info in results.items():
        save_npy(info["fused"], os.path.join(ex_dir, f"fused_{name}.npy"))
    save_json({k: {"norm": v["norm"], "time_ms": v["time_ms"]} for k, v in results.items()},
              os.path.join(ex_dir, "summary.json"))

def example_03_complete_multimodal_processor():
    example_name = "example_03_complete_multimodal_processor"
    ex_dir = create_example_dir(example_name)
    log = get_example_logger(example_name, ex_dir)

    processor = MultimodalProcessor(d_model=512, fusion_strategy="attention")
    sample = create_sample_data()

    result = processor.process(sample, cache_key="demo")

    save_json(result, os.path.join(ex_dir, "full_result.json"))
    save_npy(result["fused_embedding"], os.path.join(ex_dir, "fused_embedding.npy"))
    for mod, emb in result["individual_embeddings"].items():
        save_npy(emb, os.path.join(ex_dir, f"individual_{mod}.npy"))

    log.info(f"Fusion strategy: {result['fusion_strategy']}")
    log.info(f"Processed modalities: {result['processed_modalities']}")
    log.info(f"Fused norm: {np.linalg.norm(result['fused_embedding']):.3f}")

def example_04_modality_alignment():
    example_name = "example_04_modality_alignment"
    ex_dir = create_example_dir(example_name)
    log = get_example_logger(example_name, ex_dir)

    sample = create_sample_data()
    encoders = {Modality.TEXT: TextEncoder, Modality.IMAGE: ImageEncoder, Modality.AUDIO: AudioEncoder}
    raw_embs = {mod: encoders[mod](d_model=512).encode(sample[mod]) for mod in encoders}

    alignment = ModalityAlignment(d_model=512)
    aligned = alignment.align_embeddings(raw_embs)
    sim_matrix = alignment.compute_similarity_matrix(aligned)

    labels = [mod.value for mod in aligned.keys()]
    plot_similarity_matrix(sim_matrix, labels, os.path.join(ex_dir, "similarity_matrix.png"))
    save_json({"matrix": sim_matrix.tolist(), "labels": labels}, os.path.join(ex_dir, "similarity.json"))

    log.info("Similarity matrix saved + visualized")

def example_05_multimodal_rag():
    example_name = "example_05_multimodal_rag"
    ex_dir = create_example_dir(example_name)
    log = get_example_logger(example_name, ex_dir)

    rag = MultimodalRAG(d_model=512)

    # Add 5 documents
    for i in range(5):
        data = create_sample_data()
        rag.add_multimodal_document(
            doc_id=f"doc_{i:02d}",
            text=data[Modality.TEXT],
            image=data[Modality.IMAGE],
            audio=data[Modality.AUDIO],
            metadata={"source": f"sample_{i}"}
        )

    # Query with text+image
    query = create_sample_data()
    retrieved = rag.retrieve(
        query_text=query[Modality.TEXT],
        query_image=query[Modality.IMAGE],
        top_k=3
    )

    retrieval_results = [
        {"doc_id": r["doc_id"], "similarity": float(r["similarity"])} for r in retrieved
    ]
    save_json(retrieval_results, os.path.join(ex_dir, "retrieval_results.json"))
    log.info(f"Top-3 retrieved: {[r['doc_id'] for r in retrieved]}")

# ----------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# ----------------------------------------------------------------------
if __name__ == "__main__":
    examples = [
        example_01_individual_encoders,
        example_02_fusion_strategies_comparison,
        example_03_complete_multimodal_processor,
        example_04_modality_alignment,
        example_05_multimodal_rag,
    ]

    main_logger.info(f"Running {len(examples)} multimodal processor examples …")
    for fn in examples:
        try:
            fn()
            main_logger.info(f"Completed {fn.__name__}")
        except Exception as e:
            main_logger.error(f"Failed {fn.__name__}: {e}", exc_info=True)

    main_logger.info(f"All examples finished – see {BASE_OUTPUT_DIR}")
    main_logger.info("=" * 80)
