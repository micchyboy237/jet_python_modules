#!/usr/bin/env python3
"""
Multimodal Context Processing Lab
=================================

Context Engineering Course - Module 02: Context Processing
Production-ready multimodal context integration for text, image, and audio.

Learning Objectives:
- Build unified multimodal representations
- Implement cross-modal attention and fusion mechanisms
- Create production-ready multimodal processing pipelines
- Deploy multimodal RAG and content analysis systems

Research Foundation:
- CLIP (Radford et al.) - Vision-language understanding
- ALIGN (Jia et al.) - Large-scale multimodal alignment
- Flamingo (Alayrac et al.) - Few-shot learning across modalities
- DALL-E 2 (Ramesh et al.) - Text-to-image generation principles
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

import shutil
from pathlib import Path

try:
    from jet.logger import CustomLogger
    base_logger = CustomLogger(name="multimodal_lab", filename="multimodal_lab.log", overwrite=True)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    base_logger = logging.getLogger("multimodal_lab")

def create_example_dir(example_name: str) -> Path:
    base_dir = Path(__file__).parent / "generated" / Path(__file__).stem
    example_dir = base_dir / example_name
    shutil.rmtree(example_dir, ignore_errors=True)
    example_dir.mkdir(parents=True, exist_ok=True)
    return example_dir

def get_logger(name: str, output_dir: Path):
    log_file = output_dir / "run.log"
    return CustomLogger(name=name, filename=log_file, overwrite=True)

def save_json(data: Any, path: Path, name: str = "data") -> None:
    with open(path / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))

def save_plot(fig: plt.Figure, path: Path, name: str = "plot") -> None:
    fig.savefig(path / f"{name}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

def save_numpy(arr: np.ndarray, path: Path, name: str) -> None:
    np.save(path / f"{name}.npy", arr)

# ============================================================================
# CORE INTERFACES & UTILITIES
# ============================================================================

class Modality(Enum):
    """Supported modalities for context processing."""
    TEXT = "text"
    IMAGE = "image"  
    AUDIO = "audio"
    MULTIMODAL = "multimodal"

@dataclass
class ModalityData:
    """Container for single modality data."""
    modality: Modality
    data: np.ndarray
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class MultimodalContext:
    """Unified container for multimodal context data."""
    text: Optional[np.ndarray] = None
    image: Optional[np.ndarray] = None
    audio: Optional[np.ndarray] = None
    fused: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def available_modalities(self) -> List[Modality]:
        """Get list of available modalities."""
        modalities = []
        if self.text is not None:
            modalities.append(Modality.TEXT)
        if self.image is not None:
            modalities.append(Modality.IMAGE)
        if self.audio is not None:
            modalities.append(Modality.AUDIO)
        return modalities
    
    def get_modality_data(self, modality: Modality) -> Optional[np.ndarray]:
        """Get data for specific modality."""
        if modality == Modality.TEXT:
            return self.text
        elif modality == Modality.IMAGE:
            return self.image
        elif modality == Modality.AUDIO:
            return self.audio
        elif modality == Modality.MULTIMODAL:
            return self.fused
        return None

class ModalityEncoder(ABC):
    """Base interface for modality-specific encoders."""
    
    @abstractmethod
    def encode(self, data: Any) -> np.ndarray:
        """Encode raw data into embedding space."""
        pass
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Output dimension of encoder."""
        pass

class MultimodalFusion(ABC):
    """Base interface for multimodal fusion strategies."""
    
    @abstractmethod
    def fuse(self, multimodal_context: MultimodalContext) -> np.ndarray:
        """Fuse multiple modalities into unified representation."""
        pass

# ============================================================================
# MODALITY ENCODERS
# ============================================================================

class TextEncoder(ModalityEncoder):
    """Production-ready text encoder with contextual embeddings."""
    
    def __init__(self, d_model: int = 512, vocab_size: int = 50000, max_seq_len: int = 512):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Simplified transformer-like encoder
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.positional_encoding = self._create_positional_encoding()
        
        # Multi-head attention parameters
        self.num_heads = 8
        self.d_k = d_model // self.num_heads
        self.W_qkv = np.random.randn(d_model, 3 * d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02
        
        # Feed-forward network
        self.ff_hidden = d_model * 4
        self.W_ff1 = np.random.randn(d_model, self.ff_hidden) * 0.02
        self.W_ff2 = np.random.randn(self.ff_hidden, d_model) * 0.02
    
    def encode(self, text_tokens: np.ndarray) -> np.ndarray:
        """Encode text tokens into contextual embeddings."""
        seq_len = min(text_tokens.shape[0], self.max_seq_len)
        tokens = text_tokens[:seq_len]
        
        # Token embeddings + positional encoding
        token_embeds = self.token_embedding[tokens.astype(int) % self.vocab_size]
        pos_embeds = self.positional_encoding[:seq_len]
        x = token_embeds + pos_embeds
        
        # Self-attention layer
        x = self._self_attention(x)
        
        # Feed-forward layer
        x = self._feed_forward(x)
        
        # Global pooling for sequence representation
        return np.mean(x, axis=0)
    
    def _create_positional_encoding(self) -> np.ndarray:
        """Create sinusoidal positional encodings."""
        pos_enc = np.zeros((self.max_seq_len, self.d_model))
        position = np.arange(self.max_seq_len)[:, None]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pos_enc[:, 0::2] = np.sin(position * div_term)
        pos_enc[:, 1::2] = np.cos(position * div_term)
        
        return pos_enc
    
    def _self_attention(self, x: np.ndarray) -> np.ndarray:
        """Apply multi-head self-attention."""
        seq_len, d_model = x.shape
        
        # Compute Q, K, V
        qkv = x @ self.W_qkv
        qkv = qkv.reshape(seq_len, 3, self.num_heads, self.d_k)
        q, k, v = qkv.transpose(1, 2, 0, 3)  # (3, num_heads, seq_len, d_k)
        
        # Scaled dot-product attention
        scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(self.d_k)
        attn_weights = self._softmax(scores)
        out = np.matmul(attn_weights, v)
        
        # Concatenate heads and apply output projection
        out = out.transpose(1, 0, 2).reshape(seq_len, d_model)
        return x + (out @ self.W_o)  # Residual connection
    
    def _feed_forward(self, x: np.ndarray) -> np.ndarray:
        """Apply feed-forward network."""
        ff_out = np.maximum(0, x @ self.W_ff1) @ self.W_ff2  # ReLU activation
        return x + ff_out  # Residual connection
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        max_vals = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - max_vals)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    @property
    def output_dim(self) -> int:
        return self.d_model

class ImageEncoder(ModalityEncoder):
    """Production-ready image encoder with convolutional features."""
    
    def __init__(self, d_model: int = 512, input_channels: int = 3, image_size: int = 224):
        self.d_model = d_model
        self.input_channels = input_channels
        self.image_size = image_size
        
        # Simplified CNN-like encoder (representing ResNet/ViT-like architecture)
        # Patch embedding for vision transformer approach
        self.patch_size = 16
        self.num_patches = (image_size // self.patch_size) ** 2
        patch_dim = input_channels * self.patch_size * self.patch_size
        
        self.patch_embedding = np.random.randn(patch_dim, d_model) * 0.02
        self.positional_embedding = np.random.randn(self.num_patches + 1, d_model) * 0.02  # +1 for CLS token
        self.cls_token = np.random.randn(1, d_model) * 0.02
        
        # Transformer layers for image processing
        self.num_heads = 8
        self.d_k = d_model // self.num_heads
        self.W_qkv = np.random.randn(d_model, 3 * d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02
        
        # Classification head
        self.classifier = np.random.randn(d_model, d_model) * 0.02
    
    def encode(self, image_data: np.ndarray) -> np.ndarray:
        """Encode image into embedding space."""
        # Simulate patch extraction and embedding
        # In practice, this would be proper CNN/ViT processing
        
        if len(image_data.shape) == 3:
            # Single image: (height, width, channels)
            patches = self._extract_patches(image_data)
        elif len(image_data.shape) == 1:
            # Pre-flattened image features
            patches = image_data.reshape(-1, self.d_model)[:self.num_patches]
        else:
            # Use as-is for flexibility
            patches = image_data[:self.num_patches] if image_data.shape[0] >= self.num_patches else image_data
        
        # Ensure correct dimensions
        if patches.shape[1] != self.d_model:
            # Project to correct dimension
            if patches.shape[1] > self.d_model:
                patches = patches[:, :self.d_model]
            else:
                padding = np.zeros((patches.shape[0], self.d_model - patches.shape[1]))
                patches = np.concatenate([patches, padding], axis=1)
        
        # Add CLS token and positional embeddings
        cls_tokens = np.repeat(self.cls_token, 1, axis=0)
        x = np.concatenate([cls_tokens, patches], axis=0)
        
        # Add positional embeddings
        seq_len = min(x.shape[0], self.positional_embedding.shape[0])
        x = x[:seq_len] + self.positional_embedding[:seq_len]
        
        # Apply transformer attention
        x = self._image_attention(x)
        
        # Return CLS token representation
        return x[0] @ self.classifier
    
    def _extract_patches(self, image: np.ndarray) -> np.ndarray:
        """Extract patches from image for vision transformer."""
        # Simplified patch extraction
        height, width, channels = image.shape
        
        # Resize to expected size if needed
        if height != self.image_size or width != self.image_size:
            # Simple interpolation simulation
            scale_h = self.image_size / height
            scale_w = self.image_size / width
            image = self._simple_resize(image, (self.image_size, self.image_size))
        
        # Extract patches
        patches = []
        for i in range(0, self.image_size, self.patch_size):
            for j in range(0, self.image_size, self.patch_size):
                patch = image[i:i+self.patch_size, j:j+self.patch_size, :]
                patch_flat = patch.flatten()
                
                # Project to model dimension
                if len(patch_flat) == self.patch_size * self.patch_size * self.input_channels:
                    patch_embed = patch_flat @ self.patch_embedding[:len(patch_flat), :self.d_model]
                else:
                    # Handle edge patches
                    patch_embed = np.zeros(self.d_model)
                
                patches.append(patch_embed)
        
        return np.array(patches)
    
    def _simple_resize(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Simple image resizing simulation."""
        # In practice, use proper image resizing libraries
        height, width = image.shape[:2]
        target_h, target_w = target_size
        
        # Simple nearest neighbor simulation
        resized = np.zeros((target_h, target_w, image.shape[2]))
        for i in range(target_h):
            for j in range(target_w):
                src_i = int(i * height / target_h)
                src_j = int(j * width / target_w)
                resized[i, j] = image[min(src_i, height-1), min(src_j, width-1)]
        
        return resized
    
    def _image_attention(self, x: np.ndarray) -> np.ndarray:
        """Apply attention mechanism to image patches."""
        seq_len, d_model = x.shape
        
        # Compute Q, K, V
        qkv = x @ self.W_qkv
        qkv = qkv.reshape(seq_len, 3, self.num_heads, self.d_k)
        q, k, v = qkv.transpose(1, 2, 0, 3)
        
        # Attention computation
        scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(self.d_k)
        attn_weights = self._softmax(scores)
        out = np.matmul(attn_weights, v)
        
        # Output projection
        out = out.transpose(1, 0, 2).reshape(seq_len, d_model)
        return x + (out @ self.W_o)
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        max_vals = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - max_vals)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    @property
    def output_dim(self) -> int:
        return self.d_model

class AudioEncoder(ModalityEncoder):
    """Production-ready audio encoder with spectral features."""
    
    def __init__(self, d_model: int = 512, sample_rate: int = 16000, n_fft: int = 512):
        self.d_model = d_model
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = n_fft // 4
        
        # Spectral feature processing
        self.mel_filters = self._create_mel_filterbank(n_fft // 2 + 1, 80)  # 80 mel bands
        
        # Temporal modeling with recurrent-like processing
        self.temporal_weights = np.random.randn(80, d_model) * 0.02
        self.temporal_bias = np.zeros(d_model)
        
        # Attention mechanism for temporal aggregation
        self.temporal_attention = np.random.randn(d_model, 1) * 0.02
        
        # Output projection
        self.output_projection = np.random.randn(d_model, d_model) * 0.02
    
    def encode(self, audio_data: np.ndarray) -> np.ndarray:
        """Encode audio into embedding space."""
        # Handle different input formats
        if len(audio_data.shape) == 1:
            # Raw audio waveform
            spectral_features = self._compute_mel_spectrogram(audio_data)
        elif len(audio_data.shape) == 2:
            # Pre-computed features
            spectral_features = audio_data
        else:
            # Flatten to 1D and process
            audio_data = audio_data.flatten()
            spectral_features = self._compute_mel_spectrogram(audio_data)
        
        # Temporal modeling
        temporal_embeddings = self._temporal_modeling(spectral_features)
        
        # Temporal attention pooling
        attended_features = self._temporal_attention_pooling(temporal_embeddings)
        
        # Final output projection
        return attended_features @ self.output_projection
    
    def _compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel-scale spectrogram from audio."""
        # Simplified STFT computation
        # In practice, use librosa or similar
        
        # Ensure minimum length
        if len(audio) < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - len(audio)))
        
        # Windowed STFT simulation
        num_frames = (len(audio) - self.n_fft) // self.hop_length + 1
        stft_matrix = np.zeros((self.n_fft // 2 + 1, num_frames))
        
        # Hanning window
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(self.n_fft) / (self.n_fft - 1)))
        
        for i in range(num_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.n_fft] * window
            
            # Simplified DFT
            fft_frame = np.fft.fft(frame)[:self.n_fft // 2 + 1]
            stft_matrix[:, i] = np.abs(fft_frame)
        
        # Apply mel filterbank
        mel_spectrogram = self.mel_filters @ stft_matrix
        
        # Log compression
        return np.log(mel_spectrogram + 1e-8)
    
    def _create_mel_filterbank(self, n_fft_bins: int, n_mels: int) -> np.ndarray:
        """Create mel-scale filterbank."""
        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595) - 1)
        
        # Frequency range
        min_freq = 0
        max_freq = self.sample_rate / 2
        
        # Mel frequency points
        mel_min = hz_to_mel(min_freq)
        mel_max = hz_to_mel(max_freq)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Convert to FFT bin indices
        bin_points = np.floor((n_fft_bins * 2 - 1) * hz_points / self.sample_rate).astype(int)
        
        # Create filterbank
        filterbank = np.zeros((n_mels, n_fft_bins))
        
        for i in range(1, n_mels + 1):
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
    
    def _temporal_modeling(self, spectral_features: np.ndarray) -> np.ndarray:
        """Model temporal dependencies in spectral features."""
        n_mels, n_frames = spectral_features.shape
        
        # Simple recurrent-like processing
        embeddings = []
        hidden_state = np.zeros(self.d_model)
        
        for t in range(n_frames):
            # Input transformation
            input_features = spectral_features[:, t] @ self.temporal_weights + self.temporal_bias
            
            # Simple recurrent update
            hidden_state = 0.7 * hidden_state + 0.3 * np.tanh(input_features)
            embeddings.append(hidden_state.copy())
        
        return np.array(embeddings)
    
    def _temporal_attention_pooling(self, temporal_embeddings: np.ndarray) -> np.ndarray:
        """Apply attention-based temporal pooling."""
        # Compute attention weights
        attention_scores = temporal_embeddings @ self.temporal_attention
        attention_weights = self._softmax(attention_scores.flatten())
        
        # Weighted aggregation
        attended_features = np.sum(temporal_embeddings * attention_weights[:, None], axis=0)
        
        return attended_features
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        max_val = np.max(x)
        exp_x = np.exp(x - max_val)
        return exp_x / np.sum(exp_x)
    
    @property
    def output_dim(self) -> int:
        return self.d_model

# ============================================================================
# MULTIMODAL FUSION STRATEGIES
# ============================================================================

class CrossModalAttentionFusion(MultimodalFusion):
    """Cross-modal attention fusion with learnable interactions."""
    
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Cross-modal attention parameters
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02
        
        # Modality-specific projections
        self.text_proj = np.random.randn(d_model, d_model) * 0.02
        self.image_proj = np.random.randn(d_model, d_model) * 0.02
        self.audio_proj = np.random.randn(d_model, d_model) * 0.02
        
        # Final fusion layers
        self.fusion_gate = np.random.randn(d_model * 3, 3) * 0.02  # Changed from (d_model*3, d_model)
        self.output_projection = np.random.randn(d_model, d_model) * 0.02
    
    def fuse(self, multimodal_context: MultimodalContext) -> np.ndarray:
        """Fuse multimodal context using cross-modal attention."""
        modality_embeddings = []
        modality_masks = []
        
        # Project each available modality
        if multimodal_context.text is not None:
            text_embed = multimodal_context.text @ self.text_proj
            modality_embeddings.append(text_embed)
            modality_masks.append(1.0)
        else:
            modality_embeddings.append(np.zeros(self.d_model))
            modality_masks.append(0.0)
        
        if multimodal_context.image is not None:
            image_embed = multimodal_context.image @ self.image_proj
            modality_embeddings.append(image_embed)
            modality_masks.append(1.0)
        else:
            modality_embeddings.append(np.zeros(self.d_model))
            modality_masks.append(0.0)
        
        if multimodal_context.audio is not None:
            audio_embed = multimodal_context.audio @ self.audio_proj
            modality_embeddings.append(audio_embed)
            modality_masks.append(1.0)
        else:
            modality_embeddings.append(np.zeros(self.d_model))
            modality_masks.append(0.0)
        
        # Stack modalities
        modality_stack = np.array(modality_embeddings)  # Shape: (3, d_model)
        modality_mask = np.array(modality_masks)
        
        # Cross-modal attention
        attended_modalities = self._cross_modal_attention(modality_stack, modality_mask)
        
        # Gated fusion
        concatenated = attended_modalities.flatten()
        fusion_weights = self._sigmoid(concatenated @ self.fusion_gate)
        fusion_weights = fusion_weights / (fusion_weights.sum() + 1e-8)  # Softmax-like normalization
        
        # Weighted combination
        fused_embedding = np.sum(attended_modalities * fusion_weights[:, None], axis=0)
        
        # Final projection
        return fused_embedding @ self.output_projection
    
    def _cross_modal_attention(self, modality_embeddings: np.ndarray, 
                              modality_mask: np.ndarray) -> np.ndarray:
        """Apply cross-modal attention between modalities."""
        n_modalities, d_model = modality_embeddings.shape
        
        # Compute Q, K, V for each modality
        Q = modality_embeddings @ self.W_q  # (n_modalities, d_model)
        K = modality_embeddings @ self.W_k
        V = modality_embeddings @ self.W_v
        
        # Reshape for multi-head attention
        Q = Q.reshape(n_modalities, self.num_heads, self.d_k).transpose(1, 0, 2)
        K = K.reshape(n_modalities, self.num_heads, self.d_k).transpose(1, 0, 2)
        V = V.reshape(n_modalities, self.num_heads, self.d_k).transpose(1, 0, 2)
        
        # Scaled dot-product attention
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.d_k)
        
        # Apply modality mask
        mask_expanded = modality_mask[None, None, :] * modality_mask[None, :, None]
        scores = np.where(mask_expanded, scores, -1e9)
        
        # Softmax and attend
        attn_weights = self._softmax(scores, axis=-1)
        attended = np.matmul(attn_weights, V)
        
        # Concatenate heads and project
        attended = attended.transpose(1, 0, 2).reshape(n_modalities, d_model)
        output = attended @ self.W_o
        
        # Residual connection with mask
        return modality_embeddings + output * modality_mask[:, None]
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        max_vals = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - max_vals)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class HierarchicalFusion(MultimodalFusion):
    """Hierarchical fusion with early and late fusion stages."""
    
    def __init__(self, d_model: int = 512):
        self.d_model = d_model
        
        # Early fusion (pairwise)
        self.text_image_fusion = np.random.randn(d_model * 2, d_model) * 0.02
        self.text_audio_fusion = np.random.randn(d_model * 2, d_model) * 0.02
        self.image_audio_fusion = np.random.randn(d_model * 2, d_model) * 0.02
        
        # Late fusion (final combination)
        self.final_fusion = np.random.randn(d_model * 3, d_model) * 0.02
        
        # Adaptive weighting
        self.modality_weights = np.random.randn(3, 1) * 0.02
    
    def fuse(self, multimodal_context: MultimodalContext) -> np.ndarray:
        """Hierarchical fusion with early and late stages."""
        modalities = []
        
        # Early fusion - pairwise interactions
        if (multimodal_context.text is not None and 
            multimodal_context.image is not None):
            text_image = np.concatenate([multimodal_context.text, multimodal_context.image])
            text_image_fused = np.tanh(text_image @ self.text_image_fusion)
            modalities.append(text_image_fused)
        else:
            modalities.append(np.zeros(self.d_model))
        
        if (multimodal_context.text is not None and 
            multimodal_context.audio is not None):
            text_audio = np.concatenate([multimodal_context.text, multimodal_context.audio])
            text_audio_fused = np.tanh(text_audio @ self.text_audio_fusion)
            modalities.append(text_audio_fused)
        else:
            modalities.append(np.zeros(self.d_model))
        
        if (multimodal_context.image is not None and 
            multimodal_context.audio is not None):
            image_audio = np.concatenate([multimodal_context.image, multimodal_context.audio])
            image_audio_fused = np.tanh(image_audio @ self.image_audio_fusion)
            modalities.append(image_audio_fused)
        else:
            modalities.append(np.zeros(self.d_model))
        
        # Late fusion - combine all pairwise interactions
        all_fused = np.concatenate(modalities)
        final_embedding = np.tanh(all_fused @ self.final_fusion)
        
        return final_embedding

# ============================================================================
# MULTIMODAL PROCESSING PIPELINE
# ============================================================================

class MultimodalProcessor:
    """Production-ready multimodal context processor."""
    
    def __init__(self, d_model: int = 512, fusion_strategy: str = 'cross_attention'):
        self.d_model = d_model
        
        # Initialize encoders
        self.text_encoder = TextEncoder(d_model)
        self.image_encoder = ImageEncoder(d_model)
        self.audio_encoder = AudioEncoder(d_model)
        
        # Initialize fusion strategy
        if fusion_strategy == 'cross_attention':
            self.fusion = CrossModalAttentionFusion(d_model)
        elif fusion_strategy == 'hierarchical':
            self.fusion = HierarchicalFusion(d_model)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        # Processing cache for efficiency
        self.cache = {}
        self.processing_stats = []
    
    def process_multimodal_context(self, 
                                  text_data: Optional[Any] = None,
                                  image_data: Optional[Any] = None,
                                  audio_data: Optional[Any] = None,
                                  cache_key: Optional[str] = None) -> MultimodalContext:
        """Process multimodal inputs into unified context."""
        
        start_time = time.time()
        
        # Check cache
        if cache_key and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Encode each modality
        context = MultimodalContext()
        
        if text_data is not None:
            context.text = self.text_encoder.encode(text_data)
        
        if image_data is not None:
            context.image = self.image_encoder.encode(image_data)
        
        if audio_data is not None:
            context.audio = self.audio_encoder.encode(audio_data)
        
        # Fuse modalities
        if len(context.available_modalities) > 1:
            context.fused = self.fusion.fuse(context)
        elif len(context.available_modalities) == 1:
            # Single modality - use directly
            modality = context.available_modalities[0]
            context.fused = context.get_modality_data(modality)
        
        # Store metadata
        context.metadata = {
            'processing_time': time.time() - start_time,
            'available_modalities': [m.value for m in context.available_modalities],
            'fusion_strategy': type(self.fusion).__name__
        }
        
        # Cache result
        if cache_key:
            self.cache[cache_key] = context
        
        # Record stats
        self.processing_stats.append({
            'modalities_count': len(context.available_modalities),
            'processing_time': context.metadata['processing_time'],
            'fusion_strategy': context.metadata['fusion_strategy']
        })
        
        return context
    
    def similarity_search(self, query_context: MultimodalContext, 
                         context_database: List[MultimodalContext],
                         top_k: int = 5) -> List[Tuple[int, float]]:
        """Multimodal similarity search."""
        
        if query_context.fused is None:
            return []
        
        similarities = []
        
        for i, db_context in enumerate(context_database):
            if db_context.fused is not None:
                # Cosine similarity
                dot_product = np.dot(query_context.fused, db_context.fused)
                norm_product = (np.linalg.norm(query_context.fused) * 
                              np.linalg.norm(db_context.fused))
                
                if norm_product > 0:
                    similarity = dot_product / norm_product
                else:
                    similarity = 0.0
                
                similarities.append((i, similarity))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# ============================================================================
# PRACTICAL APPLICATIONS
# ============================================================================

class MultimodalRAG:
    """Multimodal Retrieval-Augmented Generation system."""
    
    def __init__(self, d_model: int = 512):
        self.processor = MultimodalProcessor(d_model)
        self.knowledge_base = []
        self.index_metadata = []
    
    def add_to_knowledge_base(self, 
                             text_data: Optional[Any] = None,
                             image_data: Optional[Any] = None,
                             audio_data: Optional[Any] = None,
                             metadata: Optional[Dict] = None):
        """Add multimodal content to knowledge base."""
        
        context = self.processor.process_multimodal_context(
            text_data=text_data,
            image_data=image_data, 
            audio_data=audio_data
        )
        
        self.knowledge_base.append(context)
        self.index_metadata.append(metadata or {})
        
        return len(self.knowledge_base) - 1  # Return index
    
    def retrieve_relevant_context(self, 
                                 query_text: Optional[Any] = None,
                                 query_image: Optional[Any] = None,
                                 query_audio: Optional[Any] = None,
                                 top_k: int = 3) -> List[Dict]:
        """Retrieve relevant multimodal context for query."""
        
        # Process query
        query_context = self.processor.process_multimodal_context(
            text_data=query_text,
            image_data=query_image,
            audio_data=query_audio
        )
        
        # Search knowledge base
        results = self.processor.similarity_search(
            query_context, self.knowledge_base, top_k=top_k
        )
        
        # Format results with metadata
        formatted_results = []
        for idx, similarity in results:
            result = {
                'index': idx,
                'similarity': similarity,
                'context': self.knowledge_base[idx],
                'metadata': self.index_metadata[idx]
            }
            formatted_results.append(result)
        
        return formatted_results

class MultimodalContentAnalyzer:
    """Analyze and classify multimodal content."""
    
    def __init__(self, d_model: int = 512):
        self.processor = MultimodalProcessor(d_model)
        
        # Classification heads
        self.sentiment_classifier = np.random.randn(d_model, 3) * 0.02  # positive, neutral, negative
        self.topic_classifier = np.random.randn(d_model, 10) * 0.02     # 10 topics
        self.safety_classifier = np.random.randn(d_model, 2) * 0.02     # safe, unsafe
    
    def analyze_content(self, 
                       text_data: Optional[Any] = None,
                       image_data: Optional[Any] = None,
                       audio_data: Optional[Any] = None) -> Dict[str, Any]:
        """Comprehensive multimodal content analysis."""
        
        # Process content
        context = self.processor.process_multimodal_context(
            text_data=text_data,
            image_data=image_data,
            audio_data=audio_data
        )
        
        if context.fused is None:
            return {'error': 'No processable content found'}
        
        # Run classifications
        sentiment_scores = self._softmax(context.fused @ self.sentiment_classifier)
        topic_scores = self._softmax(context.fused @ self.topic_classifier)
        safety_scores = self._softmax(context.fused @ self.safety_classifier)
        
        # Compute quality metrics
        quality_metrics = self._compute_quality_metrics(context)
        
        return {
            'sentiment': {
                'positive': float(sentiment_scores[0]),
                'neutral': float(sentiment_scores[1]),
                'negative': float(sentiment_scores[2]),
                'predicted': ['positive', 'neutral', 'negative'][np.argmax(sentiment_scores)]
            },
            'topic_distribution': {
                f'topic_{i}': float(score) for i, score in enumerate(topic_scores)
            },
            'safety': {
                'safe_score': float(safety_scores[0]),
                'unsafe_score': float(safety_scores[1]),
                'classification': 'safe' if safety_scores[0] > safety_scores[1] else 'unsafe'
            },
            'quality_metrics': quality_metrics,
            'modalities_used': [m.value for m in context.available_modalities],
            'processing_time': context.metadata.get('processing_time', 0)
        }
    
    def _compute_quality_metrics(self, context: MultimodalContext) -> Dict[str, float]:
        """Compute content quality metrics."""
        
        if context.fused is None:
            return {}
        
        # Embedding magnitude (proxy for confidence)
        magnitude = float(np.linalg.norm(context.fused))
        
        # Modality consistency (if multiple modalities)
        consistency = 1.0
        if len(context.available_modalities) > 1:
            modality_embeddings = []
            if context.text is not None:
                modality_embeddings.append(context.text)
            if context.image is not None:
                modality_embeddings.append(context.image)
            if context.audio is not None:
                modality_embeddings.append(context.audio)
            
            if len(modality_embeddings) > 1:
                # Compute pairwise cosine similarities
                similarities = []
                for i in range(len(modality_embeddings)):
                    for j in range(i + 1, len(modality_embeddings)):
                        sim = np.dot(modality_embeddings[i], modality_embeddings[j])
                        sim /= (np.linalg.norm(modality_embeddings[i]) * 
                               np.linalg.norm(modality_embeddings[j]) + 1e-8)
                        similarities.append(sim)
                
                consistency = float(np.mean(similarities)) if similarities else 1.0
        
        return {
            'confidence': min(1.0, magnitude / 10.0),  # Normalize magnitude
            'multimodal_consistency': max(0.0, consistency),
            'completeness': len(context.available_modalities) / 3.0  # Fraction of available modalities
        }
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        max_val = np.max(x)
        exp_x = np.exp(x - max_val)
        return exp_x / np.sum(exp_x)

# ============================================================================
# UTILITIES & SAMPLE DATA
# ============================================================================

def create_sample_data():
    """Create sample multimodal data for testing."""
    
    # Sample text tokens (simulated)
    text_tokens = np.random.randint(0, 1000, size=50)
    
    # Sample image (RGB, 224x224)
    image_data = np.random.rand(224, 224, 3) * 255
    
    # Sample audio (1 second at 16kHz)
    audio_data = np.random.randn(16000) * 0.1
    
    return text_tokens, image_data, audio_data

def benchmark_multimodal_processing():
    """Benchmark multimodal processing performance."""
    
    base_logger.info("="*60)
    base_logger.info("MULTIMODAL PROCESSING BENCHMARK")
    base_logger.info("="*60)
    
    # Test different configurations
    fusion_strategies = ['cross_attention', 'hierarchical']
    modality_combinations = [
        ('text_only', [True, False, False]),
        ('image_only', [False, True, False]),
        ('audio_only', [False, False, True]),
        ('text_image', [True, True, False]),
        ('text_audio', [True, False, True]),
        ('image_audio', [False, True, True]),
        ('all_modalities', [True, True, True])
    ]
    
    results = {}
    
    for strategy in fusion_strategies:
        results[strategy] = {}
        processor = MultimodalProcessor(fusion_strategy=strategy)
        
        base_logger.info(f"Testing {strategy} fusion")
        
        for combo_name, (use_text, use_image, use_audio) in modality_combinations:
            # Create sample data
            text_tokens, image_data, audio_data = create_sample_data()
            
            # Prepare inputs
            text_input = text_tokens if use_text else None
            image_input = image_data if use_image else None
            audio_input = audio_data if use_audio else None
            
            # Benchmark processing
            start_time = time.time()
            context = processor.process_multimodal_context(
                text_data=text_input,
                image_data=image_input,
                audio_data=audio_input
            )
            processing_time = time.time() - start_time
            
            results[strategy][combo_name] = {
                'processing_time': processing_time,
                'modalities_used': len(context.available_modalities),
                'output_available': context.fused is not None
            }
            
            base_logger.info(f"  {combo_name:15s}: {processing_time*1000:6.2f}ms, "
                             f"modalities: {len(context.available_modalities)}")
    
    return results

def visualize_multimodal_results(benchmark_results: Dict) -> plt.Figure:
    """Visualize multimodal processing benchmark results."""
    
    strategies = list(benchmark_results.keys())
    combinations = list(benchmark_results[strategies[0]].keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Multimodal Processing Performance Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Processing time by combination
    ax = axes[0, 0]
    x_pos = np.arange(len(combinations))
    
    for i, strategy in enumerate(strategies):
        times = [benchmark_results[strategy][combo]['processing_time'] * 1000 
                for combo in combinations]
        ax.bar(x_pos + i * 0.35, times, 0.35, label=strategy, alpha=0.7)
    
    ax.set_xlabel('Modality Combination')
    ax.set_ylabel('Processing Time (ms)')
    ax.set_title('Processing Time by Modality Combination')
    ax.set_xticks(x_pos + 0.17)
    ax.set_xticklabels(combinations, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Modality count vs processing time
    ax = axes[0, 1]
    
    for strategy in strategies:
        modality_counts = []
        processing_times = []
        
        for combo in combinations:
            result = benchmark_results[strategy][combo]
            modality_counts.append(result['modalities_used'])
            processing_times.append(result['processing_time'] * 1000)
        
        ax.scatter(modality_counts, processing_times, label=strategy, s=100, alpha=0.7)
    
    ax.set_xlabel('Number of Modalities')
    ax.set_ylabel('Processing Time (ms)')
    ax.set_title('Scaling with Modality Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Strategy comparison
    ax = axes[1, 0]
    
    avg_times = {}
    for strategy in strategies:
        times = [benchmark_results[strategy][combo]['processing_time'] * 1000 
                for combo in combinations]
        avg_times[strategy] = np.mean(times)
    
    bars = ax.bar(avg_times.keys(), avg_times.values(), alpha=0.7, 
                  color=['blue', 'orange'])
    ax.set_ylabel('Average Processing Time (ms)')
    ax.set_title('Strategy Performance Comparison')
    
    # Add value labels
    for bar, value in zip(bars, avg_times.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{value:.1f}ms', ha='center', va='bottom')
    
    # Plot 4: Modality utilization
    ax = axes[1, 1]
    
    modality_usage = {'Text': 0, 'Image': 0, 'Audio': 0}
    total_tests = len(combinations) * len(strategies)
    
    for strategy in strategies:
        for combo in combinations:
            if 'text' in combo:
                modality_usage['Text'] += 1
            if 'image' in combo:
                modality_usage['Image'] += 1
            if 'audio' in combo:
                modality_usage['Audio'] += 1
    
    usage_percentages = [count / total_tests * 100 for count in modality_usage.values()]
    
    bars = ax.bar(modality_usage.keys(), usage_percentages, alpha=0.7,
                  color=['green', 'red', 'purple'])
    ax.set_ylabel('Usage Percentage (%)')
    ax.set_title('Modality Usage in Tests')
    
    # Add percentage labels
    for bar, pct in zip(bars, usage_percentages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{pct:.0f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

from typing import Any

def to_list(obj: Any) -> list:
    """Safely convert .shape (tuple or ndarray attribute) to list."""
    if hasattr(obj, "shape"):
        return list(obj.shape)
    return list(obj) if isinstance(obj, (tuple, list)) else [obj]

def example_01_individual_modality_encoding():
    example_dir = create_example_dir("example_01_individual_encoding")
    logger = get_logger("ex01", example_dir)
    logger.info("="*80)
    logger.info("Example 01: Individual Modality Encoding")
    logger.info("="*80)

    text_tokens, image_data, audio_data = create_sample_data()
    processor = MultimodalProcessor(fusion_strategy='cross_attention')

    modality_cases = {
        "text":  (Modality.TEXT,  text_tokens),
        "image": (Modality.IMAGE, image_data),
        "audio": (Modality.AUDIO, audio_data),
    }

    results = {}
    for name, (modality_enum, data) in modality_cases.items():
        context = processor.process_multimodal_context(
            text_data=text_tokens if modality_enum == Modality.TEXT else None,
            image_data=image_data if modality_enum == Modality.IMAGE else None,
            audio_data=audio_data if modality_enum == Modality.AUDIO else None,
        )

        embedding = context.get_modality_data(modality_enum)
        
        # SAFE handling: convert tuple shape to list
        shape_list = list(embedding.shape) if hasattr(embedding, 'shape') else [embedding.shape]
        
        results[name] = {
            "embedding_shape": shape_list,
            "norm": float(np.linalg.norm(embedding)),
            "processing_time_ms": context.metadata["processing_time"] * 1000
        }
        save_numpy(embedding, example_dir, f"{name}_embedding")
        logger.info(f"{name.title()} → shape: {shape_list}, "
                    f"time: {context.metadata['processing_time']*1000:.2f}ms")

    save_json(results, example_dir, "individual_results")
    logger.info("Example 01 completed")


def example_02_cross_modal_fusion():
    example_dir = create_example_dir("example_02_cross_modal_fusion")
    logger = get_logger("ex02", example_dir)
    logger.info("="*80)
    logger.info("Example 02: Cross-Modal Attention Fusion")
    logger.info("="*80)

    text_tokens, image_data, audio_data = create_sample_data()
    processor = MultimodalProcessor(fusion_strategy='cross_attention')

    context = processor.process_multimodal_context(
        text_data=text_tokens,
        image_data=image_data,
        audio_data=audio_data
    )

    result = {
        "available_modalities": [m.value for m in context.available_modalities],
        "fused_shape": to_list(context.fused),
        "fused_norm": float(np.linalg.norm(context.fused)),
        "processing_time_ms": context.metadata["processing_time"] * 1000,
        "fusion_strategy": context.metadata["fusion_strategy"]
    }
    save_json(result, example_dir, "fusion_result")
    save_numpy(context.fused, example_dir, "fused_embedding")
    save_numpy(context.text, example_dir, "text_embedding")
    save_numpy(context.image, example_dir, "image_embedding")
    save_numpy(context.audio, example_dir, "audio_embedding")

    logger.info(f"Fusion complete → modalities: {len(context.available_modalities)}, "
                f"time: {context.metadata['processing_time']*1000:.2f}ms")
    logger.info("Example 02 completed")


def example_03_multimodal_rag():
    example_dir = create_example_dir("example_03_multimodal_rag")
    logger = get_logger("ex03", example_dir)
    logger.info("="*80)
    logger.info("Example 03: Multimodal RAG System")
    logger.info("="*80)

    rag = MultimodalRAG()
    texts = [np.random.randint(0, 1000, size=np.random.randint(20, 60)) for _ in range(3)]
    for i, t in enumerate(texts):
        rag.add_to_knowledge_base(text_data=t, metadata={"type": "text", "doc_id": f"text_{i}"})
    for i in range(2):
        t, img, aud = create_sample_data()
        rag.add_to_knowledge_base(text_data=t, image_data=img, audio_data=aud,
                                  metadata={"type": "multimodal", "doc_id": f"mm_{i}"})

    query_text = np.random.randint(0, 1000, 20)
    results = rag.retrieve_relevant_context(query_text=query_text, top_k=4)

    retrieval_output = []
    for r in results:
        entry = {
            "rank": len(retrieval_output) + 1,
            "similarity": round(float(r["similarity"]), 4),
            "metadata": r["metadata"],
            "modalities": [m.value for m in r["context"].available_modalities]
        }
        retrieval_output.append(entry)
        logger.info(f"Rank {entry['rank']}: sim={entry['similarity']:.4f}, type={entry['metadata']['type']}")

    save_json(retrieval_output, example_dir, "retrieval_results")
    logger.info("Example 03 completed")


def example_04_content_analysis():
    example_dir = create_example_dir("example_04_content_analysis")
    logger = get_logger("ex04", example_dir)
    logger.info("="*80)
    logger.info("Example 04: Multimodal Content Analysis")
    logger.info("="*80)

    analyzer = MultimodalContentAnalyzer()
    text_tokens, image_data, audio_data = create_sample_data()

    cases = [
        ("text_only", text_tokens, None, None),
        ("image_only", None, image_data, None),
        ("audio_only", None, None, audio_data),
        ("all_modalities", text_tokens, image_data, audio_data)
    ]

    analysis_results = {}
    for name, t, i, a in cases:
        res = analyzer.analyze_content(text_data=t, image_data=i, audio_data=a)
        analysis_results[name] = res
        logger.info(f"{name.replace('_', ' ').title()}: "
                    f"sentiment={res['sentiment']['predicted']} "
                    f"({res['sentiment'][res['sentiment']['predicted']]:.3f}), "
                    f"safety={res['safety']['classification']}, "
                    f"confidence={res['quality_metrics']['confidence']:.3f}")

    save_json(analysis_results, example_dir, "analysis_results")
    logger.info("Example 04 completed")


def example_05_hierarchical_vs_cross_attention():
    example_dir = create_example_dir("example_05_fusion_comparison")
    logger = get_logger("ex05", example_dir)
    logger.info("="*80)
    logger.info("Example 05: Hierarchical vs Cross-Attention Fusion")
    logger.info("="*80)

    text_tokens, image_data, audio_data = create_sample_data()
    comparison = {}

    for strategy in ['cross_attention', 'hierarchical']:
        processor = MultimodalProcessor(fusion_strategy=strategy)
        start = time.time()
        ctx = processor.process_multimodal_context(text_tokens, image_data, audio_data)
        duration = (time.time() - start) * 1000
        comparison[strategy] = {
            "processing_time_ms": round(duration, 2),
            "fused_norm": float(np.linalg.norm(ctx.fused)),
            "fused_shape": to_list(ctx.fused),
            "modalities": len(ctx.available_modalities)
        }
        save_numpy(ctx.fused, example_dir, f"fused_{strategy}")
        logger.info(f"{strategy.replace('_', ' ').title()}: {duration:.2f}ms")

    save_json(comparison, example_dir, "fusion_comparison")
    logger.info("Example 05 completed")


def example_06_performance_benchmark():
    example_dir = create_example_dir("example_06_benchmark")
    logger = get_logger("ex06", example_dir)
    logger.info("="*80)
    logger.info("Example 06: Full Performance Benchmark")
    logger.info("="*80)

    results = benchmark_multimodal_processing()  # already uses logger
    save_json(results, example_dir, "benchmark_results")

    fig = visualize_multimodal_results(results)
    save_plot(fig, example_dir, "performance_dashboard")
    logger.info("Benchmark + dashboard saved")
    logger.info("Example 06 completed")


def main():
    base_logger.info("\n" + "="*80)
    base_logger.info("MULTIMODAL CONTEXT PROCESSING LAB")
    base_logger.info("All artifacts → ./generated/multimodal_lab/example_*")
    base_logger.info("="*80)

    example_01_individual_modality_encoding()
    example_02_cross_modal_fusion()
    example_03_multimodal_rag()
    example_04_content_analysis()
    example_05_hierarchical_vs_cross_attention()
    example_06_performance_benchmark()

    base_logger.info("="*80)
    base_logger.info("ALL EXAMPLES COMPLETED – 6 folders with logs, JSON, NPY, PNG")
    base_logger.info("="*80)

    base_logger.info("\nKey Takeaways:")
    base_logger.info("• Unified representation enables cross-modal understanding")
    base_logger.info("• Cross-modal attention captures modality interactions")
    base_logger.info("• Hierarchical fusion handles missing modalities gracefully")
    base_logger.info("• Production systems need caching and efficient encoding")
    base_logger.info("• Content analysis benefits from multimodal context")
    
    base_logger.info("\nPractical Applications:")
    base_logger.info("• Multimodal search and retrieval systems")
    base_logger.info("• Content moderation across text, image, and audio")
    base_logger.info("• Enhanced chatbots with multimodal understanding")
    base_logger.info("• Cross-modal content generation and editing")
    base_logger.info("• Accessibility tools for multimodal content")

if __name__ == "__main__":
    main()
