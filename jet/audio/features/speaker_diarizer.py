import os
import json
from typing import List, Optional
from omegaconf import OmegaConf
import torch
from pyannote.core import Annotation
from nemo.collections.asr.models import NeuralDiarizer
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels, labels_to_pyannote_object
from jet.logger import logger


class SpeakerDiarizer:
    """A class for speaker diarization using NVIDIA NeMo's NeuralDiarizer.

    Attributes:
        data_dir (str): Directory for storing input data.
        output_dir (str): Directory for storing output files.
        config (OmegaConf): Configuration for diarization.
    """

    def __init__(self, data_dir: str, output_dir: str, config_path: Optional[str] = None):
        """Initialize the SpeakerDiarizer.

        Args:
            data_dir: Directory for input data (manifest files, configs).
            output_dir: Directory for output files (RTTM, logs).
            config_path: Optional path to a custom configuration file.
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        self.input_manifest_file = os.path.join(
            self.data_dir, "input_manifest.json")
        self.config = self._configure(config_path)

    def _configure(self, config_path: Optional[str]) -> OmegaConf:
        """Configure the diarization process with default or custom settings.
        Args:
            config_path: Path to a custom YAML configuration file, if provided.
        Returns:
            OmegaConf: Configuration object for diarization.
        """
        default_config = {
            "device": "cpu" if not torch.cuda.is_available() else "cuda",
            "diarizer": {
                "manifest_filepath": self.input_manifest_file,
                "out_dir": self.output_dir,
                "rttm_filepath": "",  # Placeholder for RTTM file path
                "oracle_vad": False,
                "collar": 0.25,
                "ignore_overlap": True,
                "msdd_model": {
                    "model_path": "diar_msdd_telephonic",
                    "parameters": {
                        "sigmoid_threshold": [0.7, 1.0],
                        "diar_window_length": 1.5,
                        "infer_batch_size": 16,
                        "seq_eval_mode": False
                    }
                },
                "speaker_embeddings": {
                    "model_path": "titanet_large",
                    "parameters": {
                        "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                        "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
                        "multiscale_weights": [1, 1, 1, 1, 1],
                        "save_embeddings": False
                    }
                },
                "vad": {
                    "model_path": "vad_multilingual_marblenet",
                    "parameters": {
                        "onset": 0.8,
                        "offset": 0.6,
                        "pad_offset": -0.05
                    }
                },
                "clustering": {
                    "parameters": {
                        "oracle_num_speakers": False,
                        "max_num_speakers": 8
                    }
                }
            },
            "num_workers": 1
        }
        config = OmegaConf.create(default_config)
        if config_path and os.path.exists(config_path):
            custom_config = OmegaConf.load(config_path)
            config = OmegaConf.merge(config, custom_config)
        logger.debug(f"Loaded configuration: {OmegaConf.to_yaml(config)}")
        logger.info("Diarization configuration loaded")
        return config

    def create_manifest(self, audio_path: str, num_speakers: Optional[int] = None) -> dict:
        """Create a manifest file for diarization.

        Args:
            audio_path: Path to the input audio file.
            num_speakers: Optional number of speakers, if known.

        Returns:
            dict: Manifest data for the audio file.
        """
        audio_file_name_no_ext = os.path.splitext(
            os.path.basename(audio_path))[0]
        rttm_file = os.path.join(
            self.output_dir, f"pred_rttms/{audio_file_name_no_ext}.rttm")

        manifest = {
            "audio_filepath": audio_path,
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": num_speakers,
            "rttm_filepath": rttm_file,
            "uem_filepath": None
        }

        os.makedirs(os.path.dirname(rttm_file), exist_ok=True)
        with open(self.input_manifest_file, "w") as fp:
            json.dump(manifest, fp)
            fp.write("\n")

        logger.info(f"Manifest created for {audio_path}")
        return manifest

    def diarize(self, audio_path: str, num_speakers: Optional[int] = None) -> Annotation:
        """Perform speaker diarization on the given audio file.
        Args:
            audio_path: Path to the input audio file.
            num_speakers: Optional number of speakers, if known.
        Returns:
            Annotation: Diarization result in pyannote Annotation format.
        """
        manifest = self.create_manifest(audio_path, num_speakers)
        # Update config with rttm_filepath from manifest
        OmegaConf.update(self.config, "diarizer.rttm_filepath",
                         manifest["rttm_filepath"])
        logger.debug(
            f"Updated config with rttm_filepath: {self.config.diarizer.rttm_filepath}")
        logger.debug(
            f"Initializing NeuralDiarizer with config: {OmegaConf.to_yaml(self.config)}")
        model = NeuralDiarizer(cfg=self.config).to(self.config.device)
        logger.debug(
            f"NeuralDiarizer initialized on device: {self.config.device}")
        logger.info("Starting neural diarization")
        model.diarize()
        logger.info("Neural diarization completed")
        pred_labels = rttm_to_labels(manifest["rttm_filepath"])
        logger.debug(f"Predicted labels: {pred_labels}")
        hypothesis = labels_to_pyannote_object(pred_labels)
        hypothesis.uri = os.path.splitext(os.path.basename(audio_path))[0]
        return hypothesis
