# audioflux_audio_insights_extractor.py

from typing import Dict, Optional, Tuple
import numpy as np
import audioflux as af
from audioflux.type import (
    SpectralFilterBankScaleType,
    WaveletContinueType,
    PitchType,
    NoveltyType,
    SpectralDataType,
    WindowType,
)
from audioflux.utils import note_to_hz


class AudioInsightsExtractor:
    """
    A reusable, modular extractor for common audio insights using audioflux.
    Designed to be generic, configurable, and easy to extend without hard-coded business logic.
    """

    def __init__(
        self,
        samplate: int,
        radix2_exp: int = 12,
        slide_length: Optional[int] = None,
    ):
        self.samplate = samplate
        self.radix2_exp = radix2_exp
        self.slide_length = slide_length or (1 << radix2_exp) // 4

    def load_audio(self, path: str) -> np.ndarray:
        """Load audio file and return mono array."""
        audio_arr, sr = af.read(path)
        if sr != self.samplate:
            raise ValueError(f"Sample rate mismatch: expected {self.samplate}, got {sr}")
        return audio_arr.squeeze() if audio_arr.ndim > 1 else audio_arr

    def mel_spectrogram_and_mfcc(
        self,
        audio: np.ndarray,
        mel_num: int = 128,
        cc_num: int = 13,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Mel spectrogram and MFCC.
        Returns: (mel_spec_abs, mfcc, mel_fre_band_arr)
        """
        spec, mel_fre_band_arr = af.mel_spectrogram(
            audio, num=mel_num, radix2_exp=self.radix2_exp, samplate=self.samplate
        )
        mfcc, _ = af.mfcc(
            audio, cc_num=cc_num, mel_num=mel_num, radix2_exp=self.radix2_exp, samplate=self.samplate
        )
        return np.abs(spec), mfcc, mel_fre_band_arr

    def cwt_and_synsq(
        self,
        audio: np.ndarray,
        num: int = 84,
        low_fre: float = None,
        bin_per_octave: int = 12,
        wavelet_type: WaveletContinueType = WaveletContinueType.MORSE,
    ) -> Tuple[np.ndarray, np.ndarray, af.CWT]:
        """
        Compute CWT and its synchrosqueezed version.
        Returns: (cwt_abs, synsq_abs, cwt_obj)
        """
        low_fre = low_fre or note_to_hz("C1")
        cwt_obj = af.CWT(
            num=num,
            radix2_exp=self.radix2_exp,
            samplate=self.samplate,
            low_fre=low_fre,
            bin_per_octave=bin_per_octave,
            wavelet_type=wavelet_type,
            scale_type=SpectralFilterBankScaleType.OCTAVE,
        )
        cwt_spec = cwt_obj.cwt(audio)
        synsq_obj = af.Synsq(
            num=cwt_obj.num, radix2_exp=cwt_obj.radix2_exp, samplate=cwt_obj.samplate
        )
        synsq = synsq_obj.synsq(
            cwt_spec,
            filter_bank_type=cwt_obj.scale_type,
            fre_arr=cwt_obj.get_fre_band_arr(),
        )
        return np.abs(cwt_spec), np.abs(synsq), cwt_obj

    def cqt_and_chroma(
        self,
        audio: np.ndarray,
        num: int = 84,
    ) -> Tuple[np.ndarray, np.ndarray, af.CQT]:
        """Compute CQT and Chroma from CQT."""
        cqt_obj = af.CQT(num=num, samplate=self.samplate)
        cqt = cqt_obj.cqt(audio)
        chroma = cqt_obj.chroma(cqt)
        return np.abs(cqt), chroma, cqt_obj

    def spectral_features(
        self,
        audio: np.ndarray,
        num: int = 256,
        data_type: SpectralDataType = SpectralDataType.MAG,
    ) -> Dict[str, np.ndarray]:
        """Compute common spectral features: flatness, novelty, entropy, rms, slope."""
        bft_obj = af.BFT(
            num=num,
            samplate=self.samplate,
            radix2_exp=self.radix2_exp,
            slide_length=self.slide_length,
            data_type=data_type,
            scale_type=SpectralFilterBankScaleType.LINEAR,
        )
        spec = np.abs(bft_obj.bft(audio))
        spectral = af.Spectral(num=bft_obj.num, fre_band_arr=bft_obj.get_fre_band_arr())
        spectral.set_time_length(spec.shape[-1])

        return {
            "flatness": spectral.flatness(spec),
            "novelty": spectral.novelty(spec),
            "entropy": spectral.entropy(spec),
            "rms": spectral.rms(spec),
            "slope": spectral.slope(spec),
        }

    def pitch_yin(self, audio: np.ndarray) -> np.ndarray:
        """Estimate fundamental frequency using YIN algorithm."""
        obj = af.Pitch(pitch_type=PitchType.YIN)
        fre_arr, _, _ = obj.pitch(audio)
        fre_arr[fre_arr < 1] = np.nan
        return fre_arr

    def onset_detection(
        self,
        audio: np.ndarray,
        num: int = 128,
        novelty_type: NoveltyType = NoveltyType.FLUX,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect onsets.
        Returns: (onset_times, envelope, peak_values)
        """
        bft_obj = af.BFT(
            num=num,
            samplate=self.samplate,
            radix2_exp=11,
            slide_length=2048,
            scale_type=SpectralFilterBankScaleType.MEL,
            data_type=SpectralDataType.POWER,
        )
        spec = bft_obj.bft(audio)
        spec_db = af.utils.power_to_db(np.abs(spec))

        onset_obj = af.Onset(
            time_length=spec_db.shape[-1],
            fre_length=spec_db.shape[0],
            slide_length=bft_obj.slide_length,
            samplate=self.samplate,
            novelty_type=novelty_type,
        )
        _, evn_arr, time_arr, value_arr = onset_obj.onset(spec_db)
        return time_arr, evn_arr, value_arr

    def hpss_separation(
        self,
        audio: np.ndarray,
        h_order: int = 21,
        p_order: int = 31,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Harmonic-Percussive Source Separation."""
        hpss_obj = af.HPSS(
            radix2_exp=self.radix2_exp,
            window_type=WindowType.HAMM,
            slide_length=self.slide_length,
            h_order=h_order,
            p_order=p_order,
        )
        h, p = hpss_obj.hpss(audio)
        return h, p