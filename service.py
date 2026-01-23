from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Annotated, Any, Dict, List, Tuple

import bentoml
from bentoml.validators import ContentType
from pydantic import Field

# -----------------------------------------------------------------------------
# Make `src/` importable inside the Bento container (Cloud Run)
# -----------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

# -----------------------------------------------------------------------------
# Optional deps
# -----------------------------------------------------------------------------
try:
    import numpy as np  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("numpy is required for this service") from e

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

try:
    from omegaconf import OmegaConf  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "omegaconf is required for this service (your repo uses Hydra/OmegaConf)."
    ) from e

# Use librosa if you install it; otherwise WAV-only fallback will be used.
try:
    import librosa  # type: ignore
except Exception:  # pragma: no cover
    librosa = None

# Your model class (must be importable in Cloud Run)
try:
    from audio_emotion.model import Model  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Could not import audio_emotion.model.Model. "
        "Make sure `src/**` is included in bentofile.yaml and that this service.py adds `src/` to sys.path."
    ) from e


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _strip_prefix(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Strip a prefix from keys if present (e.g. 'module.' or '_orig_mod.')."""
    if any(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def _to_device_str(device_env: str) -> str:
    d = (device_env or "cpu").strip()
    return d if d else "cpu"


def _torch_load_any(path: Path, device: str) -> Any:
    if torch is None:
        raise RuntimeError("Torch is not installed but a torch model is required.")
    try:
        # Safer when the file is a pure state_dict
        return torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location=device)


# -----------------------------------------------------------------------------
# Runtime image config (used by `bentoml build` / `bentoml containerize`)
# -----------------------------------------------------------------------------
bento_image = (
    bentoml.images.Image(
        python_version="3.11",
        distro="debian",
        lock_python_packages=False,
    )
    .system_packages("ffmpeg", "libsndfile1")
    .requirements_file("requirements.bento.txt")
)


@bentoml.service(
    image=bento_image,
    http={"host": "0.0.0.0", "port": int(os.environ.get("PORT", "3000"))},
)
class Service:
    """
    Audio emotion inference service.

    Endpoints:
      - POST /healthz
      - POST /metadata
      - POST /predict   (multipart: -F audio=@file.wav;type=audio/wav)
    """

    # Defaults (env vars override)
    sample_rate: int = int(os.environ.get("SAMPLE_RATE", "16000"))
    max_seconds: float = float(os.environ.get("MAX_SECONDS", "4.0"))
    n_mels: int = int(os.environ.get("N_MELS", "64"))
    n_fft: int = int(os.environ.get("N_FFT", "1024"))
    hop_length: int = int(os.environ.get("HOP_LENGTH", "256"))

    def __init__(self) -> None:
        self.device: str = _to_device_str(os.environ.get("DEVICE", "cpu"))

        # Load config before constructing Model(cfg)
        self.cfg = self._load_cfg()
        self._maybe_override_audio_params_from_cfg()

        self.labels: List[str] = self._load_labels()
        self.backend, self.model = self._load_model()

    # --------------------
    # Public APIs
    # --------------------
    @bentoml.api(route="/healthz")
    def healthz(self) -> Dict[str, str]:
        return {"status": "ok"}

    @bentoml.api
    def metadata(self) -> Dict[str, Any]:
        return {
            "labels": self.labels,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "max_seconds": self.max_seconds,
            "feature": {
                "type": "log_mel_spectrogram",
                "n_mels": self.n_mels,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
            },
            "backend": self.backend,
            "model_path": str(Path(os.environ.get("MODEL_PATH", "models/model.pt"))),
            "config_path": str(
                Path(os.environ.get("CONFIG_PATH", "configs/config.yaml"))
            ),
        }

    @bentoml.api
    def predict(
        self,
        audio: Annotated[Path, ContentType("audio/*")] = Field(
            description="Audio file upload (wav/mp3/ogg/etc.). If curl defaults to octet-stream, add ;type=audio/wav"
        ),
        top_k: int = Field(
            default=3, ge=1, le=50, description="How many labels to return"
        ),
        return_scores: bool = Field(
            default=True, description="Include per-label probabilities"
        ),
    ) -> Dict[str, Any]:
        features = self._extract_features(audio)

        if self.backend == "torch":
            scores = self._predict_torch(features)
        elif self.backend == "sklearn":
            scores = self._predict_sklearn(features)
        else:
            raise RuntimeError(f"Unsupported backend: {self.backend}")

        scores = self._to_probabilities(scores)
        top = sorted(enumerate(scores), key=lambda t: float(t[1]), reverse=True)[
            : min(top_k, len(scores))
        ]

        result: Dict[str, Any] = {
            "label": self.labels[top[0][0]] if self.labels and top else None,
            "top_k": [
                {
                    "label": self.labels[i] if i < len(self.labels) else str(i),
                    "score": float(s),
                }
                for i, s in top
            ],
        }
        if return_scores:
            result["scores"] = {
                (self.labels[i] if i < len(self.labels) else str(i)): float(scores[i])
                for i in range(len(scores))
            }
        return result

    @bentoml.api(route="/predict_npy")
    def predict_npy(
        self,
        features: Annotated[Path, ContentType("application/octet-stream")] = Field(
            description="Upload a .npy feature file (same format as training data/processed)"
        ),
        top_k: int = Field(default=3, ge=1, le=50),
        return_scores: bool = Field(default=True),
    ) -> Dict[str, Any]:
        arr = np.load(str(features), allow_pickle=False)

        # If arr is saved with extra keys (npz), you can handle it here:
        # if isinstance(arr, np.lib.npyio.NpzFile): arr = arr["arr_0"]

        # Run inference
        if self.backend == "torch":
            scores = self._predict_torch(arr)
        elif self.backend == "sklearn":
            scores = self._predict_sklearn(arr)
        else:
            raise RuntimeError(f"Unsupported backend: {self.backend}")

        scores = self._to_probabilities(scores)
        top = sorted(enumerate(scores), key=lambda t: float(t[1]), reverse=True)[
            : min(top_k, len(scores))
        ]

        result: Dict[str, Any] = {
            "label": self.labels[top[0][0]] if self.labels and top else None,
            "top_k": [
                {
                    "label": self.labels[i] if i < len(self.labels) else str(i),
                    "score": float(s),
                }
                for i, s in top
            ],
        }

        if return_scores:
            result["scores"] = {
                (self.labels[i] if i < len(self.labels) else str(i)): float(scores[i])
                for i in range(len(scores))
            }

        return result

    # --------------------
    # Config / Model loading
    # --------------------
    def _load_cfg(self):
        cfg_path = Path(os.environ.get("CONFIG_PATH", "configs/config.yaml"))
        if not cfg_path.exists():
            raise RuntimeError(
                f"Config file not found at {cfg_path}. "
                "Make sure `configs/**` is included in bentofile.yaml or set CONFIG_PATH."
            )
        return OmegaConf.load(str(cfg_path))

    def _maybe_override_audio_params_from_cfg(self) -> None:
        def env_set(name: str) -> bool:
            return name in os.environ and os.environ[name] != ""

        candidates = {
            "sample_rate": [
                "audio.sample_rate",
                "preprocess.sample_rate",
                "data.sample_rate",
            ],
            "max_seconds": [
                "audio.max_seconds",
                "preprocess.max_seconds",
                "data.max_seconds",
            ],
            "n_mels": ["audio.n_mels", "preprocess.n_mels", "feature.n_mels"],
            "n_fft": ["audio.n_fft", "preprocess.n_fft", "feature.n_fft"],
            "hop_length": [
                "audio.hop_length",
                "preprocess.hop_length",
                "feature.hop_length",
            ],
        }

        for field_name, paths in candidates.items():
            if env_set(field_name.upper()):
                continue
            for p in paths:
                v = OmegaConf.select(self.cfg, p)
                if v is not None:
                    try:
                        if field_name in {
                            "sample_rate",
                            "n_mels",
                            "n_fft",
                            "hop_length",
                        }:
                            setattr(self, field_name, int(v))
                        else:
                            setattr(self, field_name, float(v))
                    except Exception:
                        pass
                    break

    def _load_labels(self) -> List[str]:
        candidates = [
            Path("models/labels.json"),
            Path("models/label2id.json"),
            Path("models/classes.json"),
            Path("models/classes.txt"),
            Path("models/labels.txt"),
        ]
        for p in candidates:
            if p.exists():
                if p.suffix == ".json":
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if (
                        isinstance(data, dict)
                        and "labels" in data
                        and isinstance(data["labels"], list)
                    ):
                        return [str(x) for x in data["labels"]]
                    if isinstance(data, dict):
                        try:
                            items = sorted(
                                ((int(k), v) for k, v in data.items()),
                                key=lambda kv: kv[0],
                            )
                            return [str(v) for _, v in items]
                        except Exception:
                            return [str(v) for v in data.values()]
                    if isinstance(data, list):
                        return [str(x) for x in data]
                else:
                    return [
                        ln.strip()
                        for ln in p.read_text(encoding="utf-8").splitlines()
                        if ln.strip()
                    ]

        return ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]

    def _load_model(self) -> Tuple[str, Any]:
        bento_tag = os.environ.get("BENTO_MODEL_TAG")
        device = self.device

        # If you use BentoML model store (optional)
        if bento_tag:
            if torch is None:
                raise RuntimeError("BENTO_MODEL_TAG is set but torch is not installed.")
            model = bentoml.pytorch.load_model(bento_tag, device_id=device)
            model.eval()
            return "torch", model

        # Local weights file (your training saves state_dict)
        model_path = Path(os.environ.get("MODEL_PATH", "models/model.pt"))
        if not model_path.exists():
            for alt in ["models/vgg16_audio.pt", "models/model.pth"]:
                if Path(alt).exists():
                    model_path = Path(alt)
                    break

        if not model_path.exists():
            raise RuntimeError(
                f"Model file not found. Looked for {model_path}. "
                "Make sure `models/**` is included in bentofile.yaml or set MODEL_PATH."
            )

        if torch is None:
            raise RuntimeError(f"Found {model_path} but torch is not installed.")

        obj = _torch_load_any(model_path, device)

        # Full module
        if isinstance(obj, torch.nn.Module):
            obj.to(device)
            obj.eval()
            return "torch", obj

        # state_dict dict
        if isinstance(obj, dict):
            state_dict = obj.get("state_dict", obj)
            if not isinstance(state_dict, dict):
                raise RuntimeError(
                    f"{model_path} loaded as dict but has no usable state_dict. Keys: {list(obj.keys())[:30]}"
                )

            state_dict = _strip_prefix(state_dict, "module.")
            state_dict = _strip_prefix(state_dict, "_orig_mod.")

            model = Model(self.cfg).to(device)

            strict_env = os.environ.get("STRICT_LOAD", "true").strip().lower()
            strict = strict_env not in {"0", "false", "no", "off"}

            model.load_state_dict(state_dict, strict=strict)
            model.eval()
            return "torch", model

        raise RuntimeError(
            f"Unsupported torch artifact type in {model_path}: {type(obj)}. Expected dict or torch.nn.Module."
        )

    # --------------------
    # Preprocess + inference
    # --------------------
    def _read_audio(self, audio_path: Path) -> np.ndarray:
        target_len = int(self.sample_rate * self.max_seconds)

        if librosa is not None:
            y, _sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        else:
            # WAV-only fallback
            import wave

            with wave.open(str(audio_path), "rb") as wf:
                if wf.getnchannels() != 1:
                    raise RuntimeError(
                        "WAV fallback only supports mono WAV. Install librosa for more formats."
                    )
                if wf.getsampwidth() not in (2, 4):
                    raise RuntimeError(
                        "Unsupported sample width in WAV fallback. Install librosa."
                    )
                fr = wf.getframerate()
                if fr != self.sample_rate:
                    raise RuntimeError(
                        f"WAV fallback requires sample_rate={self.sample_rate}. Got {fr}. Install librosa."
                    )
                frames = wf.readframes(wf.getnframes())
                dtype = np.int16 if wf.getsampwidth() == 2 else np.int32
                y = np.frombuffer(frames, dtype=dtype).astype(np.float32)
                y /= float(np.iinfo(dtype).max)

        if y.shape[0] < target_len:
            y = np.pad(y, (0, target_len - y.shape[0]))
        else:
            y = y[:target_len]
        return y

    def _extract_features(self, audio_path: Path) -> np.ndarray:
        y = self._read_audio(audio_path)

        # If librosa isn't installed, we just return waveform (likely won't match your model!)
        if librosa is None:
            return y.astype(np.float32)

        mels = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=2.0,
        )
        logmels = np.log(mels + 1e-6).astype(np.float32)
        logmels = (logmels - logmels.mean()) / (logmels.std() + 1e-6)
        return logmels

    def _predict_torch(self, features: np.ndarray) -> np.ndarray:
        if torch is None:
            raise RuntimeError("Torch backend selected but torch is not installed.")

        x = torch.from_numpy(np.asarray(features)).float()

        # Make sure we end up with (B, C, H, W)
        if x.ndim == 2:
            # (H, W) -> (1, 1, H, W)
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            # Could be (C, H, W) or (H, W, C)
            if x.shape[0] in (1, 2, 3, 4):  # assume channels-first
                x = x.unsqueeze(0)  # (1, C, H, W)
            elif x.shape[-1] in (1, 2, 3, 4):  # channels-last
                x = x.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            else:
                # Fallback: treat as (C, H, W)
                x = x.unsqueeze(0)
        elif x.ndim == 4:
            # Already (B, C, H, W) or maybe (B, H, W, C)
            if x.shape[-1] in (1, 2, 3, 4) and x.shape[1] not in (1, 2, 3, 4):
                x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        else:
            raise RuntimeError(
                f"Unsupported feature shape for conv2d: {tuple(x.shape)}"
            )

        x = x.to(self.device)

        with torch.no_grad():
            out = self.model(x)

        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, dict):
            out = out.get("logits", out.get("output", None))
        if out is None:
            raise RuntimeError("Model output format not understood.")

        logits = out.detach().cpu().float()
        if logits.ndim == 2:
            logits = logits[0]
        return logits.numpy()

    def _predict_sklearn(self, features: np.ndarray) -> np.ndarray:
        x = features.reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x)
            return np.asarray(proba[0], dtype=np.float32)
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(x)
            return np.asarray(scores[0], dtype=np.float32)
        pred = self.model.predict(x)
        scores = np.zeros((len(self.labels),), dtype=np.float32)
        try:
            scores[int(pred[0])] = 1.0
        except Exception:
            pass
        return scores

    def _to_probabilities(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if np.all(scores >= 0) and np.isclose(scores.sum(), 1.0, atol=1e-3):
            return scores
        m = float(scores.max()) if scores.size else 0.0
        ex = np.exp(scores - m)
        s = float(ex.sum()) if ex.size else 1.0
        return ex / (s + 1e-9)
