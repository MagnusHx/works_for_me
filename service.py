from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any, Dict, List, Tuple

import bentoml
from bentoml.validators import ContentType
from pydantic import Field

# Optional deps for audio
try:
    import librosa  # type: ignore
except Exception:  # pragma: no cover
    librosa = None

try:
    import numpy as np  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("numpy is required for this service") from e

# Optional deps for torch models
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

# Your repo's model + config loader
try:
    from omegaconf import OmegaConf  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "omegaconf is required for this service (your repo uses Hydra/OmegaConf)."
    ) from e

try:
    from audio_emotion.model import Model  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Could not import audio_emotion.model.Model. "
        "Make sure you're running from the repo root and your package is installed (e.g. `uv sync`)."
    ) from e


def _strip_prefix(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Strip a prefix from keys if present (e.g. 'module.' from DataParallel, '_orig_mod.' from torch.compile)."""
    if any(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def _torch_load_any(path: Path, device: str) -> Any:
    """
    Load torch artifacts robustly across torch versions.
    - If the artifact is a state_dict, weights_only=True is safest.
    - Some torch versions don't support weights_only -> fallback.
    """
    if torch is None:
        raise RuntimeError("Torch is not installed but a torch model is required.")

    # Prefer "weights_only=True" to avoid unpickling whole objects when possible.
    try:
        return torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location=device)


def _to_device_str(device_env: str) -> str:
    """
    Normalize DEVICE env values.
    Accepts: 'cpu', 'cuda', 'cuda:0', 'mps' (mac), etc.
    """
    d = (device_env or "cpu").strip()
    if d == "":
        return "cpu"
    return d


# ---- Runtime image for build/containerize ----
bento_image = (
    bentoml.images.Image(
        python_version="3.11",
        distro="debian",
        lock_python_packages=False,  # disables uv lock/compile behavior
    )
    .system_packages("ffmpeg", "libsndfile1")
    .requirements_file("requirements.bento.txt")
)


@bentoml.service(
    image=bento_image,
    http={"port": int(os.environ.get("PORT", "3000"))},
)
class Service:
    """
    Audio emotion inference service.

    Endpoints:
      - GET /healthz
      - GET /metadata
      - POST /predict   (multipart: -F audio=@file.wav)
    """

    # Defaults (may be overridden by config.yaml or env vars)
    sample_rate: int = int(os.environ.get("SAMPLE_RATE", "16000"))
    max_seconds: float = float(os.environ.get("MAX_SECONDS", "4.0"))
    n_mels: int = int(os.environ.get("N_MELS", "64"))
    n_fft: int = int(os.environ.get("N_FFT", "1024"))
    hop_length: int = int(os.environ.get("HOP_LENGTH", "256"))

    def __init__(self) -> None:
        self.device: str = _to_device_str(os.environ.get("DEVICE", "cpu"))

        # Load cfg first so we can (optionally) pull preprocessing params from it
        self.cfg = self._load_cfg()

        # If config contains audio/feature params, use them unless env overrides were explicitly set.
        # (This keeps your service aligned with training defaults.)
        self._maybe_override_audio_params_from_cfg()

        # Labels + model
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
            description="Audio file upload (wav/mp3/ogg/etc.)"
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

        # Normalize + top-k
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
                "Set CONFIG_PATH to the same config.yaml you trained with (needed to construct Model(cfg))."
            )
        return OmegaConf.load(str(cfg_path))

    def _maybe_override_audio_params_from_cfg(self) -> None:
        """
        Best-effort: if your config has audio/preprocess fields, use them.
        Env vars still win because they were already set at class level.
        """

        # Only override if env var wasn't explicitly set
        def env_set(name: str) -> bool:
            return name in os.environ and os.environ[name] != ""

        # Try common config paths (you can add more if your config differs)
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
        """
        Tries common label files in ./models.
        Falls back to a standard emotion set.
        """
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
        """
        Loading order:
          1) If BENTO_MODEL_TAG is set, load BentoML-stored PyTorch model (full module).
          2) Otherwise, load local ./models/model.pt/.pth
             - supports state_dict (your training saves model.state_dict())
             - supports full nn.Module if you ever switch to torch.save(model, ...)
        """
        bento_tag = os.environ.get("BENTO_MODEL_TAG")
        device = self.device

        # 1) BentoML model store (PyTorch full module)
        if bento_tag:
            if torch is None:
                raise RuntimeError("BENTO_MODEL_TAG is set but torch is not installed.")
            model = bentoml.pytorch.load_model(bento_tag, device_id=device)
            model.eval()
            return "torch", model

        # 2) Local model file (your repo saves state_dict)
        model_path = Path(os.environ.get("MODEL_PATH", "models/model.pt"))
        if not model_path.exists():
            # try common alternates
            for alt in ["models/vgg16_audio.pt", "models/model.pth"]:
                if Path(alt).exists():
                    model_path = Path(alt)
                    break

        if not model_path.exists():
            raise RuntimeError(
                f"Model file not found. Looked for {model_path} (and common alternates). "
                "Set MODEL_PATH to your weights file (e.g. models/vgg16_audio.pt)."
            )

        if torch is None:
            raise RuntimeError(f"Found {model_path} but torch is not installed.")

        obj = _torch_load_any(model_path, device)

        # Case A: someone saved full module
        if isinstance(obj, torch.nn.Module):
            obj.to(device)
            obj.eval()
            return "torch", obj

        # Case B: state_dict or checkpoint dict
        if isinstance(obj, dict):
            # some people save {"state_dict": ...}
            state_dict = obj.get("state_dict", obj)
            if not isinstance(state_dict, dict):
                raise RuntimeError(
                    f"{model_path} loaded as dict, but didn't contain a usable state_dict. Keys: {list(obj.keys())[:50]}"
                )

            state_dict = _strip_prefix(state_dict, "module.")
            state_dict = _strip_prefix(state_dict, "_orig_mod.")

            model = Model(self.cfg).to(device)

            strict_env = os.environ.get("STRICT_LOAD", "true").strip().lower()
            strict = strict_env not in {"0", "false", "no", "off"}

            try:
                model.load_state_dict(state_dict, strict=strict)
            except RuntimeError as e:
                hint = (
                    "\n\nYour weights don't match the constructed Model(cfg). Common causes:\n"
                    "  - CONFIG_PATH is not the same config used during training\n"
                    "  - model architecture changed since training\n"
                    "  - state_dict keys have unexpected prefixes\n\n"
                    "Try:\n"
                    "  - set CONFIG_PATH to the training config\n"
                    "  - or set STRICT_LOAD=false to see if it loads with missing/unexpected keys\n"
                )
                raise RuntimeError(str(e) + hint) from e

            model.eval()
            return "torch", model

        raise RuntimeError(
            f"Unsupported torch artifact type in {model_path}: {type(obj)}. "
            "Expected a state_dict dict or a torch.nn.Module."
        )

    # --------------------
    # Preprocess + inference
    # --------------------
    def _read_audio(self, audio_path: Path) -> np.ndarray:
        target_len = int(self.sample_rate * self.max_seconds)

        if librosa is not None:
            y, _sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        else:
            # WAV-only fallback (PCM)
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

        # pad/truncate
        if y.shape[0] < target_len:
            y = np.pad(y, (0, target_len - y.shape[0]))
        else:
            y = y[:target_len]
        return y

    def _extract_features(self, audio_path: Path) -> np.ndarray:
        """
        Default: log-mel spectrogram [n_mels, time]
        This should match what your training pipeline fed into Model(cfg).
        """
        y = self._read_audio(audio_path)

        if librosa is None:
            # Minimal fallback: raw waveform
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

        # normalize (per-sample)
        logmels = (logmels - logmels.mean()) / (logmels.std() + 1e-6)
        return logmels

    def _predict_torch(self, features: np.ndarray) -> np.ndarray:
        if torch is None:
            raise RuntimeError("Torch backend selected but torch is not installed.")

        # Shape input similar to training: (B, C, M, T) for spectrograms
        if features.ndim == 1:
            x = torch.from_numpy(features).float().unsqueeze(0)  # (1, T)
        else:
            x = (
                torch.from_numpy(features).float().unsqueeze(0).unsqueeze(0)
            )  # (1, 1, M, T)

        x = x.to(self.device)

        with torch.no_grad():
            out = self.model(x)

        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, dict):
            out = out.get("logits", out.get("output", None))
        if out is None:
            raise RuntimeError(
                "Model output format not understood (got dict without logits/output)."
            )

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
