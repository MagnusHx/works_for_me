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


@bentoml.service(http={"port": int(os.environ.get("PORT", "3000"))})
class Service:
    """
    Audio emotion inference service.

    Endpoints (default routes):
      - GET/POST /healthz
      - POST /predict   (multipart file upload: audio=<file>)
      - GET/POST /metadata
    """

    # Sensible defaults; override with env vars if you want
    sample_rate: int = int(os.environ.get("SAMPLE_RATE", "16000"))
    max_seconds: float = float(os.environ.get("MAX_SECONDS", "4.0"))
    n_mels: int = int(os.environ.get("N_MELS", "64"))
    n_fft: int = int(os.environ.get("N_FFT", "1024"))
    hop_length: int = int(os.environ.get("HOP_LENGTH", "256"))

    def __init__(self) -> None:
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
            "sample_rate": self.sample_rate,
            "max_seconds": self.max_seconds,
            "feature": {
                "type": "log_mel_spectrogram",
                "n_mels": self.n_mels,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
            },
            "backend": self.backend,
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
        top = sorted(
            enumerate(scores),
            key=lambda t: float(t[1]),
            reverse=True,
        )[: min(top_k, len(scores))]

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
    # Model loading
    # --------------------
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
                    # allow {"labels":[...]} or {"0":"neutral",...} or ["neutral",...]
                    if (
                        isinstance(data, dict)
                        and "labels" in data
                        and isinstance(data["labels"], list)
                    ):
                        return [str(x) for x in data["labels"]]
                    if isinstance(data, dict):
                        # sort by int key if possible
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

        # fallback
        return ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]

    def _load_model(self) -> Tuple[str, Any]:
        """
        Loading order:
          1) If BENTO_MODEL_TAG is set, try to load a BentoML-stored PyTorch model.
          2) Otherwise, try common local filenames in ./models.
        """
        bento_tag = os.environ.get("BENTO_MODEL_TAG")
        device = os.environ.get("DEVICE", "cpu")

        # 1) BentoML model store (PyTorch)
        if bento_tag:
            if torch is None:
                raise RuntimeError("BENTO_MODEL_TAG is set but torch is not installed.")
            try:
                model = bentoml.pytorch.load_model(bento_tag, device_id=device)
                model.eval()
                return "torch", model
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load BentoML PyTorch model '{bento_tag}'. "
                    f"Unset BENTO_MODEL_TAG to load from ./models instead."
                ) from e

        # 2) Local files
        models_dir = Path("models")
        if not models_dir.exists():
            raise RuntimeError(
                "No ./models directory found. Either add your model files under ./models "
                "or set BENTO_MODEL_TAG to load from BentoML model store."
            )

        # PyTorch: model.pt / model.pth
        for name in [
            "model.pt",
            "model.pth",
            "best.pt",
            "best.pth",
            "checkpoint.pt",
            "checkpoint.pth",
        ]:
            p = models_dir / name
            if p.exists():
                if torch is None:
                    raise RuntimeError(f"Found {p} but torch is not installed.")
                obj = torch.load(p, map_location=device)
                # support either a full nn.Module or a state_dict checkpoint
                if isinstance(obj, torch.nn.Module):
                    model = obj
                elif isinstance(obj, dict) and "state_dict" in obj:
                    raise RuntimeError(
                        f"{p} looks like a Lightning/Checkpoint dict (has 'state_dict'). "
                        "Update _load_model() to construct your model class and load the state_dict."
                    )
                else:
                    raise RuntimeError(
                        f"{p} isn't a torch.nn.Module. Update _load_model() to match your training artifact."
                    )
                model.eval()
                return "torch", model

        # Sklearn: model.pkl
        for name in ["model.pkl", "model.joblib", "clf.pkl", "classifier.pkl"]:
            p = models_dir / name
            if p.exists():
                try:
                    import joblib  # type: ignore
                except Exception as e:
                    raise RuntimeError(f"Found {p} but joblib is not installed.") from e
                model = joblib.load(p)
                return "sklearn", model

        raise RuntimeError(
            "No supported model file found in ./models. "
            "Add one of: model.pt/model.pth (torch) or model.pkl (sklearn), "
            "or set BENTO_MODEL_TAG to load from BentoML model store."
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
        Adjust this to match your model’s expected input.
        """
        y = self._read_audio(audio_path)

        if librosa is None:
            # Minimal fallback: raw waveform
            # Many models won't work with this; install librosa or adapt to your pipeline.
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

        # If you fell back to raw waveform, make it 2D-ish
        if features.ndim == 1:
            x = torch.from_numpy(features).float().unsqueeze(0)  # (1, T)
        else:
            # common CNN audio shape: (B, C, M, T)
            x = torch.from_numpy(features).float().unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            out = self.model(x)

        # Handle common patterns: tensor, (tensor,), {"logits": tensor}
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
        # last resort
        pred = self.model.predict(x)
        # convert single class id to one-hot-ish
        scores = np.zeros((len(self.labels),), dtype=np.float32)
        try:
            scores[int(pred[0])] = 1.0
        except Exception:
            pass
        return scores

    def _to_probabilities(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float32)
        if scores.ndim != 1:
            scores = scores.reshape(-1)

        # if already looks like probs
        if np.all(scores >= 0) and np.isclose(scores.sum(), 1.0, atol=1e-3):
            return scores

        # softmax
        m = float(scores.max()) if scores.size else 0.0
        ex = np.exp(scores - m)
        s = float(ex.sum()) if ex.size else 1.0
        return ex / (s + 1e-9)
