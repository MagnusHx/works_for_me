from pathlib import Path
from torch.utils.data import Dataset
import torchaudio
import librosa 
import numpy as np
import wave

from audio_emotion.download import download_audio_emotions

class AudioDataset(Dataset):
    def __init__(self, data_dir: Path) -> None:
        self.data_path = download_audio_emotions(data_dir)
        self.audio_files = sorted(self.data_path.rglob("*.wav"))

        if not self.audio_files:
            raise RuntimeError(
                f"No .wav files found after download in {self.data_path}"
            )

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, index: int):
        audio_path = self.audio_files[index]
        waveform, sample_rate = torchaudio.load(audio_path)
        label = audio_path.parent.name
        return waveform, label

    def download_raw(self, output_folder: Path) -> None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        print(f"Preprocessing data from {self.data_path} to {output_folder}")

    def preprocess(
        self,
        wav_path: str | Path,
        sample_rate: int = 16000,
        clip_seconds: float = 4.0,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int | None = 400,
        center: bool = True,
        epsilon: float = 1e-8,
    ) -> np.ndarray:
        """Preprocess an audio file into a log-magnitude STFT spectrogram for the CNN.

        Loads an audio file, pads or truncates it to a fixed length, computes the
        Short-Time Fourier Transform (STFT), and returns a normalized log-magnitude
        spectrogram.

        Args:
            wav_path: Path to the audio file (.wav) to preprocess.
            sample_rate: Target sample rate in Hz for resampling the audio. Defaults to 16000.
            clip_seconds: Duration in seconds to extract from the audio. Audio shorter than
                this will be zero-padded; longer audio will be truncated. Defaults to 4.0.
            n_fft: FFT window size for STFT computation. Determines frequency resolution.
                Output will have n_fft//2 + 1 frequency bins. Defaults to 512.
            hop_length: Number of samples between successive STFT frames. Smaller values
                give higher time resolution but more frames. Defaults to 160.
            win_length: Window length for each STFT frame. If None, defaults to n_fft.
                Defaults to 400.
            center: If True, pads the signal so frames are centered. Defaults to True.
            epsilon: Small constant added before log to avoid log(0). Defaults to 1e-8.

        Returns:
            A numpy array of shape (1, F, T) containing the normalized log-magnitude
            spectrogram, where F = n_fft//2 + 1 is the number of frequency bins and
            T is the number of time frames.
        """
        y, sr = librosa.load(
            wav_path,
            sr=sample_rate,
            mono=True,
            offset=0.0,
            duration=clip_seconds,
            dtype=np.float32,
        )

        target_len = int(sr * clip_seconds)
        if y.shape[0] < target_len:
            y = np.pad(y, (0, target_len - y.shape[0]), mode="constant")
        elif y.shape[0] > target_len:
            y = y[:target_len]

        S = librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hann",
            center=center,
        )

        mag = np.abs(S)
        log_mag = np.log(mag + epsilon)

        log_mag = (log_mag - log_mag.mean()) / (log_mag.std() + 1e-6)

        return log_mag[np.newaxis, :, :].astype(np.float32)

    def send_to_processed(self, output_folder: str | Path = "data/processed") -> Path:
        """Preprocess all audio files and save spectrograms to the processed data folder.

        Iterates through all audio files in the dataset, preprocesses each one into
        a log-magnitude spectrogram, and saves them as .npy files in the specified
        output folder, preserving the emotion label subdirectory structure.

        Args:
            output_folder: Path to the output directory for processed spectrograms.
                Defaults to "data/processed".

        Returns:
            Path to the output folder containing processed spectrograms.
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        for audio_file in self.audio_files:
            label = audio_file.parent.name
            label_folder = output_folder / label
            label_folder.mkdir(parents=True, exist_ok=True)

            spectrogram = self.preprocess(audio_file)

            output_path = label_folder / f"{audio_file.stem}.npy"
            np.save(output_path, spectrogram)

        print(f"Processed {len(self.audio_files)} files to {output_folder}")
        return output_folder