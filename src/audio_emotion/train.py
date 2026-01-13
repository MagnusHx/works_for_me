from audio_emotion.model import Model
from audio_emotion.data import AudioDataset

def train():
    dataset = AudioDataset("data/raw")
    waveform, label = dataset[0]
    model = Model()
    # add rest of your training code here

if __name__ == "__main__":
    train()
