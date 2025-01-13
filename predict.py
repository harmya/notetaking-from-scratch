import torch
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from model import Conv1DModel 

def load_label_encoder():
    labels = ["hello", "my", "name", "is", "yes", "no", "stop", "sam"]
    le = LabelEncoder()
    le.fit(labels)
    return le

def preprocess_audio(audio_path, target_sample_rate=8000, audio_length=8000):
    samples, sample_rate = librosa.load(audio_path, sr=16000)
    samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=target_sample_rate)
    if len(samples) < audio_length:
        samples = np.pad(samples, (0, audio_length - len(samples)), mode='constant')
    elif len(samples) > audio_length:
        samples = samples[:audio_length]
    return torch.tensor(samples, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def load_model(model_path, num_labels):
    model = Conv1DModel(num_labels)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(audio_path, model, label_encoder):
    input_tensor = preprocess_audio(audio_path)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_label_idx = torch.argmax(probabilities, dim=1).item()
    predicted_label = label_encoder.inverse_transform([predicted_label_idx])[0]
    return predicted_label

if __name__ == "__main__":
    model_path = "audio_classification_model.pth"
    audio_path = "my_recording.wav"
    label_encoder = load_label_encoder()
    num_labels = len(label_encoder.classes_)
    model = load_model(model_path, num_labels)
    predicted_label = predict(audio_path, model, label_encoder)
    print(f"Predicted Label: {predicted_label}")
