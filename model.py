import os
import librosa   
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

example_audio = 'audio/train/hello'
samples, sample_rate = librosa.load(example_audio+'yes/0a7c2a8d_nohash_0.wav', sr = 16000)

ipd.Audio(samples, rate=sample_rate)
print(sample_rate)

samples = librosa.resample(samples, sample_rate, 5000)
ipd.Audio(samples, rate=5000)

labels=os.listdir(train_audio_path)
all_wave = []
all_label = []
for label in labels:
    print(label)
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr = 16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        if(len(samples)== 8000) : 
            all_wave.append(samples)
            all_label.append(label)



le = LabelEncoder()
y=le.fit_transform(all_label)
classes= list(le.classes_)


X_train, X_test, y_train, y_test = train_test_split(all_wave, y, test_size=0.2, random_state=42, stratify=y)

class AudioDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


train_dataset = AudioDataset(X_train, y_train)
test_dataset = AudioDataset(X_test, y_test)

class Conv1DModel(nn.Module):
    def __init__(self, num_labels):
        super(Conv1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=13, stride=1, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=3)
        self.drop1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=11, stride=1, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=3)
        self.drop2 = nn.Dropout(0.3)
        
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=9, stride=1, padding=0)
        self.pool3 = nn.MaxPool1d(kernel_size=3)
        self.drop3 = nn.Dropout(0.3)
        
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=0)
        self.pool4 = nn.MaxPool1d(kernel_size=3)
        self.drop4 = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(64 * 26, 256)  
        self.drop_fc1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.drop_fc2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, num_labels)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.drop3(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool4(x)
        x = self.drop4(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop_fc1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop_fc2(x)
        
        x = self.fc3(x)
        x = F.softmax(x, dim=1)  
        return x

num_labels = len(labels)  
model = Conv1DModel(num_labels)

print(model)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

def evaluate_model(model, test_loader, criterion, device):
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f"Test Loss: {total_loss / len(test_loader)}, Accuracy: {accuracy * 100:.2f}%")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
train_model(model, train_loader, criterion, optimizer, num_epochs, device)

evaluate_model(model, test_loader, criterion, device)

torch.save(model.state_dict(), "audio_classification_model.pth")

