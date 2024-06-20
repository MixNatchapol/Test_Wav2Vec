import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

# Define the path to the dataset
DATASET_PATH = 'snore_dataset'

# Function to load the .wav files and their labels using torchaudio
def load_data(dataset_path):
    data = []
    labels = []

    for label in ['0', '1']:
        class_path = os.path.join(dataset_path, label)
        for filename in os.listdir(class_path):
            if filename.endswith('.wav'):
                filepath = os.path.join(class_path, filename)
                waveform, sample_rate = torchaudio.load(filepath)
                data.append(waveform.mean(dim=0).numpy())  # Convert stereo to mono if necessary
                labels.append(int(label))
    
    return data, labels

# Custom Dataset class
class SnoreDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Load data
data, labels = load_data(DATASET_PATH)

# Pad sequences to the same length
max_length = max(len(sample) for sample in data)
data = [np.pad(sample, (0, max_length - len(sample)), 'constant') for sample in data]

# Normalize the data safely
data = np.array(data)
data_max = np.max(np.abs(data), axis=1, keepdims=True)
data_max[data_max == 0] = 1  # Avoid division by zero
data = data / data_max

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Create PyTorch datasets and dataloaders
train_dataset = SnoreDataset(X_train, y_train)
test_dataset = SnoreDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define a simple 1D CNN model with LeakyReLU and logits output
class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=2, padding=3)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(32 * (max_length // 16), 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)  # Logits output
        return x

# Initialize model, loss function, and optimizer
model = CNN1D()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.float().unsqueeze(1)  # Match the output shape
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = (torch.sigmoid(outputs) > 0.5).int()  # Use sigmoid to get probabilities
        total += labels.size(0)
        correct += (predicted.squeeze() == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the model
torch.save(model.state_dict(), 'snore_detection_model.pth')
