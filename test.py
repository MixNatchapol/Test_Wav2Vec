import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from datasets import load_dataset, Audio

# Load your dataset (example using a hypothetical dataset)
dataset = load_dataset('your_dataset_name', split='train')

# Determine the number of unique labels
num_labels = len(set(dataset['label']))

print(f"Number of labels: {num_labels}")

# Load the pre-trained processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h", num_labels=num_labels)

# Ensure the audio data is loaded correctly
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Preprocess the dataset
def preprocess_function(examples):
    audio = examples["audio"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt", padding=True)
    inputs["labels"] = examples["label"]
    return inputs

# Apply the preprocessing function to the dataset
encoded_dataset = dataset.map(preprocess_function, remove_columns=["audio"], batched=True)

# Create PyTorch DataLoader
from torch.utils.data import DataLoader

train_loader = DataLoader(encoded_dataset, batch_size=8, shuffle=True)

# Define the Training Loop
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

# Move model to the appropriate device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = CrossEntropyLoss()

# Training loop
num_epochs = 3  # Set the number of epochs
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader):
        # Move inputs and labels to device
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)
        
        # Forward pass
        outputs = model(**inputs)
        loss = criterion(outputs.logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Evaluate the Model
# Load the validation/test set (example using a hypothetical dataset)
test_dataset = load_dataset('your_dataset_name', split='test')
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
encoded_test_dataset = test_dataset.map(preprocess_function, remove_columns=["audio"], batched=True)
test_loader = DataLoader(encoded_test_dataset, batch_size=8)

# Evaluation loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        # Move inputs and labels to device
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)
        
        # Forward pass
        outputs = model(**inputs)
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Accuracy: {accuracy:.4f}")

# Save the Fine-Tuned Model
model.save_pretrained("path_to_save_your_model")
processor.save_pretrained("path_to_save_your_model")
