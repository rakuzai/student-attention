import os
import sys
import torch
from torch.utils.data import DataLoader, random_split, Subset
from dataset import StudentAttentionDataset 
from tqdm import tqdm
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.gru import GRUModel
from models.rnn import RNNModel
from models.lstm import LSTMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def train(model, loader, optimizer, device, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for src, target in pbar:
        src, target = src.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(src)

        # 🔍 Debug print before loss
        # print(f"output dtype: {output.dtype}, shape: {output.shape}")
        # print(f"target dtype: {target.dtype}, shape: {target.shape}")

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * src.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += src.size(0)
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc

def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating", leave=False)
        for src, target in pbar:
            src, target = src.to(device), target.to(device)

            output = model(src)
            loss = criterion(output, target)

            total_loss += loss.item() * src.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += src.size(0)
            pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc

if __name__ == "__main__":
    DATA_PATH = "student_attention.pkl"
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 0.001

    with open(os.path.join(os.path.dirname(__file__), "student_attention.pkl"), "rb") as f:
        raw_data = pickle.load(f)

    full_dataset = StudentAttentionDataset(raw_data) 

    # Get all unique video names
    unique_videos = list(set(full_dataset.video_names))
    unique_videos.sort()

    print(unique_videos)

    # Split videos 80/20
    split_idx = int(0.8 * len(unique_videos))
    train_video_names = unique_videos[:split_idx]
    test_video_names = unique_videos[split_idx:]

    # Map videos to sample indices
    video_to_indices = full_dataset.video_to_sample_indices

    # Flatten all indices per split
    train_indices = [idx for vid in train_video_names for idx in video_to_indices[vid]]
    test_indices = [idx for vid in test_video_names for idx in video_to_indices[vid]]

    # Create subsets
    train_set = Subset(full_dataset, train_indices)
    test_set = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # Debug prints
    print(f"[DEBUG] Total videos        : {len(unique_videos)}")
    print(f"[DEBUG] Train video count  : {len(train_video_names)}")
    print(f"[DEBUG] Test video count   : {len(test_video_names)}")
    print(f"[DEBUG] Train sample count : {len(train_set)}")
    print(f"[DEBUG] Test sample count  : {len(test_set)}")

    model = GRUModel(input_size=34, hidden_size=128, num_classes=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train(model, train_loader, optimizer, device, criterion)
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)
        
        print(f"[Train] Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"[Test ] Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%")
    