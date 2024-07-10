import os
import random
import load
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold
from random import shuffle
from tqdm import tqdm

def classify_files(directory):
    categories = {"blink": [], "nod": [], "yawn": [],"normal":[]}
    for category in categories:
        categories[category] = []

    files = os.listdir(directory)
    for file in files:
        if file.endswith(".npy"):
            filename = os.path.splitext(file)[0]
            category = filename.split('_')[0]
            if category in categories:
                categories[category].append(file)
    return categories

def one_hot_encode(self, label, num_classes):
    one_hot = torch.zeros(num_classes)
    if label >= 0:
        one_hot[label] = 1
    return one_hot
def k_fold_split(categories, k_fold):
    k_fold_lists = []
    for i in range(k_fold):
        test_list = []
        train_list = []
        for category, files in categories.items():
            shuffle(files)
            split_size = len(files) // k_fold
            start = i * split_size
            end = (i + 1) * split_size
            test_files = files[start:end]
            test_list.extend(test_files)
            train_files = files[:start] + files[end:]
            train_list.extend(train_files)
        k_fold_lists.append((train_list, test_list))

    return k_fold_lists

class SimpleResNet(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleResNet, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


def train_model(train_loader, test_loader, num_epochs=20):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SimpleResNet()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):            
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    print(y_true,y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    classification_rep = classification_report(y_true, y_pred,zero_division=0)
    
    torch.save(model.state_dict(), 'model_weights.pth')
    # 保存整个模型
    torch.save(model, 'model_complete.pth')

    return accuracy, classification_rep


if __name__ == "__main__":
    data_path = r'D:\gy\fatigueDetection\piecesData_3'
    k_fold = 5

    categories = classify_files(data_path)
    k_fold_lists = k_fold_split(categories, k_fold)

    for i, (train_list, test_list) in enumerate(k_fold_lists):
        print(f"Iteration {i + 1}:")
        
        train_data = load.RadarDataset(data_path, train_list)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=8)

        test_data = load.RadarDataset(data_path, test_list)
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=8)

        accuracy, classification_rep = train_model(train_loader, test_loader, num_epochs=10)

        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{classification_rep}")

#torch.save(model.state_dict(), 'model_weights.pth')
