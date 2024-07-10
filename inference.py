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
from model import SimpleResNet
from load import RadarDataset
from random import shuffle
import model

def inference_model(test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    model = SimpleResNet(num_classes=4)  # 用相同的参数初始化模型
    model.load_state_dict(torch.load('model_weights.pth'))
    model.to(device)

    # 确保模型在评估模式
    model.eval()
    # 初始化一个列表来存储预测和真实标签
    predictions = []
    true_labels = []

    with torch.no_grad():  
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())

        return predictions,true_labels

def dictolist(testlistDict):
    test_list = []
    for category, files in categories.items():
        shuffle(files)
        test_list += files
    return test_list 

    # 之后您可以使用predictions和true_labels进行性能评估
if __name__ == "__main__":
    data_path = r'D:\gy\fatigueDetection\data_3\test\user3'

    categories = model.classify_files(data_path)
    
    test_list = dictolist(categories)

    test_data = RadarDataset(data_path, test_list)  # DataLoader
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    predictions, true_labels = inference_model(test_loader)
    
    print(f"Predictions: {predictions}")
    print(f"Labels:\n{true_labels}")
    
    print('111')