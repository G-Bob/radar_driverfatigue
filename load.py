import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import numpy as np
from sklearn.preprocessing import normalize
from scipy.signal.windows import hamming
from scipy.fftpack import fft, fft2
from scipy.ndimage import gaussian_filter
class RadarDataset(Dataset):
    def __init__(self, data_path,data_list,num_classes=4):
        self.data_paths = data_path
        self.data_list = data_list
        self.num_classes = num_classes
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if idx >= len(self.data_list):
            raise IndexError("Index out of range")
        # print(len(self.data_list))
        data_file = os.path.join(self.data_paths,self.data_list[idx])
        data = np.load(data_file)


        mean_along_first_dimension = np.mean(data, axis=0)
        data = mean_along_first_dimension.reshape(900,40)
        abs_data = np.abs(data)
        norm_data = normalize(abs_data)
        data = norm_data.reshape(3,300,40)


        data = torch.tensor(data)
        data = data.type(torch.FloatTensor)
        label = self.get_label_from_path(self,data_file)
        return data, label

    @staticmethod
    def get_label_from_path(self,path):
        filename = os.path.basename(path)
        label_str = filename.split('_')[0]
        if label_str == 'normal':
            return 0
        elif label_str == 'blink':
            return 1
        elif label_str == 'nod':
            return 2
        elif label_str == 'yawn':
            return 3
        else:
            return -1

