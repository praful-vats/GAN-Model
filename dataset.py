import os
import numpy as np
import torch.utils.data as data
from PIL import Image

class MyDataset(data.Dataset):
    def __init__(self, dataroot):
        self.root_dir = dataroot
        self.images = [f for f in os.listdir(self.root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.images[index])
        img = Image.open(img_path).convert('RGB')
        return img

# class MyDataset(data.Dataset):
#     def __init__(self, dataroot):
#         self.root_dir = dataroot
#         self.images = os.listdir(self.root_dir)

#     def __getitem__(self, index):
#         img_path = os.path.join(self.root_dir, self.images[index])
#         if not img_path.endswith('.jpg') and not img_path.endswith('.png'):
#             return self.__getitem__(index + 1) # skip non-image files
#         image = Image.open(img_path).convert('RGB')
#         image = np.array(image)
#         return image, image

#     def __len__(self):
#         return len(self.images)
