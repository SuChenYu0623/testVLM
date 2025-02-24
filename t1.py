import torch # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from torchvision import transforms # type: ignore

import os
import pandas as pd # type: ignore
from PIL import Image


def load_captions(caption_file):
    """
    讀取 caption.txt，返回圖片名稱到標註的映射
    """
    df = pd.read_csv(caption_file)
    samples = list(df.itertuples(index=False, name=None))  
    return samples

class MyDataset(Dataset):
    def __init__(self, image_dir, caption_file, transform=None):
        """
        image_dir: 圖片存放的資料夾路徑
        caption_file: caption.txt 的檔案路徑
        transform: 影像的預處理方法
        """
        self.image_dir = image_dir
        self.samples = load_captions(caption_file)  # 讀取所有 (image, caption) 對
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        根據索引 idx 獲取對應的圖像與標註
        """
        image_name, caption = self.samples[idx]  # 直接從 (image, caption) 列表中取值
        image_path = os.path.join(self.image_dir, image_name)  # 完整圖片路徑

        # 讀取圖片
        image = Image.open(image_path).convert("RGB")

        # 應用 transform
        if self.transform:
            image = self.transform(image)

        return image, caption

captions = load_captions('dataset/caption.csv')
print(captions)


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 重新調整大小
    transforms.ToTensor(),  # 轉換為 Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
])

image_dir = 'dataset/source_images'
caption_file = 'dataset/caption.csv'
dataset = MyDataset(image_dir, caption_file, transform=transform)

# 建立 DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# 測試讀取一個 batch
for images, captions in dataloader:
    print(f"Image batch shape: {images.shape}")  # (32, 3, 224, 224)
    print(f"Caption batch sample: {captions[:3]}")  # 顯示前 3 個標註
    break  # 只跑一次
