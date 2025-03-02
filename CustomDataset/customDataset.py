from torch.utils.data import Dataset # type: ignore
import pandas as pd # type: ignore
import os
from PIL import Image

def load_captions(caption_file):
    """
    讀取 caption.txt，返回圖片名稱到標註的映射
    """
    df = pd.read_csv(caption_file)
    samples = list(df.itertuples(index=False, name=None))  
    return samples

class MyDataset(Dataset):
    def __init__(self, image_dir, caption_file, transform=None, tokenizer=None):
        """
        image_dir: 圖片存放的資料夾路徑
        caption_file: caption.txt 的檔案路徑
        transform: 影像的預處理方法
        """
        self.image_dir = image_dir
        self.samples = load_captions(caption_file)  # 讀取所有 (image, caption) 對
        self.transform = transform
        self.tokenizer = tokenizer

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
        image_tensor = self.transform(image)
        if image_tensor is None: 
            return None

        text = caption.lower()
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=128
        )
        input_ids = encoding['input_ids'].squeeze(0)  # 去掉batch维度
        return image_tensor, input_ids