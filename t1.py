import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from torchvision import transforms # type: ignore
from torchvision.models import vit_b_16 # type: ignore

from transformers import BertModel, BertTokenizer # type: ignore
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
    
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        # 使用 torchvision 提供的預訓練 ViT 模型
        self.vit = vit_b_16(pretrained=True)
        # 移除分類頭（實際上只取中間的特徵）
        self.vit.heads = nn.Identity()
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        x: 圖像 tensor，形狀 (B, C, H, W)
        返回: 圖像特徵，形狀 (B, embed_dim)
        """
        features = self.vit(x)  # 這裡得到的是全局特徵向量
        return features
    
class QFormer(nn.Module):
    def __init__(self, num_query_tokens=32, embed_dim=768, num_layers=4, num_heads=8):
        super().__init__()
        # 學習查詢 token (固定數量，且會在訓練中更新)
        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, embed_dim))
        # 使用 Transformer Encoder 處理查詢與圖像特徵的融合
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, img_features):
        """
        img_features: 來自 ImageEncoder 的圖像特徵，形狀 (B, embed_dim)
        返回: 查詢 token 的表示，形狀 (B, num_query_tokens, embed_dim)
        """
        B = img_features.size(0)
        # 擴展查詢 token 給每個 batch
        queries = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, num_query_tokens, embed_dim)
        
        # 這裡做一個簡單的設計：將圖像全局特徵作為額外 token 與查詢 token 串接
        img_token = img_features.unsqueeze(1)  # (B, 1, embed_dim)
        tokens = torch.cat([queries, img_token], dim=1)  # (B, num_query_tokens+1, embed_dim)
        
        # Transformer 模塊要求序列維度在第一維，因此做轉置
        tokens = tokens.transpose(0, 1)  # (S, B, embed_dim) 其中 S = num_query_tokens + 1
        tokens = self.transformer(tokens)
        tokens = tokens.transpose(0, 1)  # 回到 (B, S, embed_dim)
        
        # 返回查詢部分（去掉額外加入的圖像 token）
        return tokens[:, :queries.size(1), :]



transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 重新調整大小
    transforms.ToTensor(),  # 轉換為 Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

image_dir = 'dataset/flickr8k/images'
caption_file = 'dataset/flickr8k/captions.csv'
dataset = MyDataset(image_dir, caption_file, transform=transform, tokenizer=tokenizer)

# 建立 DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

imageEncoder = ImageEncoder()
qformer = QFormer()
# 測試讀取一個 batch
for images, input_ids in dataloader:
    print(f"Image batch shape: {images.shape}")  # (32, 3, 224, 224)
    print(f"Caption batch sample: {input_ids.shape}")  # 顯示前 3 個標註

    img_features = imageEncoder(images)
    print(f"img_features: {img_features.shape}")

    memory = qformer(img_features)
    print(f"memory: {memory.shape}")
    break  # 只跑一次
