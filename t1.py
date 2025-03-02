import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
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

class TextDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, num_layers=6, num_heads=8, max_seq_len=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, tgt_tokens, memory, tgt_mask=None, tgt_key_padding_mask=None):
        """
        tgt_tokens: 目標文本 token id，形狀 (B, seq_len)
        memory: 來自 Q-Former 的跨模態表示，形狀 (B, mem_len, embed_dim)
        返回: 預測的 logits，形狀 (B, seq_len, vocab_size)
        """
        B, seq_len = tgt_tokens.size()
        positions = torch.arange(0, seq_len, device=tgt_tokens.device).unsqueeze(0).expand(B, seq_len)
        tgt_embeddings = self.token_embedding(tgt_tokens) + self.pos_embedding(positions)
        
        # Transformer Decoder 要求輸入形狀 (seq_len, B, embed_dim)
        tgt_embeddings = tgt_embeddings.transpose(0, 1)
        # 將 memory 轉置至 (mem_len, B, embed_dim)
        memory = memory.transpose(0, 1)
        
        decoder_output = self.transformer_decoder(
            tgt_embeddings, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        decoder_output = decoder_output.transpose(0, 1)  # (B, seq_len, embed_dim)
        logits = self.fc_out(decoder_output)
        return logits

class BLIPModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, num_query_tokens=32):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.qformer = QFormer(num_query_tokens=num_query_tokens, embed_dim=embed_dim)
        self.text_decoder = TextDecoder(vocab_size=vocab_size, embed_dim=embed_dim)
    
    def forward(self, images, tgt_tokens, tgt_mask=None, tgt_key_padding_mask=None):
        """
        images: 輸入圖像，形狀 (B, C, H, W)
        tgt_tokens: 目標文本 token id，形狀 (B, seq_len)
        返回: 文本生成 logits，形狀 (B, seq_len, vocab_size)
        """
        # 步驟1：提取圖像特徵
        # print(f'It is in model.  images: {images.shape}, tgt_tokens: {tgt_tokens.shape}')
        img_features = self.image_encoder(images)
        # print(f'after image encoder. img_features: {img_features.shape}')

        # 步驟2：利用 Q-Former 生成跨模態查詢表示
        memory = self.qformer(img_features)
        # print(f'after qformer. memory: {memory.shape}')

        # 步驟3：透過 Text Decoder 生成文本 logits
        # print(f'before text decoder. tgt_tokens: {tgt_tokens.shape}, memory: {memory.shape}, tgt_key_padding_mask: {tgt_key_padding_mask}')
        logits = self.text_decoder(tgt_tokens, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        # print(f'after text decoder. logits: {logits.shape}')

        return logits

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 重新調整大小
    transforms.ToTensor(),  # 轉換為 Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print('正式開始跑了')

image_dir = 'dataset/flickr8k/images'
caption_file = 'dataset/flickr8k/captions.csv'
dataset = MyDataset(image_dir, caption_file, transform=transform, tokenizer=tokenizer)
print('dataset', dataset.__len__())

# 建立 DataLoader
dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4, drop_last=True)
print('len:', len(dataloader))

vocab_size = len(tokenizer.vocab) 
print('vocab_size', vocab_size)
imageEncoder = ImageEncoder()
qformer = QFormer(num_query_tokens=32, embed_dim=768)
textDecoder = TextDecoder(vocab_size=vocab_size, embed_dim=768, max_seq_len=128)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)
model = BLIPModel(vocab_size=vocab_size)
model = model.to(device)
criterion = nn.CrossEntropyLoss()       # 損失函數
optimizer = optim.Adam(model.parameters(), lr=1e-5)  # 優化器
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.05)
num_epochs = 50
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
for epoch in range(num_epochs):
    model.train()
    # 測試讀取一個 batch
    for batch_idx, (images, input_ids) in enumerate(dataloader):
        '''
        print(f"Image batch shape: {images.shape}")  # (32, 3, 224, 224)
        print(f"Caption batch sample: {input_ids.shape}")  # 顯示前 3 個標註

        img_features = imageEncoder(images)
        print(f"img_features: {img_features.shape}")

        memory = qformer(img_features)
        print(f"memory: {memory.shape}")

        tgt_mask = None
        tgt_key_padding_mask = None
        tgt_tokens = input_ids
        logits = textDecoder(tgt_tokens, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        print(f'logits: {logits.shape}')
        '''

        images = images.to(device)
        captions = input_ids.to(device)
        # print(f'before into model. images: {images.shape}, captions: {captions.shape}' )
        outputs = model(images, captions)  # 前向傳播
        attention_mask = (captions != 0).float()

        outputs = outputs.view(-1, vocab_size)         # (max_seq_length, vocab_size)
        targets = captions.view(-1)                    # (max_seq_length)
                
        # loss = criterion(outputs, targets)            # 計算損失
        # 使用遮罩過濾有效的 targets
        active_loss = attention_mask.view(-1) == 1  # 獲取有效的loss位置
        active_output = outputs[active_loss]  # 僅保留有效的輸出
        active_targets = targets[active_loss]  # 僅保留有效的targets

        loss = criterion(active_output, active_targets)  # 計算損失
        loss.backward()                               # 反向傳播
        optimizer.step()                              # 更新權重

        if batch_idx % 100 == 0:
            pass
            # print(f'Epoch [{epoch}], idx[{batch_idx}], Loss: {loss.item():.4f}')  # 顯示訓練損失

        
        # break  # 只跑一次
    print(f'Epoch [{epoch}], Loss: {loss.item():.4f}')  # 顯示訓練損失
    scheduler.step()


print('===== generate text =====')
# 生成
model.eval()
for batch_idx, (images, captions) in enumerate(dataloader):
    print('images & captions:', images.shape, captions.shape)

    # ---
    caption = captions[0]
    print('origin_text:', tokenizer.decode(caption.squeeze(0), skip_special_tokens=True))
    print('1', caption.squeeze(0).shape)
    images = images.to(device)
    captions = captions.to(device)
    outputs = model(images, captions)
    print('output:', outputs.shape)

    # ---
    output = outputs[0]
    print(output.shape)
    print('output text:', tokenizer.decode(output.argmax(dim=-1), skip_special_tokens=True))
    print('2', output.argmax(dim=-1).shape)

    # ---
    attention_mask = (caption != 0).float()
    # outputs = outputs.view(-1, vocab_size)         # (max_seq_length, vocab_size)
    active_loss = attention_mask.view(-1) == 1  # 獲取有效的loss位置
    active_output = output[active_loss]  # 僅保留有效的輸出
    print('output text:\n', tokenizer.decode(active_output.argmax(dim=-1), skip_special_tokens=True))
    print('3', active_output.shape)
    
    break