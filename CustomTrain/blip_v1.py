from ..CustomDataset.customDataset import MyDataset
from ..CustomImageEncoder.vit_b_16 import ImageEncoder
from ..CustomImageEncoder.qformer import QFormer
from ..CustomTextDecoder.textDecoder import TextDecoder

from torchvision import transforms # type: ignore
from transformers import BertModel, BertTokenizer # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore

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
tokenizer = BertTokenizer.from_pretrained(weights='bert-base-uncased')


print('正式開始跑了 裡面的版本')

image_dir = 'dataset/flickr8k/images'
caption_file = 'dataset/flickr8k/captions.csv'
dataset = MyDataset(image_dir, caption_file, transform=transform, tokenizer=tokenizer)
print('dataset', dataset.__len__())

# 建立 DataLoader
dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4, drop_last=True)
print('len:', len(dataloader))

vocab_size = len(tokenizer.vocab) 
print('vocab_size', vocab_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)
model = BLIPModel(vocab_size=vocab_size)
model = model.to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)       # 損失函數
optimizer = optim.Adam(model.parameters(), lr=1e-5)  # 優化器
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.05)
num_epochs = 1
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
for epoch in range(num_epochs):
    model.train()
    # 測試讀取一個 batch
    for batch_idx, (images, input_ids) in enumerate(dataloader):
        images = images.to(device)
        captions = input_ids.to(device)
        # print(f'before into model. images: {images.shape}, captions: {captions.shape}' )
        
        optimizer.zero_grad()            # 梯度歸零
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
            print(f'Epoch [{epoch}], idx[{batch_idx}], Loss: {loss.item():.4f}')  # 顯示訓練損失
        
        if batch_idx == 500:
            pass
            # break

        
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
    print('output\n', output)

    # ---
    attention_mask = (caption != 0).float()
    # outputs = outputs.view(-1, vocab_size)         # (max_seq_length, vocab_size)
    active_loss = attention_mask.view(-1) == 1  # 獲取有效的loss位置
    active_output = output[active_loss]  # 僅保留有效的輸出
    print('output text:\n', tokenizer.decode(active_output.argmax(dim=-1), skip_special_tokens=True))
    print('3', active_output.shape)
    
    break

