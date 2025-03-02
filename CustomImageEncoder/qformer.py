import torch # type: ignore
import torch.nn as nn # type: ignore

class QFormer(nn.Module):
    def __init__(self, num_query_tokens=32, embed_dim=768, num_layers=4, num_heads=8):
        super().__init__()
        # 學習查詢 token (固定數量，且會在訓練中更新)
        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, embed_dim))
        # 使用 Transformer Encoder 處理查詢與圖像特徵的融合
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
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