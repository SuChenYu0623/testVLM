import torch.nn as nn # type: ignore
from torchvision.models import vit_b_16, ViT_B_16_Weights # type: ignore

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        # 使用 torchvision 提供的預訓練 ViT 模型
        # self.vit = vit_b_16(pretrained=True)
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
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