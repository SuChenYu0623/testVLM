import torch # type: ignore
import torch.nn as nn # type: ignore

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