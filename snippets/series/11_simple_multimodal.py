import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTextEncoder(nn.Module):
    """簡易的なテキストエンコーダ（教育目的）"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x: [batch, seq_len] のトークンID
        emb = self.embedding(x)  # [batch, seq_len, embed_dim]
        _, (h, _) = self.lstm(emb)  # h: [2, batch, hidden_dim]
        h = torch.cat([h[0], h[1]], dim=-1)  # [batch, hidden_dim * 2]
        out = self.projection(h)  # [batch, output_dim]
        return F.normalize(out, dim=-1)  # L2正規化


class SimpleImageEncoder(nn.Module):
    """簡易的な画像エンコーダ（教育目的）"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # x: [batch, input_dim] の画像特徴
        out = self.encoder(x)
        return F.normalize(out, dim=-1)  # L2正規化


class SimpleCLIP(nn.Module):
    """簡易的なCLIPモデル（教育目的）"""

    def __init__(self, vocab_size, text_embed_dim, text_hidden_dim, image_input_dim, image_hidden_dim, shared_dim):
        super().__init__()
        self.text_encoder = SimpleTextEncoder(vocab_size, text_embed_dim, text_hidden_dim, shared_dim)
        self.image_encoder = SimpleImageEncoder(image_input_dim, image_hidden_dim, shared_dim)
        # 学習可能な温度パラメータ（log scaleで初期化）
        self.log_temperature = nn.Parameter(torch.tensor([0.07]).log())

    def forward(self, text_tokens, image_features):
        text_emb = self.text_encoder(text_tokens)
        image_emb = self.image_encoder(image_features)
        return text_emb, image_emb

    def compute_loss(self, text_tokens, image_features):
        text_emb, image_emb = self.forward(text_tokens, image_features)
        temperature = self.log_temperature.exp()

        # 類似度行列
        logits = (text_emb @ image_emb.T) / temperature

        # ラベル
        batch_size = text_emb.size(0)
        labels = torch.arange(batch_size, device=text_emb.device)

        # 対称な損失
        loss_t2i = F.cross_entropy(logits, labels)
        loss_i2t = F.cross_entropy(logits.T, labels)

        return (loss_t2i + loss_i2t) / 2


# 使用例
model = SimpleCLIP(
    vocab_size=10000,
    text_embed_dim=128,
    text_hidden_dim=256,
    image_input_dim=2048,  # 例：ResNetの特徴
    image_hidden_dim=512,
    shared_dim=256,
)

# ダミーデータ
batch_size = 16
text_tokens = torch.randint(0, 10000, (batch_size, 20))  # [batch, seq_len]
image_features = torch.randn(batch_size, 2048)  # [batch, feature_dim]

loss = model.compute_loss(text_tokens, image_features)
print(f"Loss: {loss.item():.4f}")

# 推論時：テキストと画像の類似度を計算
text_emb, image_emb = model(text_tokens, image_features)
similarity = text_emb @ image_emb.T
print(f"Similarity matrix shape: {similarity.shape}")
