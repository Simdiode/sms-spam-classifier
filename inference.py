import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.position_embedding = nn.Embedding(64, 64)

    def forward(self, x):
        positions = torch.arange(
            x.size(1),
            device=x.device
        )

        return x + self.position_embedding(positions)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.word_embedding = nn.Embedding(
            vocab_size,
            64,
            padding_idx=0
        )

        self.position_encoding = PositionalEncoding()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.classifier = nn.Linear(64, 2)

    def forward(self, input_ids):
        x = self.word_embedding(input_ids)
        x = self.position_encoding(x)

        padding_mask = input_ids == 0

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        mask = (input_ids != 0).unsqueeze(-1)

        x = x * mask
        x = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return self.classifier(x)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint = torch.load(
    "spam_transformer.pth",
    map_location=device,
    weights_only=True
)

vocab = checkpoint["vocab"]
max_length = checkpoint["max_length"]

model = TransformerModel(
    vocab_size=len(vocab)
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


def predict_sms(text):
    words = text.lower().split()

    token_ids = [
        vocab.get(word, vocab["<UNK>"])
        for word in words
    ]

    token_ids = token_ids[:max_length]

    padding_length = max_length - len(token_ids)

    token_ids += [
        vocab["<PAD>"]
    ] * padding_length

    input_tensor = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        prediction = logits.argmax(dim=1).item()

    return {
        "prediction": "spam" if prediction == 1 else "ham",
        "ham_probability": probabilities[0, 0].item(),
        "spam_probability": probabilities[0, 1].item()
    }