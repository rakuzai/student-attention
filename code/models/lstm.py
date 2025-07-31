import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=34, hidden_size=128, num_layers=1, num_classes=5, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, T, 17, 2) → reshape to (B, T, 34)
        B, T, J, C = x.size()
        x = x.view(B, T, J * C)

        out, _ = self.lstm(x)               # out: (B, T, hidden)
        last_hidden = out[:, -1, :]         # last timestep (B, hidden)

        logits = self.classifier(last_hidden)  # (B, num_classes)
        return logits