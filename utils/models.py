import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import math


class RotaryEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1e4, batch_first=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))
        pe = torch.zeros(int(max_len), 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if not self.batch_first:
            x = x + self.pe[:x.size(0)]
        else:
            x = x + self.pe[:x.size(1)].permute(1, 0, 2)
        return self.dropout(x)


class MIDIK(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = nn.ModuleDict({key: nn.Embedding(val + 1, config.token) for key, val in config.ranges.items()})

        self.squeeze = nn.Sequential(
            nn.Linear(len(config.ranges.keys()) * config.token, config.embed * 2),
            nn.ReLU(),
            nn.Linear(config.embed * 2, config.embed)
        )

        self.positional = RotaryEncoding(config.embed, max_len=4e4, batch_first=True)

        encoder = nn.TransformerEncoderLayer(config.embed, config.heads, config.feed, batch_first=True)

        self.layers = nn.ModuleList([encoder for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.embed)

        self.heads = nn.ModuleDict({key: nn.Linear(config.embed, val + 1) for key, val in config.ranges.items()})

    def forward(self, inputs, context=None):
        x = None
        for key in inputs:
            # print(key, torch.amax(inputs[key]))
            embed = self.embedding[key](inputs[key])
            if x is None:
                x = embed
            else:
                x = torch.cat([x, embed], dim=-1)

        # print(x.shape)

        x = self.squeeze(x)

        x = self.positional(x)

        def wrap(layer):
            def forward(y):
                src_mask = nn.Transformer.generate_square_subsequent_mask(y.shape[1], device=y.device)
                return layer(y, src_mask=src_mask)
            return forward

        output = x
        for l in self.layers:
            # output = cp.checkpoint(
            #     wrap(l),
            #     output,
            #     use_reentrant=True
            # )
            output = wrap(l)(output)

        x = self.norm(output)
        outputs = {key: value(x) for key, value in self.heads.items()}

        return outputs


if __name__ == "__main__":
    from data import *
    from config import *

    root = ".." if os.getcwd().endswith("utils") else ""

    config = Config().load(os.path.join(root, "configs", "config.json"))
    dataset = LakhData(config.dataset, fixMissing=False)

    batch = LakhData.collate([dataset[i] for i in range(64)])

    model = MIDIK(config.model)

    outputs = model(batch["sequences"])
