import torch
import torch.nn as nn
import torch.utils.checkpoint as cp


class MIDIK(nn.Module):
    def __init__(self, config):
        super().__init__()

        encoder = nn.TransformerEncoderLayer(config.embed, config.heads, config.feed, batch_first=True)

        self.embedding = nn.ModuleDict({key: nn.Embedding(val, config.token) for key, val in config.ranges.items()})

        self.positional = nn.Identity()

        self.layers = nn.ModuleList([encoder for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.embed)

        self.heads = nn.ModuleDict({key: nn.Linear(config.embed, val) for key, val in config.ranges.items()})

    def forward(self, inputs):
        sequences, mask = inputs["sequences"], inputs["mask"]

        x = None
        for key in sequences:
            embed = self.embedding[key](sequences[key])
            if x is None:
                x = embed
            else:
                x = torch.cat([x, embed], dim=-1)

        x = self.positional(x)

        def wrap(layer):
            def forward(y, src_kpm):
                src_mask = nn.Transformer.generate_square_subsequent_mask(y.shape[1], dtype=torch.bool)
                return layer(y, src_mask=src_mask, src_key_padding_mask=src_kpm)
            return forward

        output = x
        for l in self.layers:
            output = cp.checkpoint(
                wrap(l),
                output,
                mask,
                use_reentrant=True
            )

        x = self.norm(output)
        outputs = {key: value(x) for key, value in self.heads.items()}

        return outputs


if __name__ == "__main__":
    from data import *

    config = Config().load(os.path.join("..", "configs", "config.json"))
    dataset = LakhData(config.dataset)
    batch = LakhData.collate([dataset[i] for i in range(32)])

    model = MIDIK(config.model)

    outputs = model(batch)
