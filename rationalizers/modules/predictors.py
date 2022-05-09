import torch
from torch import nn

from rationalizers.builders import build_sentence_encoder


"""Copied from https://github.com/bastings/interpretable_predictions"""


class SentimentPredictor(nn.Module):
    """
    The Encoder takes an input text (and rationale z) and computes p(y|x,z)

    Supports a sigmoid on the final result (for regression)
    If not sigmoid, will assume cross-entropy loss (for classification)

    """

    def __init__(
        self,
        embed: nn.Embedding = None,
        hidden_size: int = 200,
        output_size: int = 1,
        dropout: float = 0.1,
        layer: str = "rcnn",
        nonlinearity: str = "sigmoid",
    ):

        super().__init__()

        emb_size = embed.weight.shape[1]

        self.embed_layer = nn.Sequential(embed, nn.Dropout(p=dropout))

        self.enc_layer = build_sentence_encoder(layer, emb_size, hidden_size)

        if hasattr(self.enc_layer, "cnn"):
            enc_size = self.enc_layer.cnn.out_channels
        else:
            enc_size = hidden_size * 2

        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(enc_size, output_size),
            nn.Sigmoid() if nonlinearity == "sigmoid" else nn.LogSoftmax(dim=-1),
        )

    def forward(self, x, z, mask=None):
        x = x.cuda()
        z = z.cuda()
        mask = mask.cuda()
        rnn_mask = mask
        emb = self.embed_layer(x)
        # apply z to main inputs
        if z is not None:
            # mask = torch.stack([mask, mask],dim=2)
            # print(mask)
            print('mask',mask.size())
            print('z.size',z.size())
            print(z)
            # z_mask = (torch.stack((mask,mask), dim = 2) * z)#.unsqueeze(-1)  # [B, T, 1]
            # print('z_mask', z_mask.size())
            # print(z_mask)
            z_mask = (mask.float() * z).unsqueeze(-1)  # [B, T, 1]
            print('**********z_mask', z_mask.size())

            rnn_mask = z_mask.squeeze(-1) > 0.0  # z could be continuous
            print('rnn_mask', rnn_mask.size())
            print('emb.size()',emb.size())
            emb = emb * z_mask

        # z is also used to control when the encoder layer is active
        lengths = mask.long().sum(1)

        # encode the sentence
        _, final = self.enc_layer(emb, rnn_mask, lengths)

        # predict sentiment from final state(s)
        y = self.output_layer(final)

        return y
