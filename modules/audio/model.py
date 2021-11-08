import torch
from .cpc import CPC


class Model(torch.nn.Module):
    def __init__(
        self, args, strides, filter_sizes, padding, genc_hidden, gar_hidden,
    ):
        super(Model, self).__init__()

        self.args = args
        self.strides = strides
        self.filter_sizes = filter_sizes
        self.padding = padding
        self.genc_input = 1
        self.genc_hidden = genc_hidden
        self.gar_hidden = gar_hidden

        self.model = CPC(
            args,
            strides,
            filter_sizes,
            padding,
            self.genc_input,
            genc_hidden,
            gar_hidden,
        )

    def forward(self, x, anc_mel, pos_mel, neg_mel):
        """Forward through the network"""

        loss, trpl_loss, recons_loss, accuracy, _, z = self.model(x, anc_mel, pos_mel, neg_mel)
        return loss, trpl_loss, recons_loss
