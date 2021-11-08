import torch
from .encoder import Encoder
from .autoregressor import Autoregressor
from .infonce import InfoNCE
from .resnet import ResNetSimCLR


def l2norm(p, q):
    return torch.mean(torch.norm(p - q, dim=1))


class CPC(torch.nn.Module):
    def __init__(
        self, args, strides, filter_sizes, padding, genc_input, genc_hidden, gar_hidden,
    ):
        super(CPC, self).__init__()

        self.args = args

        """
        First, a non-linear encoder genc maps the input sequence of observations xt to a
        sequence of latent representations zt = genc(xt), potentially with a lower temporal resolution.
        """
        self.encoder = Encoder(genc_input, genc_hidden, strides, filter_sizes, padding,)

        """
        We then use a GRU RNN [17] for the autoregressive part of the model, gar with 256 dimensional hidden state.
        """
        self.autoregressor = Autoregressor(args, input_dim=genc_hidden, hidden_dim=gar_hidden)

        self.loss = InfoNCE(args, gar_hidden, genc_hidden)
        self.static_encoder = ResNetSimCLR('resnet50', 256)
        self.triplet_loss = torch.nn.TripletMarginLoss(5.0)

    def get_latent_size(self, input_size):
        x = torch.zeros(input_size).to(self.args.device)

        if self.args.fp16:
            x = x.half()

        z, c = self.get_latent_representations(x)
        return c.size(2), c.size(1)

    def get_latent_representations(self, x):
        """
        Calculate latent representation of the input with the encoder and autoregressor
        :param x: inputs (B x C x L)
        :return: loss - calculated loss
                accuracy - calculated accuracy
                z - latent representation from the encoder (B x L x C)
                c - latent representation of the autoregressor  (B x C x L)
        """

        if self.args.fp16:
            x = x.half()

        # calculate latent represention from the encoder
        z = self.encoder(x)
        z = z.permute(0, 2, 1)  # swap L and C

        # calculate latent representation from the autoregressor
        # c = self.autoregressor(z)

        # TODO checked
        return z

    def forward(self, x, anc, pos, neg):
        # x: (b, 1, 20480)
        z = self.get_latent_representations(x)
        ca, cp, cn = self.static_encoder(anc), self.static_encoder(pos), self.static_encoder(neg)
        ca, cp, cn = ca.squeeze(), cp.squeeze(), cn.squeeze()
        # print(l2norm(ca, cn) - l2norm(ca, cp), l2norm(ca, cn), l2norm(ca, cp))
        # z: (b, 128, 512) c: (b, 128, 256)
        loss, accuracy = self.loss.get(x, z, ca.unsqueeze(1).expand(-1, 128, -1))
        trpl_loss = self.triplet_loss(ca, cp, cn)
        return loss, trpl_loss, accuracy, z, ca

