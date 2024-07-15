import torch
from torch import nn


class MultiplyingFeedbackAttentionModule(nn.Module):
    def __init__(self, in_channels: int, image_size: int, device):
        super().__init__()

        self.in_channels = in_channels
        self.image_size = image_size  # integer, for height and width
        h, w = image_size
        self.feedback_weights = nn.Parameter(torch.randn(in_channels, h, w).to(device), requires_grad=True)
        self.feedback_biases = nn.Parameter(torch.randn(in_channels, h, w).to(device), requires_grad=True)
        self.feedback_activations = None

    def set_feedback_activations(self, acts):
        self.feedback_activations = acts

    def forward(self, inp):
        """ Multiplies input tensor by (biased) feedback tensor """
        if self.feedback_activations is not None:
            # Reshape activations, weights and biases so they will broadcast
            # to something compatible with the input tensor shape, i.e.
            # batch size x channels x height x width
            h, w = self.image_size
            weights = self.feedback_weights.reshape(1, self.in_channels, h, w)
            biases = self.feedback_biases.reshape(1, self.in_channels, h, w)
            feedback = self.feedback_activations

            out = inp * (weights * feedback + biases)
        else:
            out = inp
        return out
