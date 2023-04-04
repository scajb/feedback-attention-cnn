import torch
from torch import nn


# noinspection PyMethodMayBeStatic
class FeedbackAttentionModule(nn.Module):
    def __init__(self, in_channels: int, image_size: int, device):
        super(FeedbackAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.image_size = image_size  # integer, for height and width
        h, w = image_size
        self.feedback_weights = nn.Parameter(torch.randn(in_channels, h, w).to(device), requires_grad=True)
        self.feedback_biases = nn.Parameter(torch.randn(in_channels, h, w).to(device), requires_grad=True)
        self.feedback_activations = None

    def set_feedback_activations(self, acts):
        self.feedback_activations = acts

    def forward(self, inp):
        if self.feedback_activations is not None:
            # Reshape activations, weights and biases so they will broadcast
            # to something compatible with the input tensor shape, i.e.
            # batch size x channels x height x width
            h, w = self.image_size
            weights = self.feedback_weights.reshape(1, self.in_channels, h, w)
            biases = self.feedback_biases.reshape(1, self.in_channels, h, w)
            feedback = self.feedback_activations

            out = self.apply_feedback(inp, feedback, weights, biases)
        else:
            out = inp
        return out

    def apply_feedback(self, inp, feedback, weights, biases):
        """ Original additive combination of input and feedback """
        out = inp + (weights * feedback) + biases
        return out

    def multiply_feedback(self, biases, feedback, inp, weights):
        """ For use by derived multiplying attention modules. Multiplies input tensor by (biased) feedback"""
        out = inp * (weights * feedback + biases)
        return out


class MultiplyingFeedbackAttentionModule(FeedbackAttentionModule):
    def __init__(self, in_channels: int, image_size: int, device):
        super(MultiplyingFeedbackAttentionModule, self).__init__(in_channels, image_size, device)

    def apply_feedback(self, inp, feedback, weights, biases):
        return self.multiply_feedback(biases, feedback, inp, weights)


class GatedFeedbackAttentionModule(FeedbackAttentionModule):
    def __init__(self, in_channels: int, image_size: int, device):
        super(GatedFeedbackAttentionModule, self).__init__(in_channels, image_size, device)

    def apply_feedback(self, inp, feedback, weights, biases):
        """ IF feedback signal applied, ONLY returns weighted and biased feedback, with no contribution from input """
        out = (weights * feedback) + biases
        return out


class FullyConnectedFeedbackAttentionModule(nn.Module):
    def __init__(self, in_channels: int, image_size: int):
        super(FullyConnectedFeedbackAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.image_size = image_size  # integer, for height and width
        h, w = image_size
        self.linear = nn.Linear(in_features=in_channels * h * w, out_features=in_channels * h * w, bias=True)
        self.feedback_activations = None

    def set_feedback_activations(self, acts):
        self.feedback_activations = acts

    def forward(self, inp):
        if self.feedback_activations is not None:
            # Reshape activations, weights and biases so they will broadcast
            # to something compatible with the input tensor shape, i.e.
            # batch size x channels x height x width
            h, w = self.image_size
            inp_fc = inp.reshape(inp.shape[0], self.in_channels * h * w)
            out_fc = self.linear(inp_fc)
            out = out_fc.reshape(inp.shape[0], self.in_channels, h, w)
        else:
            out = inp
        return out


class ConcatenatingFeedbackAttentionModule(nn.Module):
    def __init__(self, in_channels: int, device):
        super(ConcatenatingFeedbackAttentionModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        ).to(device)
        self.feedback_activations = None

    def set_feedback_activations(self, acts):
        self.feedback_activations = acts

    def forward(self, inp):
        if self.feedback_activations is not None:
            # Concatenate input and activations, then convolve back to original number of channels
            out = torch.cat([inp, self.feedback_activations], dim=1)
            out = self.conv(out)
        else:
            out = inp
        return out

