import torch
from torch import nn

from .FeedbackModelBuilder import FeedbackModelBuilder
from ..logging_support import log_info


class FeedbackAttentionLadderCNN(nn.Module):
    def __init__(self, baseline_vgg19, insertion_layers, device, num_iterations=2, single_output=False):
        """
         UNet-style feedback CNN model, based on VGG19 feedforward model with symmetrical feedback path.
         Implements feedback within each block of convolutions grouped by image scale,
         and across whole model,using feedback decoder path that mirrors whole feedforward encoder path with
         forward skip connections.
         :param baseline_vgg19:
         :param device:
         :return:
         """
        super().__init__()

        self.device = device
        self.single_output = single_output
        self.insertion_layers = [] if not insertion_layers else [int(lstr) for lstr in insertion_layers.split(",")]
        log_info(f"Constructing FeedbackAttentionLadderCNN model with feedback to layers {self.insertion_layers}"
                 f" for {num_iterations} feedback iterations")

        # Encoder convolutional blocks based on VGG19 elements
        # Need to be individual class properties so they get included in backprop and optimisation?

        self.fb_module_1 = FeedbackModelBuilder.get_feedback_module(
            0 in self.insertion_layers, self.device, in_channels=3, image_size=(224, 224))

        self.enc_conv_1 = FeedbackModelBuilder.build_small_encoder_block(baseline_vgg19, 3, 64, [0, 2])

        # Max pooling between each conv block
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Feedback attention modules to insert between encoder conv blocks
        self.fb_module_2 = FeedbackModelBuilder.get_feedback_module(
            5 in self.insertion_layers, self.device, in_channels=64, image_size=(112, 112))

        self.enc_conv_2 = FeedbackModelBuilder.build_small_encoder_block(baseline_vgg19, 64, 128, [5, 7])

        self.fb_module_3 = FeedbackModelBuilder.get_feedback_module(
            10 in self.insertion_layers, self.device, in_channels=128, image_size=(56, 56))

        self.enc_conv_3 = FeedbackModelBuilder.build_large_encoder_block(128, 256, baseline_vgg19, [10, 12, 14, 16])

        self.fb_module_4 = FeedbackModelBuilder.get_feedback_module(
            19 in self.insertion_layers, self.device, in_channels=256, image_size=(28, 28))

        self.enc_conv_4 = FeedbackModelBuilder.build_large_encoder_block(256, 512, baseline_vgg19, [19, 21, 23, 25])

        self.fb_module_5 = FeedbackModelBuilder.get_feedback_module(
            28 in self.insertion_layers, self.device, in_channels=512, image_size=(14, 14))

        self.enc_conv_5 = FeedbackModelBuilder.build_large_encoder_block(512, 512, baseline_vgg19, [28, 30, 32, 34])

        # Decoder upsampling + 2x convolution modules, for each decoder group in FB path, from high to low level
        # Only include feedback convolutions that are relevant to specified feedback module insertion layers
        min_fb_layer = 999 if len(self.insertion_layers) == 0 else min(self.insertion_layers)
        self.dec_ups_conv_1 = FeedbackModelBuilder.build_optional_decoder_upsampler_2convs(min_fb_layer <= 28, 512, 512)

        # Decoder conv modules, to process result of feedback concatentation, for all but lowest-level decoder group
        self.dec_out_conv_1 = FeedbackModelBuilder.build_optional_decocder_conv(min_fb_layer <= 28, 1024, 512)

        self.dec_ups_conv_2 = FeedbackModelBuilder.build_optional_decoder_upsampler_2convs(min_fb_layer <= 19, 512, 256)
        self.dec_out_conv_2 = FeedbackModelBuilder.build_optional_decocder_conv(min_fb_layer <= 19, 512, 256)

        self.dec_ups_conv_3 = FeedbackModelBuilder.build_optional_decoder_upsampler_2convs(min_fb_layer <= 10, 256, 128)
        self.dec_out_conv_3 = FeedbackModelBuilder.build_optional_decocder_conv(min_fb_layer <= 10, 256, 128)

        self.dec_ups_conv_4 = FeedbackModelBuilder.build_optional_decoder_upsampler_2convs(min_fb_layer <= 5, 128, 64)
        self.dec_out_conv_4 = FeedbackModelBuilder.build_optional_decocder_conv(min_fb_layer <= 5, 128, 64)

        self.dec_ups_conv_5 = FeedbackModelBuilder.build_optional_decoder_upsampler_2convs(min_fb_layer == 0, 64, 3)

        # Output pooling and FC layers
        self.output_pooling = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.output_linear_layers = FeedbackModelBuilder.build_output_linear_layers(baseline_vgg19)

        self.num_iterations = num_iterations

        self.linear_embedding_layers = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False)
        )
        if baseline_vgg19 is not None:
            FeedbackModelBuilder.copy_weights(baseline_vgg19.classifier, [0, 3], self.linear_embedding_layers, [0, 3])

        # Overwrite base property with linear layers to combine outputs from N iterations
        self.output_linear_layers = nn.Sequential(
            nn.Linear(in_features=4096 * (1 + num_iterations), out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=1000, bias=True)
        )
        if baseline_vgg19 is not None:
            FeedbackModelBuilder.copy_weights(baseline_vgg19.classifier, [6], self.output_linear_layers, [3])

    @staticmethod
    def build_from_weights(device, model_weights_path):
        num_iterations = int(model_weights_path.split("-iterations")[0][-1])
        model = FeedbackAttentionLadderCNN(None, "0,5,10,19,28", device=device, num_iterations=num_iterations)

        log_info(f"Loading model weights from {model_weights_path} using device {device}")
        state_dict = torch.load(model_weights_path, map_location=device)
        model.load_state_dict(state_dict)
        return model.to(device)

    def forward(self, out):
        out.requires_grad = True

        # Reset all feedback activations
        self.reset_feedback()

        # Execute VGG encoder + feedback modules, for initial feedforward-only evaluation
        encoder_outputs = self.call_encoder(out)
        num_lin_chans = 4096

        # Aggregate final encoder output for feedforward pass + subsequent iterations with feedback included
        stacked_embeddings = torch.zeros((out.shape[0], (self.num_iterations + 1) * num_lin_chans)).to(self.device)
        stacked_embeddings[:, 0:num_lin_chans] = self.call_linear_embedding_layers(encoder_outputs[-1])

        all_feedback_activations = []
        for i in range(0, self.num_iterations):
            # Execute decoder layers in feedback path
            feedback_activations = self.call_decoder(encoder_outputs)
            all_feedback_activations.append(feedback_activations)

            # Apply decoder outputs as feedback activations
            self.apply_feedback(feedback_activations)

            # Call encoder side again with feedback enabled
            encoder_outputs = self.call_encoder(out)

            # Call output layers with last encoder output
            stacked_embeddings[:, num_lin_chans * (i + 1):num_lin_chans * (i + 2)] = \
                self.call_linear_embedding_layers(encoder_outputs[-1])

        final_out = self.output_linear_layers(stacked_embeddings)
        return final_out if self.single_output else final_out, all_feedback_activations

    def call_linear_embedding_layers(self, out):
        out = self.output_pooling(out)
        out = torch.flatten(out, 1)
        out = self.linear_embedding_layers(out)
        return out

    def reset_feedback(self):
        for fb in self.feedback_modules:
            if fb is not None:
                fb.set_feedback_activations(None)

    def apply_feedback(self, decoder_outputs):
        for i, fb in enumerate(self.feedback_modules):
            if fb is not None:
                fb.set_feedback_activations(decoder_outputs[i])

    @property
    def encoder_conv_groups(self):
        return [self.enc_conv_1, self.enc_conv_2, self.enc_conv_3, self.enc_conv_4, self.enc_conv_5]

    @property
    def feedback_modules(self):
        return [self.fb_module_1, self.fb_module_2, self.fb_module_3, self.fb_module_4, self.fb_module_5]

    @property
    def decoder_upsample_convs(self):
        return [self.dec_ups_conv_1, self.dec_ups_conv_2, self.dec_ups_conv_3, self.dec_ups_conv_4, self.dec_ups_conv_5]

    @property
    def decoder_output_convs(self):
        return [self.dec_out_conv_1, self.dec_out_conv_2, self.dec_out_conv_3, self.dec_out_conv_4]

    def call_encoder(self, out):
        encoder_outputs = []
        x = out
        for i, fb in enumerate(self.feedback_modules):
            x = x if fb is None else fb(x)
            x = self.encoder_conv_groups[i](x)
            x = self.pool(x)
            encoder_outputs.append(x)
        return encoder_outputs

    def call_decoder_block(self, idx, main_input, skip_connection):
        decoder_upsampler = self.decoder_upsample_convs[idx]
        if decoder_upsampler is None or main_input is None:
            return None, None

        dec_fb_out = decoder_upsampler(main_input)
        cat = torch.cat([dec_fb_out, skip_connection], 1)
        dec_out = self.decoder_output_convs[idx](cat)
        return dec_fb_out, dec_out

    def call_decoder(self, encoder_outputs):
        # First decoder block (the one nearest the encoder output!)
        # Inputs are last encoder out, with skip connection from previous encoder blocks's output
        dec_fb_out, dec_out = self.call_decoder_block(0, encoder_outputs[-1], encoder_outputs[-2])
        decoder_feedback_outputs = [dec_fb_out]

        # Middle decoder blocks, working from top (vs encoder out) to bottom (vs encoder in) layer
        for i in range(1, len(self.decoder_upsample_convs) - 1):
            # Call decoder block with previous decoder output, and skip-connection from relevant encoder output
            dec_fb_out, dec_out = self.call_decoder_block(i, dec_out, encoder_outputs[-(2 + i)])
            decoder_feedback_outputs.insert(0, dec_fb_out)

        # Last decoder block, call upsampler + 2 convs to generate feedback to bottom (input) layer of encoder
        last_decoder_upsampler = self.decoder_upsample_convs[-1]
        dec_fb_out = None if last_decoder_upsampler is None else last_decoder_upsampler(dec_out)
        decoder_feedback_outputs.insert(0, dec_fb_out)

        return decoder_feedback_outputs
