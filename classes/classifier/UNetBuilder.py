from torch import nn

from classes.classifier.FeedbackAttentionModule import MultiplyingFeedbackAttentionModule


class UNetBuilder:
    @staticmethod
    def get_feedback_module(included, feedback_module_type, device, in_channels, image_size):
        return None if not included else MultiplyingFeedbackAttentionModule(in_channels, image_size, device)

    @staticmethod
    def copy_weights(source_seq, source_indices, target_seq, target_indices):
        if len(source_indices) != len(target_indices):
            raise Exception("Source and target index lists must be same length.")

        for i, source_index in enumerate(source_indices):
            target_index = target_indices[i]
            source_state = source_seq[source_index].state_dict()
            target_seq[target_index].load_state_dict(source_state)  # apply weights and biases from source module

    @staticmethod
    def build_small_encoder_block(baseline_vgg19, in_channels, out_channels, weight_source_indices):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )
        UNetBuilder.copy_weights(baseline_vgg19.features, weight_source_indices, conv, [0, 2])
        return conv

    @staticmethod
    def build_large_encoder_block(in_channels, out_channels, baseline_vgg19, weight_source_indices):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )
        UNetBuilder.copy_weights(baseline_vgg19.features, weight_source_indices, conv, [0, 2, 4, 6])
        return conv

    @staticmethod
    def build_decoder_block(in_channels, internal_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(internal_channels, out_channels, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), output_padding=(1, 1)),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def build_decoder_transpose_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), output_padding=(1, 1)),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def build_decoder_upsampler_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def build_optional_decoder_upsampler_2convs(include, in_channels, out_channels):
        if include:
            internal_channels = out_channels
            return nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(in_channels, internal_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(internal_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True),
            )
        else:
            return None

    @staticmethod
    def build_decoder_conv(in_channels, out_channels, batch_norm=False):
        if batch_norm:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(out_channels),
                # https://forums.fast.ai/t/why-perform-batch-norm-before-relu-and-not-after/81293 # Andrew Ng does it!
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def build_optional_decocder_conv(include, in_channels, out_channels):
        if include:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True),
            )
        return None

    @staticmethod
    def build_optional_feedback_fc_layers(condition):
        return None if not condition else nn.Sequential(
            nn.Linear(in_features=1000, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=25088),
        )

    @staticmethod
    def build_output_linear_layers(baseline_vgg19):
        lin = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=1000, bias=True)
        )
        UNetBuilder.copy_weights(baseline_vgg19.classifier, [0, 3, 6], lin, [0, 3, 6])
        return lin

    @staticmethod
    def build_output_pooling():
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(output_size=(7, 7))
        )
