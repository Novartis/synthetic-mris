from model_architecture.medicalnet.resnet import BasicBlock, Bottleneck, ResNet


class ResNetBackBone(ResNet):
    """A backbone version of the MedicalNet ResNet producing the final activation layer prior to segmentation.
    This can be used for dimensionality reduction or FID score computation.

    """

    def __init__(self, block, layers, sample_input_D, sample_input_H, sample_input_W, shortcut_type="B", no_cuda=False):
        super().__init__(
            block=block,
            layers=layers,
            sample_input_D=sample_input_D,
            sample_input_H=sample_input_H,
            sample_input_W=sample_input_W,
            num_seg_classes=1,
            shortcut_type=shortcut_type,
            no_cuda=no_cuda,
        )

        del self.conv_seg

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet10_backbone(**kwargs):
    """Constructs a ResNet-18 model backbone."""
    model = ResNetBackBone(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18_backbone(**kwargs):
    """Constructs a ResNet-18 model backbone."""
    model = ResNetBackBone(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34_backbone(**kwargs):
    """Constructs a ResNet-34 model backbone."""
    model = ResNetBackBone(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50_backbone(**kwargs):
    """Constructs a ResNet-50 model backbone."""
    model = ResNetBackBone(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101_backbone(**kwargs):
    """Constructs a ResNet-101 model backbone."""
    model = ResNetBackBone(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152_backbone(**kwargs):
    """Constructs a ResNet-152 model backbone."""
    model = ResNetBackBone(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200_backbone(**kwargs):
    """Constructs a ResNet-200 model backbone."""
    model = ResNetBackBone(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
