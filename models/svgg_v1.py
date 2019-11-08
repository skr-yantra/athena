from .vgg_v1 import VGGNet


class SimpleVisionNet(VGGNet):

    def _make_conv_layer_builders(self):
        return [
            self._conv(32, 'conv1'),
            self._pool('pool1'),
            self._conv(64, 'conv2'),
            self._pool('pool2'),
            self._conv(128, 'conv3'),
            self._pool('pool3'),
            self._conv(256, 'conv4'),
            self._pool('pool4'),
            self._conv(512, 'conv5'),
            self._pool('pool5'),
        ]
