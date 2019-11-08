from ray.rllib.models import Model
from ray.rllib.utils import try_import_tf

tf = try_import_tf()


class VGGNet(Model):

    def __init__(self, *args, network_name='vgg_v1', **kwargs):
        self.conv_params = dict(
            padding='same',
            activation=tf.nn.relu,
            kernel_size=3,
            strides=(1, 1),
        )

        self.max_pool_params = dict(
            pool_size=2,
            strides=2,
        )

        self._name = network_name

        super(VGGNet, self).__init__(*args, **kwargs)

    def _conv(self, filters, name):
        def builder(inputs):
            return tf.layers.conv2d(inputs, **self.conv_params, filters=filters, name=name)

        return builder

    def _pool(self, name):
        def builder(inputs):
            return tf.layers.max_pooling2d(inputs, **self.max_pool_params, name=name)

        return builder

    def _make_conv_layer_builders(self):
        return [
            self._conv(64, 'conv1'),
            self._conv(64, 'conv2'),
            self._pool('pool1'),
            self._conv(128, 'conv3'),
            self._conv(128, 'conv4'),
            self._pool('pool2'),
            self._conv(256, 'conv5'),
            self._conv(256, 'conv6'),
            self._conv(256, 'conv7'),
            self._pool('pool3'),
            self._conv(512, 'conv8'),
            self._conv(512, 'conv9'),
            self._conv(512, 'conv10'),
            self._pool('pool4'),
            self._conv(512, 'conv11'),
            self._conv(512, 'conv12'),
            self._conv(512, 'conv13'),
            self._pool('pool5'),
        ]

    def _build_layers_v2(self, input_dict, num_outputs, options):
        inputs = input_dict['obs']
        assert inputs.shape[1:] == (128, 128, 4)

        conv_layers = self._make_conv_layer_builders()

        return self._build_vgg(inputs, conv_layers, 512, num_outputs)

    def _build_vgg(self, inputs, conv_layers, fc1_size=512, fc2_size=1):
        output = inputs
        with tf.name_scope(self._name):
            for builder in conv_layers:
                output = builder(output)

            output = tf.layers.flatten(output)

            fc1 = tf.layers.dense(
                output,
                fc1_size,
                activation=tf.nn.relu,
                name='fc1'
            )

            fc2 = tf.layers.dense(
                fc1,
                fc2_size,
                activation=None,
                name='fc2'
            )

            return fc2, fc1
