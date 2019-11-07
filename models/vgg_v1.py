from ray.rllib.models import Model
from ray.rllib.utils import try_import_tf

tf = try_import_tf()


class VGGNet(Model):

    def _build_layers_v2(self, input_dict, num_outputs, options):
        inputs = input_dict['obs']

        assert inputs.shape[1:] == (128, 128, 4)

        conv_params = dict(
            padding='same',
            activation=tf.nn.relu,
            kernel_size=3,
            strides=(1, 1),
        )

        max_pool_params = dict(
            pool_size=2,
            strides=2,
        )

        def conv(filters, name):
            def builder(inputs):
                return tf.layers.conv2d(inputs, **conv_params, filters=filters, name=name)

            return builder

        def pool(name):
            def builder(inputs):
                return tf.layers.max_pooling2d(inputs, **max_pool_params, name=name)

            return builder

        conv_layers = [
            conv(64, 'conv1'),
            conv(64, 'conv2'),
            pool('pool1'),
            conv(128, 'conv3'),
            conv(128, 'conv4'),
            pool('pool2'),
            conv(256, 'conv5'),
            conv(256, 'conv6'),
            conv(256, 'conv7'),
            pool('pool3'),
            conv(512, 'conv8'),
            conv(512, 'conv9'),
            conv(512, 'conv10'),
            pool('pool4'),
            conv(512, 'conv11'),
            conv(512, 'conv12'),
            conv(512, 'conv13'),
            pool('pool5'),
        ]

        output = inputs
        with tf.name_scope('vgg_v1'):
            for builder in conv_layers:
                output = builder(output)

            output = tf.layers.flatten(output)

            fc1 = tf.layers.dense(
                output,
                512,
                activation=tf.nn.relu,
                name='fc1'
            )

            fc2 = tf.layers.dense(
                fc1,
                num_outputs,
                activation=None,
                name='fc2'
            )

            return fc2, fc1
