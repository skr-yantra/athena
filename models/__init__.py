from ray.rllib.models import ModelCatalog

from .vgg_v1 import VGGNet as VGGNetV1
from .svgg_v1 import SimpleVisionNet as SimpleVisionNetV1

ModelCatalog.register_custom_model('vggnet_v1', VGGNetV1)
ModelCatalog.register_custom_model('svggnet_v1', SimpleVisionNetV1)
