__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional, Iterable, List, Any

import numpy as np
from jina import DocumentArray, Executor, requests


def _batch_generator(data: List[Any], batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i:min(i + batch_size, len(data))]


class ImagePaddlehubEncoder(Executor):
    """
    :class:`ImagePaddlehubEncoder` encodes ``Document`` content from a ndarray,
    potentially B x (Channel x Height x Width) into a ndarray of `B x D`.

    Internally, :class:`ImagePaddlehubEncoder` wraps the models from `paddlehub`.
    https://github.com/PaddlePaddle/PaddleHub

    :param model_name: the name of the model. Supported models include
        ``xception71_imagenet``, ``xception65_imagenet``, ``xception41_imagenet``,
        ``vgg19_imagenet``, ``vgg16_imagenet``, ``vgg13_imagenet``, ``vgg11_imagenet``,
        ``shufflenet_v2_imagenet``,``se_resnext50_32x4d_imagenet``,
        ``se_resnext101_32x4d_imagenet``,  ``resnext50_vd_64x4d_imagenet``,
        ``resnext50_vd_32x4d_imagenet``, `resnext50_64x4d_imagenet``,
        ``resnext50_32x4d_imagenet``, ``resnext152_vd_64x4d_imagenet``,
        ``resnext152_64x4d_imagenet``, ``resnext152_32x4d_imagenet``,
        ``resnext101_vd_64x4d_imagenet``, ``resnext101_vd_32x4d_imagenet``,
        ``resnext101_32x8d_wsl``, ``resnext101_32x48d_wsl``, ``resnext101_32x32d_wsl``,
        ``resnext101_32x16d_wsl``, ``resnet_v2_50_imagenet``, ``resnet_v2_34_imagenet``,
        ``resnet_v2_18_imagenet``, ``resnet_v2_152_imagenet``, ``resnet_v2_101_imagenet``,
        ``mobilenet_v2_imagenet``, ``inception_v4_imagenet``, ``googlenet_imagenet``,
        ``efficientnetb7_imagenet``, ``efficientnetb6_imagenet``, ``efficientnetb5_imagenet``,
        ``efficientnetb4_imagenet``, ``efficientnetb3_imagenet``, ``efficientnetb2_imagenet``,
        ``efficientnetb1_imagenet``, ``efficientnetb0_imagenet``, ``dpn68_imagenet``,
        ``dpn131_imagenet``, ``dpn107_imagenet``, ``densenet264_imagenet``,
        ``densenet201_imagenet``, ``densenet169_imagenet``, ``densenet161_imagenet``,
        ``densenet121_imagenet``, ``darknet53_imagenet``, ``alexnet_imagenet``,
    :param pool_strategy: the pooling strategy. Default is `None`.
    :param channel_axis: The axis of the color channel, default is -3
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """
    def __init__(
            self,
            model_name: Optional[str] = None,
            pool_strategy: Optional[str] = None,
            channel_axis: int = -3,
            default_batch_size: int = 32,
            default_traversal_path: str = 'r',
            on_gpu: bool = False,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pool_strategy = pool_strategy
        self.channel_axis = channel_axis
        self.model_name = model_name or 'xception71_imagenet'
        self.pool_strategy = pool_strategy or 'mean'
        self._default_channel_axis = -3
        self.inputs_name = None
        self.outputs_name = None
        self.on_gpu = on_gpu
        self.default_batch_size = default_batch_size
        self.default_traversal_path = default_traversal_path

        import paddlehub as hub
        module = hub.Module(name=self.model_name)
        inputs, outputs, self.model = module.context(trainable=False)
        self._get_inputs_and_outputs_name(inputs, outputs)

        import paddle.fluid as fluid
        self.device = fluid.CUDAPlace(0) if self.on_gpu else fluid.CPUPlace()
        self.exe = fluid.Executor(self.device)

    @requests
    def encode(self, docs: DocumentArray, parameters: dict, **kwargs) -> DocumentArray:
        if docs:
            document_batches_generator = self._get_input_data(docs, parameters)
            self._create_embeddings(document_batches_generator)

    def _create_embeddings(self, document_batches_generator: Iterable):
        for document_batch in document_batches_generator:
            blob_batch = [d.blob for d in document_batch]
            if self.channel_axis != self._default_channel_axis:
                blob_batch = np.moveaxis(blob_batch, self.channel_axis, self._default_channel_axis)
            feature_map, *_ = self.exe.run(
                program=self.model,
                fetch_list=[self.outputs_name],
                feed={self.inputs_name: blob_batch.astype('float32')},
                return_numpy=True
            )

            if feature_map.ndim == 2 or self.pool_strategy is None:
                embedding_batch = feature_map
            else:
                embedding_batch = self._get_pooling(feature_map)

            for document, embedding in zip(document_batch, embedding_batch):
                document.embedding = embedding

    def _get_input_data(self, docs: DocumentArray, parameters: dict):
        traversal_path = parameters.get('traversal_path', self.default_traversal_path)
        batch_size = parameters.get('batch_size', self.default_batch_size)

        # traverse thought all documents which have to be processed
        flat_docs = docs.traverse_flat(traversal_path)

        # filter out documents without images
        filtered_docs = [doc for doc in flat_docs if doc.blob is not None]

        return _batch_generator(filtered_docs, batch_size)

    def _get_pooling(self, content: 'np.ndarray') -> 'np.ndarray':
        """Get ndarray with selected pooling strategy"""
        _reduce_axis = tuple((i for i in range(len(content.shape)) if i > 1))
        return getattr(np, self.pool_strategy)(content, axis=_reduce_axis)

    def _get_inputs_and_outputs_name(self, input_dict, output_dict):
        """Get inputs_name (image name) and outputs_name (feature map)."""
        self.inputs_name = input_dict['image'].name
        self.outputs_name = output_dict['feature_map'].name
        if self.model_name.startswith('vgg') or self.model_name.startswith('alexnet'):
            self.outputs_name = f'@HUB_{self.model_name}@fc_1.tmp_2'
