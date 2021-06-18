import pytest
import os
import numpy as np
from PIL.Image import Image
from typing import Tuple, Dict
from jina import DocumentArray, Document
from jina.executors.metas import get_default_metas
from jinahub.encoder.paddle_image import ImagePaddlehubEncoder
#from paddle_image import ImagePaddlehubEncoder

directory = os.path.dirname(os.path.realpath(__file__))


input_dim = 224
target_output_dim = 2048
num_doc = 2
test_data = np.random.rand(num_doc, 3, input_dim, input_dim)
tmp_files = []


@pytest.fixture(scope='function', autouse=True)
def metas(tmpdir):
    metas = get_default_metas()
    if 'JINA_TEST_GPU' in os.environ:
        metas['on_gpu'] = True
    metas['workspace'] = str(tmpdir)
    yield metas


def test_imagepaddlehubencoder_encode(test_images: Dict[str, np.array]):

    encoder = ImagePaddlehubEncoder(channel_axis=3)

    embeddings = {}
    for name, image_arr in test_images.items():
        docs = DocumentArray([Document(blob=image_arr)])
        encoder.encode(docs, parameters={})
        embeddings[name] = docs[0].blob

    def dist(a, b):
        a_embedding = embeddings[a]
        b_embedding = embeddings[b]
        return np.linalg.norm(a_embedding - b_embedding)

    small_distance = dist('banana1', 'banana2')
    assert small_distance < dist('banana1', 'airplane')
    assert small_distance < dist('banana1', 'satellite')
    assert small_distance < dist('banana1', 'studio')
    assert small_distance < dist('banana2', 'airplane')
    assert small_distance < dist('banana2', 'satellite')
    assert small_distance < dist('banana2', 'studio')
    assert small_distance < dist('airplane', 'studio')
    assert small_distance < dist('airplane', 'satellite')
