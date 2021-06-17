import pytest
import os
import numpy as np
from PIL.Image import Image

from jina import DocumentArray, Document

from jinahub.image.encoder import ImagePaddlehubEncoder

directory = os.path.dirname(os.path.realpath(__file__))


def test_test():
    assert True