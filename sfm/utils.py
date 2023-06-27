# -*- coding:utf-8 -*-
# Time      :2023/6/25 10:37
# Author    :LanHao
# ▄▄▄█████▓ █    ██  ██ ▄█▀▓█████ ▓█████▄
# ▓  ██▒ ▓▒ ██  ▓██▒ ██▄█▒ ▓█   ▀ ▒██▀ ██▌
# ▒ ▓██░ ▒░▓██  ▒██░▓███▄░ ▒███   ░██   █▌
# ░ ▓██▓ ░ ▓▓█  ░██░▓██ █▄ ▒▓█  ▄ ░▓█▄   ▌
#   ▒██▒ ░ ▒▒█████▓ ▒██▒ █▄░▒████▒░▒████▓
#   ▒ ░░   ░▒▓▒ ▒ ▒ ▒ ▒▒ ▓▒░░ ▒░ ░ ▒▒▓  ▒
#     ░    ░░▒░ ░ ░ ░ ░▒ ▒░ ░ ░  ░ ░ ▒  ▒
#   ░       ░░░ ░ ░ ░ ░░ ░    ░    ░ ░  ░
#             ░     ░  ░      ░  ░   ░
#                                  ░

import os
import logging
from typing import List, Iterator, Iterable

import cv2
from mayavi import mlab
import numpy as np
import numpy.typing as npt

from .datas import ImageDataset

logger = logging.getLogger(__name__)


class ImageFileBytesDataset(Iterable[bytes]):
    _image_path: str
    _image_names: List[str]

    def __init__(self, path):
        img_names = os.listdir(path)
        self._image_names = sorted(img_names)
        self._image_path = path

    def __iter__(self) -> Iterator[bytes]:
        for image_name in self._image_names:
            image_path = os.path.join(self._image_path, image_name)
            with open(image_path, "rb") as f:
                yield f.read()


class LocalStorageImageDataset(ImageDataset):
    """
    磁盘目录载入数据
    """

    _bytes_iter: Iterable[bytes]

    def __init__(self, data_path):
        self._bytes_iter = ImageFileBytesDataset(data_path)

    def __iter__(self):
        for data_bytes in self._bytes_iter:
            yield cv2.imdecode(np.frombuffer(data_bytes, np.uint8),
                               cv2.IMREAD_COLOR)


def fig_v1(points: npt.NDArray):
    mlab.points3d(points[:, 0], points[:, 1], points[:, 2],
                  mode='point', name='dinosaur')
    mlab.show()


if __name__ == '__main__':
    pass
