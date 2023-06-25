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
from typing import List

import cv2

from .datas import ImageDataset

logger = logging.getLogger(__name__)


class LocalStorageImageDataset(ImageDataset):
    """
    磁盘目录载入数据
    """

    _data_path: str
    _img_names: List[str]

    def __init__(self, data_path):
        img_names = os.listdir(data_path)
        self._img_names = sorted(img_names)
        self._data_path = data_path

    def __getitem__(self, item):
        image_path = os.path.join(
            self._data_path,
            self._img_names[item]
        )
        logger.debug("%s path :%s", item, image_path)
        if not os.path.exists(image_path):
            raise Exception(f"{image_path} not exists")
        image = cv2.imread(image_path)
        return image

    def __len__(self):
        return len(self._img_names)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


if __name__ == '__main__':
    pass
