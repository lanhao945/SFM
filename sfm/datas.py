# -*- coding:utf-8 -*-
# Time      :2023/6/25 10:09
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

import abc
import logging
from typing import Any, Iterator

import numpy as np
import numpy.typing as npt
from attr import attrs, attrib

logger = logging.getLogger(__name__)

_DEFAULT_K = np.array([
    [2362.12, 0, 720],
    [0, 2362.12, 578],
    [0, 0, 1]])


@attrs
class Camera:
    mrt = attrib(type=float, default=0.7)
    k = attrib(type=npt.NDArray[np.float], default=_DEFAULT_K)
    x = attrib(type=float, default=0.5)
    y = attrib(type=int, default=1)


class ImageDataset(abc.ABC):
    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @abc.abstractmethod
    def __getitem__(self, item) -> npt.NDArray[np.uint8]:
        ...

    @abc.abstractmethod
    def __iter__(self) -> Iterator[npt.NDArray[np.uint8]]:
        ...


_DEFAULT_CAMERA = Camera()


@attrs
class SFMData:
    image = attrib(type=ImageDataset)
    camera = attrib(type=Camera, default=_DEFAULT_CAMERA)


@attrs
class ColorPoints:
    points = attrib(type=Any)
    colors = attrib(type=Any)


if __name__ == '__main__':
    pass
