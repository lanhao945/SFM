# -*- coding:utf-8 -*-
# Time      :2023/6/21 11:04
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
import unittest

import cv2
import numpy as np

from sfm import utils, core, datas

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SDKTest(unittest.TestCase):

    def test_logic(self):
        dataset: datas.ImageDataset = utils.LocalStorageImageDataset(
            "test_picture1")
        results: datas.ColorPoints = core.rebuild(datas.SFMData(
            dataset, datas.Camera()
        ))
        self.assertIsInstance(results, datas.ColorPoints, "返回类型异常")
        self.assertTrue(len(results.points) == len(results.colors),
                        "点和颜色数量不一致")
        utils.fig_v1(results.points)


if __name__ == '__main__':
    pass
