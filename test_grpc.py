# -*- coding:utf-8 -*-
# Time      :2023/6/27 10:01
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

import base64
import logging

import grpc

from sfm.utils import LocalStorageImageDataset, ImageFileBytesDataset

from server_pb2 import DataRequest, DataCamera
from server_pb2_grpc import SfmServerStub


def run():
    data_set = ImageFileBytesDataset("test_picture1")

    with grpc.insecure_channel('localhost:50000') as channel:
        stub = SfmServerStub(channel)

        def data_iter():
            camera = DataRequest(
                type_id=0,
                camera=DataCamera(
                    mrt=0.7,
                    x=0.5,
                    y=1,
                    k=[2362.12, 0, 720, 0, 2362.12, 578, 0, 0, 1]
                ),
            )
            yield camera
            for image_bytes in data_set:
                image_data = DataRequest(
                    type_id=1,
                    image=image_bytes
                )
                yield image_data

        # 客户端通过stub来实现rpc通信
        data_response = stub.rebuild(data_iter())
        for data in data_response:
            print(data)
        # for i, data in enumerate(data_response):
        #     print(f"{i}, {data}")
            print("=================")


if __name__ == "__main__":
    logging.basicConfig()
    run()
