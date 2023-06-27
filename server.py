# -*- coding:utf-8 -*-
# Time      :2023/6/27 8:50
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
import asyncio
import logging
from typing import List, Iterator, Sequence, Optional
from collections.abc import AsyncIterable
import numpy as np
from grpc import aio
from google.protobuf.internal import containers as _containers

import server_pb2_grpc
from sfm import core
from sfm.datas import ImageDataset, Camera, SFMData, ColorPoints
from sfm.utils import LocalStorageImageDataset
from server_pb2 import (DataReply, DataIter, DataColor, DataPoint, DataRequest,
                        DataCamera)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# LocalStorageImageDataset 在构造上,支持直接从bytes 加载,
BytesImageDataset = LocalStorageImageDataset


class Exchange:

    @staticmethod
    def sfm_data_2_grpc(data: ColorPoints) -> Iterator[DataReply]:
        points = data.points
        colors = data.colors
        for point, color in zip(points, colors):
            yield DataReply(row=DataIter(
                point=DataPoint(
                    x=point[0],
                    y=point[1],
                    z=point[2],
                ),
                color=DataColor(
                    r=int(color[0]),
                    g=int(color[1]),
                    b=int(color[2]),
                )
            ))

    @staticmethod
    async def grpc_2_sfm_data(datas: AsyncIterable[DataRequest]) -> SFMData:
        logger.debug("进入数据转换")
        camera: Optional[DataCamera] = None
        images: List[bytes] = []
        async for data in datas:
            data: DataRequest
            if data.type_id == 0:
                camera = data.camera
            else:
                images.append(data.image)

        camera_in = None if camera is None else Camera(
            mrt=camera.mrt,
            x=camera.x,
            y=camera.y,
            k=np.asarray(camera.k).reshape((3, 3))
        )
        image_dataset: ImageDataset = BytesImageDataset(images)
        sfm_data = SFMData(
            image=image_dataset,
            camera=camera_in
        )
        return sfm_data


class SFMServer(server_pb2_grpc.SfmServer):
    @staticmethod
    async def rebuild(request: AsyncIterable[DataRequest], context, *args,
                      **kwargs):
        logger.debug("start grpc.rebuild .. .")
        sfm_data = await Exchange.grpc_2_sfm_data(request)
        result: ColorPoints = core.rebuild(sfm_data)
        for data in Exchange.sfm_data_2_grpc(result):
            yield data


class GrpcServer:

    @staticmethod
    def run():
        async def grpc_server():
            server = aio.server(options=[
                ('grpc.max_send_message_length', -1),
                ('grpc.max_receive_message_length', -1)
            ])
            server_pb2_grpc.add_SfmServerServicer_to_server(SFMServer, server)
            server.add_insecure_port(f'[::]:50000')
            await server.start()
            await server.wait_for_termination()

        asyncio.run(grpc_server())


if __name__ == '__main__':
    GrpcServer.run()
