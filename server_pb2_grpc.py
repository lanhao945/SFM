# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import server_pb2 as server__pb2


class SfmServerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.rebuild = channel.stream_stream(
                '/sfm_server.SfmServer/rebuild',
                request_serializer=server__pb2.DataRequest.SerializeToString,
                response_deserializer=server__pb2.DataReply.FromString,
                )


class SfmServerServicer(object):
    """Missing associated documentation comment in .proto file."""

    def rebuild(self, request_iterator, context):
        """Sends a greeting
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_SfmServerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'rebuild': grpc.stream_stream_rpc_method_handler(
                    servicer.rebuild,
                    request_deserializer=server__pb2.DataRequest.FromString,
                    response_serializer=server__pb2.DataReply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'sfm_server.SfmServer', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class SfmServer(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def rebuild(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/sfm_server.SfmServer/rebuild',
            server__pb2.DataRequest.SerializeToString,
            server__pb2.DataReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
