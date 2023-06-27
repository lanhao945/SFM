from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DataRequest(_message.Message):
    __slots__ = ["type_id", "image", "camera"]
    TYPE_ID_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    CAMERA_FIELD_NUMBER: _ClassVar[int]
    type_id: int
    image: bytes
    camera: DataCamera
    def __init__(self, type_id: _Optional[int] = ..., image: _Optional[bytes] = ..., camera: _Optional[_Union[DataCamera, _Mapping]] = ...) -> None: ...

class DataCamera(_message.Message):
    __slots__ = ["mrt", "k", "x", "y"]
    MRT_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    mrt: float
    k: _containers.RepeatedScalarFieldContainer[float]
    x: float
    y: float
    def __init__(self, mrt: _Optional[float] = ..., k: _Optional[_Iterable[float]] = ..., x: _Optional[float] = ..., y: _Optional[float] = ...) -> None: ...

class DataPoint(_message.Message):
    __slots__ = ["x", "y", "z"]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    z: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ...) -> None: ...

class DataColor(_message.Message):
    __slots__ = ["r", "g", "b"]
    R_FIELD_NUMBER: _ClassVar[int]
    G_FIELD_NUMBER: _ClassVar[int]
    B_FIELD_NUMBER: _ClassVar[int]
    r: int
    g: int
    b: int
    def __init__(self, r: _Optional[int] = ..., g: _Optional[int] = ..., b: _Optional[int] = ...) -> None: ...

class DataIter(_message.Message):
    __slots__ = ["point", "color"]
    POINT_FIELD_NUMBER: _ClassVar[int]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    point: DataPoint
    color: DataColor
    def __init__(self, point: _Optional[_Union[DataPoint, _Mapping]] = ..., color: _Optional[_Union[DataColor, _Mapping]] = ...) -> None: ...

class DataReply(_message.Message):
    __slots__ = ["row"]
    ROW_FIELD_NUMBER: _ClassVar[int]
    row: DataIter
    def __init__(self, row: _Optional[_Union[DataIter, _Mapping]] = ...) -> None: ...
