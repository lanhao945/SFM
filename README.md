# sfm

三维重建算法Structure from Motion(Sfm)的python实现

## 环境准备

```shell
pip install -r requirements.txt
```

## 使用

见单元测试

## 原理参考

https://blog.csdn.net/aichipmunk/article/details/48132109

## proto命令

```shell
python -m grpc_tools.protoc -I. --python_out=. --pyi_out=. --grpc_python_out=. server.proto
```
