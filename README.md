# ROS wrapper for Hybrid Task Cascade
ROS wrapper for [mmdetection framework](https://github.com/open-mmlab/mmdetection) with [Hybrid Cascade Task model](https://github.com/open-mmlab/mmdetection/tree/master/configs/htc). This work was carried out as student trainee in the university [Hochschule Weingarten](http://www.rwu.de).

## Requirements
NVIDIA driver of host system must be [>= 410.48](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) to support CUDA 10.0 inside the docker container.

## Build docker image
```console
git clone https://github.com/iki-wgt/ros_htc_wrapper.git
cd ros_htc_wrapper/
docker build -t ros_htc . --no-cache
```

## Messages
### Service message
The [Service message](./htc_service_msg/srv/HtcServiceMsg.srv) takes a given image and returns a list of objects of [Object message](./htc_msg/msg/Object.msg) type containing recognized pixel segmented objects:
```console
sensor_msgs/Image image
---
htc_msg/Object[] objects
```

### Object message
```console
string class_name
float32 score
float32[] bbox
string rle
```

### Parsing RLE string
To transform an RLE string of Object message into a segmentation mask (2-dimensional boolean numpy array), use json and pycocotools libs:
```
import json
import numpy as np
import pycocotools.mask as maskUtils

rle = json.loads(result.objects[0].rle)  # parse string from service response
rle['counts'] = bytearray(rle['counts'])  # pycocotools demands encoded RLE
mask_np = maskUtils.decode(rle).astype(np.bool)  # convert dict to boolean numpy array 
```
