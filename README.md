# Flask API for Multi Person Action Recognition
> Pretrained actions, total 9 classes : **['stand', 'walk', 'run', 'jump', 'sit', 'squat', 'kick', 'punch', 'wave']**

  

<table  style="width:100%; table-layout:fixed;">

<tr>

<td><img  width="448"  height="224"  src="assets/aung_la.gif"></td>

<td><img  width="448"  height="224"  src="assets/aung_la_debug.gif"></td>

</tr>

<tr>

<td  align="center"><font  size="1">Fight scene demo<font></td>

<td  align="center"><font  size="1">Fight scene debug demo<font></td>

</tr>

<tr>

<td><img  width="448"  height="224"  src="assets/fun_theory.gif"></td>

<td><img  width="448"  height="224"  src="assets/fun_theory_debug.gif"></td>

</tr>

<tr>

<td  align="center"><font  size="1">Street scene demo<font></td>

<td  align="center"><font  size="1">Street scene debug demo<font></td>

</tr>

<tr>

<td><img  width="448"  height="224"  src="assets/street_walk.gif"></td>

<td><img  width="448"  height="224"  src="assets/street_walk_debug.gif"></td>

</tr>

<tr>

<td  align="center"><font  size="1">Street walk demo<font></td>

<td  align="center"><font  size="1">Street walk debug demo<font></td>

</tr>

</table>

# Overview

This is the 3 steps multi-person action recognition pipeline. But it achieves real time performance with 33 FPS for whole action recognition pipeline with 1 person video. The steps include:

1. pose estimation with [trtpose](https://github.com/NVIDIA-AI-IOT/trt_pose)

2. people tracking with [deepsort](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)

3. action classifier with [dnn](https://github.com/felixchenfy/Realtime-Action-Recognition#diagram)
# Installation
You can deploy the project either with Docker using docker-compose or with manual setup

# Using Docker

 1. Build docker image: `docker build . -t human_recognition`
 2. Run docker-compose: `docker-compose up -d`

  


  

# Manual setup
First, Python >= 3.6
## Step 1 - Install Dependencies

  

Check this [installation guide](https://github.com/CV-ZMH/Installation-Notes-for-Deeplearning-Developement#install-tensorrt) for deep learning packages installation.

  

Here is required packages for this project and you need to install each of these.

1. Nvidia-driver 450 (Not required)

2.  [Cuda-10.2](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal) and [Cudnn 8.0.5](https://developer.nvidia.com/rdp/cudnn-archive) (Not required)

3.  [Pytorch 1.7.1](https://pytorch.org/get-started/previous-versions/) and [Torchvision 0.8.2](https://pytorch.org/get-started/previous-versions/)

4.  [TensorRT 7.1.3](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-723/install-guide/index.html) (Not required)

5.  [ONNX 1.9.0](https://pypi.org/project/onnx/)

  

## Step 2 - Install [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) (Not required)

```bash

git clone https://github.com/NVIDIA-AI-IOT/torch2trt

cd torch2trt

sudo python3 setup.py install --plugins

```

## Step 3 - Install trt_pose

  

```bash

git clone https://github.com/NVIDIA-AI-IOT/trt_pose

cd trt_pose

sudo python setup.py install

```

Other python packages are in [`requirements.txt`](requirements.txt).

  

Run below command to install them.

```bash

pip install -r requirements.txt

```

## Step 4 -  Download the Pretrained Models

Action Classifier Pretrained models are already uploaded in the path `weights/classifier/dnn`.

- Download the pretrained weight files, either trained on Market1501 or Mars dataset.
- Make sure to update the config files in [confiigs/infer_trtpose_deepsort_dnn.yaml](configs/infer_trtpose_deepsort_dnn.yaml) file.

  

Examples:

- To use different reid network, [`reid_name`](https://github.com/CV-ZMH/human_activity_recognition/blob/ad2f8adfbd30e1ae1ea3b964a2f144ce757d944a/configs/infer_trtpose_deepsort_dnn.yaml#L37) and it's [`model_path`](https://github.com/CV-ZMH/human_activity_recognition/blob/ad2f8adfbd30e1ae1ea3b964a2f144ce757d944a/configs/infer_trtpose_deepsort_dnn.yaml#L38) in [`TRACKER`](https://github.com/CV-ZMH/human-action-recognition/blob/6bdf3a541eb6adc618c67e4e6df7d6fdaddffb1b/configs/infer_trtpose_deepsort_dnn.yaml#L20) node.

- Set also [`model_path`](https://github.com/CV-ZMH/human-action-recognition/blob/6bdf3a541eb6adc618c67e4e6df7d6fdaddffb1b/configs/infer_trtpose_deepsort_dnn.yaml#L13) of trtpose in [`POSE`](https://github.com/CV-ZMH/human-action-recognition/blob/6bdf3a541eb6adc618c67e4e6df7d6fdaddffb1b/configs/infer_trtpose_deepsort_dnn.yaml#L4) node.

- You can also tune other parameters of [`TRACKER`](https://github.com/CV-ZMH/human-action-recognition/blob/6bdf3a541eb6adc618c67e4e6df7d6fdaddffb1b/configs/infer_trtpose_deepsort_dnn.yaml#L20) node and [`POSE`](https://github.com/CV-ZMH/human-action-recognition/blob/6bdf3a541eb6adc618c67e4e6df7d6fdaddffb1b/configs/infer_trtpose_deepsort_dnn.yaml#L4) node for better tracking and action recognition result.


| Model Type | Name | Trained Dataset | Weight |
|---|---|---|---|
| Pose Estimation | trtpose | COCO |[densenet121](https://drive.google.com/file/d/1De2VNUArwYbP_aP6wYgDJfPInG0sRYca/view?usp=sharing) |
|||||
| Tracking | deepsort reid| Market1501 | [wide_resnet](https://drive.google.com/file/d/1xw7Sv4KhrzXMQVVQ6Pc9QeH4QDpxBfv_/view?usp=sharing)|
| Tracking | deepsort reid| Market1501 | [siamese_net](https://drive.google.com/file/d/11OmfZqnnG4UBOzr05LEKvKmTDF5MOf2H/view?usp=sharing)|
| Tracking | deepsort reid| Mars | [wide_resnet](https://drive.google.com/file/d/1lRvkNsJrR4qj50JHsKStEbaSuEXU3u1-/view?usp=sharing)|
| Tracking | deepsort reid| Mars | [siamese_net](https://drive.google.com/file/d/1eyj5zKoLjnHqfSIz2eJjXq0r9Sw7k0R0/view?usp=sharing)|
  
  

- Then put them to these folder

-  *deepsort* weight to `weights/tracker/deepsort/`

-  *trt_pose* weight to `weights/pose_estimation/trtpose`.
## Step 5 - Run the Server
```bash

export FLASK_APP=server
flask run

```
## Step 6 - Invoke API
- API endpoint: `/action`
- Input type: `multipart/form-data`
- Input: 

	 - video: file (required - allowed extensions: mp4, avi)
	 - task: (action, pose, track) (optional - default: action)
	 - config: file (optional)
	 - draw_kp_numbers: true/false (optional - default: false)
	 - debug_track: true/false (optional - default: true)
 
- Output: {url: "result_url"}
