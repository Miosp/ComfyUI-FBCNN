# ComfyUI-FBCNN
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) implementation of [FBCNN](https://github.com/jiaxi-jiang/FBCNN)

## What is it?
This is a JPEG artifact/compression removal tool, which can work automatically or with the help of a user.

## Installation
Download or git clone this repository inside ComfyUI/custom_nodes/ directory. The only required dependency is Pytorch, numpy, and requests, which should be already installed if you have ComfyUI.

## How to use it?
In ComfyUI, you can find this node under `image/upscaling` category.
The model requires an estimate of the compression level, which is a number between 0 and 100 (the same you need to provide when compressing a JPEG image). You can either set `auto_detect` to `enabled` and use built-in estimator, or set it to `disabled` and provide the value yourself in `compression_level`.
⚠️ If you leave `auto_detect` as `enabled`, the value of `compression_level` will be ignored.

## Credits
- [FBCNN](https://github.com/jiaxi-jiang/FBCNN)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)