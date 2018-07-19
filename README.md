# 3D Boxes From Imagery and Point Clouds

This project aims to find 3D boxes in imagery and pointcloud data. 


Given
 * High resolution imagery with 3D camera parameters 
 * Point cloud data such as mobile LiDAR

Find:
 * Oriented 3D boxes (aka parallelepipeds, or cuboids) that contain a target
 class.


The work is part of a [Thesis]() and we plan to make it part of a paper; if you find it useful please _check back_ for a paper to cite:

```bibtex
# Not yet even a preprint; we need this when we make the repo public
@article{liux2018srp,
  title={3D Boxes from Imagery and Point Clouds},
  author={},
  journal={},
  volume={},
  pages={},
  year={2018},
  publisher={}
}
```

# Try It With Docker

Using a prebuilt docker image.
If you have CUDA and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker):

    nvidia-docker run -it -v $CWD:/workspace -p 8888:8888 --name srp jfemiani/srp:cuda8

If you plan to use the CPU only

    docker run -it -v $CWD:/workspace -p 8888:8888 --name srp jfemiani/srp:cpu


The script should start a jupyter server, open the link that is displayed
in the terminal in order to see the code in action. 

# Dependencies

## PyTorch

  [PyTorch](www.pytorch.org) is a deep learning library that we used to do
  object detection. Use the link to install the appropriate version for your system before
  attempting to use this code. 
  For best performance you should have a [decent NVIDIA GPU](
  https://www.nvidia.com/en-us/geforce/products/10series/geforce-gtx-1080-ti/)
  and the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) installed
  on your system, however you can run the code entirely in CPU if you download
  a version of torch with no CUDA. This may be necessary if, e.g., you are
  working on a mac. 

# Setup

## Using PIP:


    pip install git+https://github.com/liux13/srp


## From source
 
    python setup.py install


## For development or building models

For development, building models, and doing experiments we use GNU make
    
    make all
  

# Documentation

Our [jupyter notebooks](nb/) can be used as a tutorial of sorts. 

API Documentation is not currently hosted. To build documentation use
    
    make doc

# Data

We build our models on data provided by the Salt River Project (SRP); we will seek permission to make it public.
In the meantime, if you want access to the data you will have to send me an email and I will forward it along. 

* LiDAR data: Please request this by email 
* RGB 4" Aerial Ortho-Imagery: Please request this by email.
* Manually annotated boxes: link TBD

The data is provided as three archives; place them directly under the repo's `data` folder and run 

    make data

To prepare the data for training. 

