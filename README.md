# Batch Fog Sim

This notebook allows you to apply realistic fog simulation on lidar point clouds.

Currently, only point clouds following the format proposed by the famous KITTI data set are supported (each point has four dimensions: x, y, z, intensity).

**This work is entirely based the awesome work done by Martin Hahner et. al.!** You can find the original repo [here](https://github.com/MartinHahner/LiDAR_fog_sim). Please check it out!

## Usage

1. Install the necessary packages (I use conda for this)
2. Place your original binary lidar files in `data/original`
3. Run the entire notebook

You will find the augmented data in `data/foggy`.

Adjusting the values `alpha`, `beta` and `gamma` in the cell containing all variables will change the properties of the fog simulation. Again, please check out Martin Hahners work for more information. His repo also provides a graphical tool to play around with the settings.

_This was tested with python version 3.9.16_.
