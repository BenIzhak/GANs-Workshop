# GANs-Workshop

## Steps in training the networks

In order to generate new images using this method you have to:

### 1. Prepare the dataset using HED and Canny.

2. Train the DCGAN to generate the edges.
3. Train the pix2pix network to paint the edges.

The frogs dataset was taken from: https://github.com/jonshamir/frog-dataset

The HED network was taken from: https://github.com/sniklaus/pytorch-hed

The DCGAN network based on: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
