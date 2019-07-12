# Edge-GAN

We propose a new approach to generate new images- splitting the generating task into two parts. The first one is generating edges and the second one is coloring them. In the first part we use Deep Convolutional GAN (DCGAN), and in the second part we use pix2pix network.

## Instructions

In order to generate new images with your dataset using this method you have to:

### 1. Prepare the edges dataset using HED and Canny.
First run `python canny.py` to extract edges from your dataset using canny algorithm,
then extract edges using the HED network (link to the HED source code down below),
finally run `combine_edges.py` to combine the edges you extracted before.  

### 2. Train the DCGAN with the edges dataset you created.
To train the DCGAN with the edges dataset run: `python DCGAN_train.py dataRoot`.
Replace 'dataRoot' with the path to the directory that contains a directory with the
edges images you created in the first step.
You can run `python DCGAN_train.py -h` to see more arguments you can change like
the batch size and number of epochs.

### 3. Train the pix2pix network to paint the edges.
To train the pix2pix network you first have to create a dataset of pairs of images from the two domains.
To do so, first you need to parition each domain into three parts- train, test and val.
Then run `python datasets/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data`.
Now to train the network, run `python train.py --dataroot ./datasets/your-dataset --name name_pix2pix --model pix2pix --direction BtoA` where "your-dataset" is a path to your dataset of pairs and "name" is the name of the folder your results will be saved to.

### 4. Generate new edges and color them
Now you can generate new edges. In order to do so run `python generate_edges.py Gpath Rpath`.
Replace 'Gpath' with path to the DCGAN weights and 'Rpath' with path to the destination directory. 
Finally to color the fake edges and get the final results, run `python test.py --dataroot Rpath --name name_pix2pix --model test --netG unet_256 --direction BtoA --dataset_mode single --norm batch`.

Note: You can also use our weights of the pre-trained model, named "checkpointG.pth" and perform only the last two steps above.  

## Usefull links
The frogs dataset was taken from: https://github.com/jonshamir/frog-dataset

The HED network: https://github.com/sniklaus/pytorch-hed

The DCGAN is based on: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

The pix2pix network: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/scripts
