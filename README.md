# Edge-GAN

We propose a new approach to generate images- splitting the generating task into two parts. The first one is generating edges and the second one is coloring them. In the first part we use Deep Convolutional GAN (DCGAN) to generate the edges of the new images, and in the second part we use pix2pix network to color them.

## Instructions

In order to generate new images with yours dataset using this method you have to:

### 1. Prepare the edges dataset using HED and Canny.
First run canny.py to extract eages from your dataset using canny algorithm,
then extract edges using the HED network (link to the HED source code down below),
finally run TODO-TAKE FROM ROTEM to combine the edges we extracted before.  

### 2. Train the DCGAN with the edges dataset we created.
To train the DCGAN with the edges dataset run: "python DCGAN_train.py dataRoot".
Replace 'dataRoot' with path to directory that contains the directory that contains
the edges images we created in the first step.
You can run "python DCGAN_train.py -help" to see more arguments you can change like the batch size
and number of epochs.

### 3. Train the pix2pix network to paint the edges.
Train pix2pix network TODO-ROTEM 

### 4. Generate new edges and color them
Now we can generate new edges. In order to do so run "python generate_edges.py Gpath Rpath".
Replace 'Gpath' with path to the DCGAN weights and 'Rpath' with path to the destination directory. 
finally we can use pix2pix TODO TAKE PARAMETERS FROM ROTEM to color the fake edges and get the final results.

You also can use our weights named "checkpointG.pth" and perform only the last two steps above.  

## Usefull links
The frogs dataset was taken from: https://github.com/jonshamir/frog-dataset

The HED network was taken from: https://github.com/sniklaus/pytorch-hed

The DCGAN network based on: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
