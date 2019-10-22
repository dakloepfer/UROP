# AUDetNet - Recognising Facial Action Units from Single RGB Images

I wrote and implemented this algorithm during my Undergraduate Research Opportunity Project (UROP)
in summer 2019. I was working at the Department of Computer Science and Technology, University of Cambridge, in Hatice Gunes' group.

## Problem Statement

This algorithm is built to recognise Facial Action Units (AUs) from single images of faces.
There are a few datasets containing photos of faces whose action units have been labelled, most notably the DISFA dataset ('DISFA: A spontaneous facial action intensity database' by Mahoor et. al.) and the BP4D dataset ('BP4D-Spontaneous: A high-resolution spontaneous 3D dynamic facial expression database' by Zhang et. al.).

However, both of these only contain samples from a few subjects, and in total only O(150,000) labelled images, due to facial action units being very time-consuming to annotate by hand. This makes overfitting the main difficulty in training an algorithm to recognise AUs reliably in unseen subjects.

## Overview of Algorithm

My main idea was to not use all the information contained in the image of a face to predict the activated AUs, as this would likely lead to the algorithm learning correlations between action units and subject identities, which would not generalise to images of new faces. I therefore first aim to disentangle the unneeded information connected to things like subject identity and head pose from the information on the facial expression (which is obviously needed to predict AUs). The action units are then only predicted based on the latter, a high-dimensional representation of the facial expression.

To disentangle these two types of information, I employ an autoencoder architecture where the internal representation is split in two parts, z1 and z2 (which should ideally contain the information on facial expression and everything else respectively). z1 is used to predict the action units, while z2 serves as a contrasting vector to encourage the network to allow z1 to contain the 'right' information.

The disentangling then is incentivised by adding a term to the loss function that penalises any mutual information between z1 and z2. Without knowing the exact distributions for the embeddings, it is not possible to calculate the mutual information explicitly, so I use the fact that the mutual information is the KL-Divergence between the joint distribution over z1 and z2, and their marginal distributions. When training on batches of images, we can approximate the joint distribution by simply the created embeddings (pairs of z1 and z2) and the marginal distributions by the z1 and z2 'shuffled' (i.e. a given z1, z2 pair now is not created from the same image). This a very similar task to that solved by a GAN, namely changing one distribution (the original z1 z2 pairs) to be as similar as possible to a different distribution (the shuffled z1 z2 pairs). So we train the network to do this adversarially, using a discriminator that aims to predict whether a given (z1 z2) pair is 'original' or was shuffled.

Requiring the network to also reconstruct the original image from z1 and z2 ensures that no (important) information in the image is forgotten by the network, which could rob z2 of its value as a representation of 'information not to contain' for z1.

## Quick Start

### Dataset Organisation

For training, a dataset directory must be specified. This directory should contain:

- pickled dictionary containing (key, value) pairs where the key is the frame id (e.g. SN005_43) and value is a numpy array with the action unit activations
- test_ids.csv and train_ids.csv that contain frame ids for training and test sets respectively, one ID per row
- imgs folder: containing all images in the format frame_id.jpg

These files can be created using the scripts in the create_dataset folder.

The directories bp4d_dataset and disfa_dataset contain the necessary metadata for the respective datasets (but not the actual images), and minimal_dataset is a sample dataset with 12 images from DISFA, for testing and debugging purposes.

### cfg File

The cfg files, saved in the cfg directory, contain all information on the detailed architecture of the network (ie which layers where), as well as some information on training and loss function.

The different sections are deliminated by a header in curly braces (e.g. one header for a new network part), and under each header there is a series of 'blocks', who start with a header in square brackets.

#### Overall Format

```python
{General}

[training_options] # this name doesn't really matter, the position does
batch_size=INTEGER
max_epochs=INTEGER
max_test_iters=INTEGER
train_main_net_every_n_batches=INTEGER
learning_rate_scheduler=None | Linear # other schedulers possible but not yet implemented
decay_lr_over_last_n_epochs=INTEGER # Only needed if use Linear learning rate scheduler

[data_options]
regress_au_intensities=True | False # Boolean deciding if we regress the AU intensities or only predict their occurrence
balance_dataset_with_momu=True | False # Boolean deciding if we apply MOMU before each epoch
epoch_size=INTEGER # Integer determining the size of one epoch; only needed if balance_dataset_with_momu=True
sampling_size=INTEGER # Integer determining sampling size for MOMU; must divide epoch_size

# Optional: List of means and stdevs to which the input tensors are normalised.
# None, only one, or both can be specified.
# Defaults are mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
# When using ResNet50 as first block, use mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# When using VGG-Face as first block, use mean=[129.186279296875, 104.76238250732422, 93.59396362304688], std=[1, 1, 1]
mean=FLOAT,FLOAT,FLOAT
std=FLOAT,FLOAT,FLOAT

[loss_options]
au_regression_loss_function=MSE | BCEWithLogits
reconstruction_loss_function=L1
gradient_loss_lambda_discr=FLOAT
gradient_loss_lambda_img_discr=FLOAT
discriminator_loss_lambda=FLOAT
img_discriminator_loss_lambda=FLOAT
reconstruction_loss_lambda=FLOAT

{Encoder}

[net]
learning_rate=FLOAT
beta1=FLOAT
beta2=FLOAT
channels=INTEGER # number of input channels

# other blocks in here


{AUPredictor}

[net]
learning_rate=FLOAT
beta1=FLOAT
beta2=FLOAT
channels=INTEGER # number of input channels

# other blocks in here


{Decoder}

[net]
learning_rate=FLOAT
beta1=FLOAT
beta2=FLOAT
channels=INTEGER # number of input channels

# other blocks in here


{Discriminator}

[net]
learning_rate=FLOAT
beta1=FLOAT
beta2=FLOAT
channels=INTEGER # number of input channels

# other blocks in here

{Image Discriminator} # optional, adversarially improves generated image quality

[net]
learning_rate=FLOAT
beta1=FLOAT
beta2=FLOAT
channels=INTEGER # number of input channels

# other blocks in here

```


#### Implemented Block Types

Something helpful when creating convolutional networks:
output_width = (input_width - kernel_size + 2*padding)/stride + 1

```python
# Convolutional Layer
[convolutional]
batch_normalise=0 | 1 # Boolean if we batch normalise
filters=INTEGER # Number of output filters
kernel_size=INTEGER # kernel size used
stride=INTEGER # Stride
# Integer giving padding or Boolean deciding whether we do zero-padding
pad=INTEGER | True | False
activation=relu | leaky_relu | tanh | sigmoid | None | not_implemented # Activation used
initialization=None | kaiming_uniform # initialization for weights, default is None

# Single Batch Normalisation Layer
[batch_norm]
num_features=INTEGER # Number of channels
# epsilon to be used in Batch norm, can also be written as e.g 1e-5
eps=FLOAT

# Max Pooling Layer
[maxpool]
pool_size=INTEGER # Pooling size
stride=INTEGER # Stride used

# Average Pooling
[avgpool]
pool_size=INTEGER # Pooling size
stride=INTEGER # Stride used

# Upsampling
[upsample]
output_sizes=None | INTEGER, INTEGER # optional output sizes as in nn.Upsample()
scale_factor=FLOAT # scale factor as in nn.Upsample()
mode=nearest | linear | bilinear | bicubic | trilinear # Mode as in nn.Upsample()

# Fully Connected Layer
[connected]
output_features=INTEGER # number of output features
activation=linear | relu | leaky_relu | sigmoid # Activation function
use_bias=True | False # Whether to use bias in Linear layer
initialization=None | kaiming_uniform # initialization for weights, default is None

# Softmax Layer
[softmax]
dim=INTEGER # dimension along which to compute softmax

# Routing Layer
# This layer splits off a branch at the layer given by branch_at_layer
# and all blocks from now on until another [route] or a [combine_concat]
# are applied on the split-off branch.
[route]
# if branch_at_layer < 0, we branch off at the block branch_at_layer before
# this one (e.g. -2 would mean we branch off at the block two before this one)
# if branch_at_layer >= 0, we branch off at the block branch_at_layer from the
# beginning (0-indexing, ignoring the [net] block)
branch_at_layer=INTEGER

# Combining by Concatenating Layer
# This layer concatenates the outputs of all the blocks in combine_layers
# (indexed in the same way as in [route]) by concatenating along dimension
[combine_concat]
combine_layers=INTEGER, INTEGER, ...
# must be between 0 and 4, standard would be dimension=1, referring to channels  
dimension=INTEGER

# Combining by Multiplication Layer
# This layer combines the outputs of all the blocks in combine_layers
# (indexed in the same way as in [route]) by element-wise multiplication
# Usually requires shapes of layers to be the same
[combine_multiply]
combine_layers=INTEGER, INTEGER, ...

# Select Subset Layer
# This layer selects a subset of origin_layer (indexed in the same way as in [route])
# and propagates it forward. The subset is input_tensor[..., start_index:end_index, ...],
# with the dimension masked being chosen by dimension.
[select_subset]
origin_layer=INTEGER
dimension=INTEGER # dimension that is masked, standard is 1 (i.e. channels)
start_index=INTEGER
end_index=INTEGER

# Identity Layer
# This layer simply implements the identity, for when we need a placeholder layer.
# It obviously needs no parameters
[identity]

# Pretrained ResNet50 Layer
# This layer loads a pretrained ResNet50 model and adds this as a single layer.
# Note: When using this as a stem network, the tensors need to be normalised to
# mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
[resnet50]
# Layer up to which we use the resnet50 layers. 
# The block that will be created will be nn.Sequential(*resnet.children()[:use_layers])
end_at_layer=INTEGER
freeze_first_n_layers=INTEGER # Number of layers in ResNet to freeze (requires_grad = False)
n_out_filters=INTEGER # Number of output channels at the end of the block

# Pretrained VGG-Face Layer
# This layer loads a pretrained VGG-Face model based on VGG16 and adds this as a single layer.
# It only loads up to including the last MaxPool2d Layer and excluding the first fully connected layer,
# to allow for different input image sizes. The number of output filters is therefore fixed at 512.
# Note: When using this as a stem network, the tensors need to be normalised to
# mean=[129.186279296875, 104.76238250732422, 93.59396362304688], std=[1, 1, 1]
[vgg-face]
# Path to file containing the state_dict as downloaded from
# http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.pth
state_dict_filepath=STRING
freeze_first_n_layers=INTEGER # Number of layers in VGG-Face to freeze, the number includes any ReLU and Pooling Layers
```

### Training

The main training script is train.py. test_disentangling.py contains some methods to test how well a given trained network disentangled information between z1 and z2.
