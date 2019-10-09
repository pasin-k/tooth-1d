Evaluation of tooth shape for dental procedure using machine learning approach
======
## Overview
This is a collaboration project with Faculty of Dentistry in order to train the dental student to do "Tooth Preparation" process which is a dental procedure to remove outer layer of the tooth.

## Goal
Given input as 3D scanned tooth model (STL file), we want to predict the score of how good the shape of tooth is.
Based on the instructor, there are multiple criteria to consider such as angle of tooth, width of the tooth, etc. 

The final score for each criteria can be 1, 3 or 5 points.

## Algorithm
Due to low amount of dataset, we cannot feed the whole 3D file directly due to too many dimensions. 
Instead, we use *cross-sections* of 3D model from the center of tooth. We also do the same with 45, 90, 135 degree rotation.

In total, we would get 4 cross-sections image from one 3D file. Image below is an example of cross-section image we get from `stl_to_image.py`

<img src="https://github.com/jobpasin/tooth-1d/blob/master/src/images/cross_section_example.png" width="200" height="200">

We remove more redundant information by fetching coordinates of the tooth contour instead and put into 1D-CNN architecture.


## Image Augmentation
We do data augmentation by slightly rotating image from the center with small degree (E.g. 1,2,3 degree).
This process is done by using `get_cross_section` function in 'get_input.py'

# How to use
## Step 1: Data preprocessing
Use `stl_to_image.py` to convert STL file into cross-section images and save as either .png format(image) or .npy (coordinates).
<br> You can select augmentation angles and option to sampling coordinate to specified amount.


## Step 2: Convert to tfrecord
Use 'image_to_tfrecord.py' to convert image or coordinates into .tfrecords file in order to feed to 'train.py'
<br> Can select k_fold option if you want to use k_fold cross validation

## Step 3: Train 
Use 'coor_train.py'. We have multiple mode available.
1. Single run: Standard way to run a model
2. k_fold run: Use this if you select k_fold option in previous step. Will run multiple times
3. Hyperparameter search run: This will run for 20 times with different hyperparameter each time using Scikit-opimize library

### Hyperparameters
There are some hyperparameters and other parameters such as input directory, batch_size, steps, etc in `/cfg` folder. 
We use protocol buffer to feed those parameters into our file. Look up `coor_tooth.config` as a reference. 
<br> Note: label_type is a score category which the instructor gave based on each criteria. You can ask for more info about these label. 


### Model Architecture
For information in model architecture, look up in `coor_model.py` which composed of multiple 1D-CNN and max pooling.


### Output and Evaluation
In the result directory, there is a `result.csv` file which shows all predictions as well as `config.csv` showing accuracy.
<br> To analyze loss and accuracy over iterations, use [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) to see the result.


