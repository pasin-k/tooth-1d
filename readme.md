Evaluation of tooth shape for dental procedure using machine learning approach
======
## Overview
This is a collaboration project with Faculty of Dentistry to evaluate the skill of the dental student to do "Tooth Preparation" process, a dental procedure to remove outer layer of the tooth into an appropriate shape.

## Data
We have  3D scanned tooth model (STL file) of students, with the score given on multiple criteria such as angle of tooth, amount of tooth removed, the sharpness of tooth, etc. 
The score for each criteria can be 1, 3 or 5 points.

## Algorithm
Due to low amount of dataset, we cannot feed the whole 3D file directly due to too large dimensions. 
Instead, we use *cross-sections* of 3D model from the center of tooth. We retrieve multiple cross-section around z-axis, then rotate the plane around z-axis at different angles. (0, 45, 90, 135 degree)

In total, we would get 4 cross-sections image from one 3D file. Image below is an example of cross-section image we get from `stl_to_image.py`

<img src="https://github.com/jobpasin/tooth-1d/blob/master/src/images/cross_section_example.png" width="200" height="200">

We further reduce input dimension by fetching coordinates of the tooth contour instead. For each cross-section image, we will have the data of x-axis and y-axis of 300 sampled coordinates. 

Then, we put the model into 1D-CNN with architecture belows:
<img src="https://github.com/jobpasin/tooth-1d/blob/master/src/images/1dcnn_architecture.png" width="200" height="200">

## Data Augmentation
We do data augmentation by slightly rotating image from the center with small degree (E.g. 1,2,3 degree).
This process is done by using `get_cross_section` function in 'get_input.py'

# How to use
## Step 1: Data preprocessing (STL File --> Image or 1D data)
Use `preprocess/stl_to_image.py` to convert STL file into cross-section images and save as either .png format(image) or .npy (coordinates).
<br> You can select augmentation angles and option to sampling coordinate to specified amount.


## Step 2: Convert to tfrecord (Image or 1D data --> tfrecords)
Use 'image_to_tfrecord.py' to convert image or coordinates into .tfrecords file in order to feed to 'train.py'
<br> Can select k_fold option if you want to use k_fold cross validation

## Step 3: Train 
Use `coor_train.py`. We have multiple mode available.
1. Standard: Standard way, train once.
2. K-fold: Use this if you select k_fold option in previous step. Will run multiple time for k-fold cross validation.
3. Hyperparameter search run: Multiple runs with different hyperparameter each time using Scikit-opimize library
4. Grid search: Hyperparameter search same as above but using grid-search approach

Our training process uses `tf.Estimator` API for doing all the training/evaluation process. If you are not familiar with this API, you can lookup [Here](https://www.tensorflow.org/guide/estimator)

### Hyperparameters
Hyperparameters and other parameters configs used for `coor_train.py` can be found in `/cfg` folder and is needed to be parsed.
Hyperparameter such as input directory, batch_size, steps, etc.

We use protocol buffer to feed those parameters into our file. Look up `coor_tooth.config` as a reference. 
<br> Note: label_type is a score category which the instructor gave based on each criteria. You can ask for more info about these label. 


### Model Architecture
For information in model architecture, look up in `coor_model.py` which is AlexNet-like version of 1D-CNN layers.


### Output and Evaluation
In the result directory, a `result.csv` file will be generated which shows all validation predictions. `config.csv` shows the accuracy and training settings.
<br> To analyze loss and accuracy over iterations, use [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) to see the result.

### Testing
Use `coor_predict` for prediction. This accepts 3 types of data: .stl file, .npy file, and .tfrecord file. Using the first two type of data will take slightly longer time due to data transformation.
