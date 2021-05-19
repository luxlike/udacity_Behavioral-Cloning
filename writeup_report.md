# **Behavioral Cloning** 

---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/cnn-architecture-624x890.png "Model Visualization"
[image2]: ./examples/center_driving.png "Center Driving"
[image3]: ./examples/left_driving.png "Left Driving"
[image4]: ./examples/right_driving.png "Right Driving"
[image5]: ./examples/origin.png "Origin Image"
[image6]: ./examples/flipped.png "Flipped Image"
[image7]: ./examples/cropped.png "Cropped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md  summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model uses the NVIDIA architecture of a deep neural network for autonomous driving. It consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths 24,36,48 and 64 (code in make_model() ).

The model includes RELU layers to introduce nonlinearity, the images are cropped and the data is normalized in the model using a Keras lambda layer (code in create_preprocessing_layers() ).

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 146). The dataset consists of 80% of training data and 20% of validation data. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (code line 158).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving three times, recovering from the left and right sides of the road one time, counter-clockwise driving and driving smoothly arround curves.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the provided NVIDIA architecture, the suggested image augmentation and strategies for data collection.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. SO I  improve the driving behavior in these cases, I recovering from the left and right sides of the road, counter-clockwise driving and driving smoothly arround curves.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with a normalization layer, three 5x5 convolution layers, two 3x3 convolution layers and 3 Full-Connected Layers and  Output Layer.

Here is a visualization of the architecture (taken from NVIDIA Develper site):

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to navigate back to the center. These images show what a recovery looks like: :

![alt text][image3]
![alt text][image4]

I  recorded one lap of counter-clockwise driving and one lap of smooth curve driving. 

To augment the data set, I also flipped images and angles (code line 97-98)
 For example, here is an image that has then been flipped::

![alt text][image5]
![alt text][image6]

In order to reduce distraction for the neural network and concentrate only on the road, I cropped 75px from the top and 25px from the bottom of each pixel:

![alt text][image5]
![alt text][image7]

After the collection process, I had 11525 number of data points. 
I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3.
I used an adam optimizer so that manually training the learning rate wasn't necessary.