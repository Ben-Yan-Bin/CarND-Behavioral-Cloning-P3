# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* run1.mp4 is the video record of one lap on track in autonomous mode

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network which is derived from AlexNet, with several convolutional layers and added with several fully connected layers. The output is a Dense(1) layer to predict the steering angle for each image. 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. Also, at the very beginning of the network, 2D Crop layer is added, to cut off the unnecessary information on top and bottom of the images to let the NN more focus on the road.

#### 2. Attempts to reduce overfitting in the model

The model contains Batch Normalization layer of Keras to reduce overfitting. 

The training data were splitted by 9:1 as trained and validated data sets to validate the loss was reduced along the way of the training. In the Keras fit function, a callback function was called at each end of the epoch based on the valication loss value, and to save the model when there is a better validation value generated. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer. The epoch was set to be large since callback function was used to save model with good validation result at each end of epoch

#### 4. Appropriate training data

I found the default data in the project can work quite well when I first try, then I just do some more data augmentation and use left, right images to make the date more sufficient, and it works.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Since Alexnet is was very successful for imagenet, and not too complex, I decided to give it a try, and also I added some more small fully connected layers to the top of the net.   
I think the track images are not complicated, and think Alexnet is sophisticated enough to recognize the images, and to learn the steering angles.

At the beginning of the training, the training set loss dropped quickly, and I assume the Alexnet-like NN is good enough for this case.
When I had test with the first version of the model, the car was much better than I thought, and drove to the bridge but stuck there off the road.   
Also the validation loss was higher than training loss after several epochs that I realized overfitting problem.
Then I added batch normalization at the end of each conv layer to reduce overfitting.


#### 2. Final Model Architecture

The final model architecture is totally based on Alexnet, addtionally with 2D crop, Lambda, on the bottom, Batch normalization for each conv layer, and 3 more Dense layer on the top with Dense(1) as output. 

Detailed model arch see model.summary() output.

#### 3. Creation of the Training Set & Training Process

I first pick 10% of the dataset as validation in the final training since I think the dataset is already big enough (7-8K), and 10% out of it can make a reliable validation.    
And I flipped the images and steering angle to help make it more generalized.  
   
The model went well in the straight lines, and easy turns at first few tries, but could not get through the hard corners to make proper turns and also did not do well to recover from side of the road to the center.   
   
Then I used the left and right images, I think they could help model to learn how to pass the corner and recover from side of the road because the images from the left camera and right camera is more sided to the road than the center images, and I used the as 0.05 correction of steering for left and right images, don't want to make it too much because I think the sided images are not so different than the center images.   
   
Also, finally I used the fatest mode of the simulator to test, and the result is good enough to make full laps without getting off the road.
