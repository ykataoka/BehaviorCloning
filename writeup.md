# **Behavioral Cloning Report by Yasuyuki Kataoka** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[img_model]: ./examples/model_now.png "Model Architecture"
[image_center]: ./examples/center_example.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image_recover1]: ./examples/center1.jpg "Recovery Image"
[image_recover2]: ./examples/center2.jpg "Recovery Image"
[image_recover3]: ./examples/center3.jpg "Recovery Image"
[image_gray]: ./examples/image_gray.jpg "Normal Image"
[image_gray_flip]: ./examples/image_gray_flip.jpg "Flipped Image"
[image_gray_contrast]: ./examples/image_gray_contrast.jpg "contrast adjusted"
[image_gray_blur]: ./examples/image_gray_blur.jpg "bluring"
[image_gray_canny]: ./examples/image_gray_canny.jpg "canny edge"
[train_valid_MSE]:  ./examples/learning_performance.png "train and valid performance"
## Rubric Points : [web site](https://review.udacity.com/#!/rubrics/432/view) 
   
### files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

This project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. The code is usable and readable

The model.py file contains the code for training and saving the convolution neural network.
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the NVIDIA's self-driving architecture.
Here, I show the graph of the architecture of my model (normal size).

| Layer (type)                | Output Shape             | Param # |
| --------------------------- |:------------------------:|:------- |
|lambda_1 (Lambda)            |(None, 160, 320, 1)       |0        |
|cropping2d_1 (Cropping2D)    |(None, 80, 320, 1)        |0        |
|conv2d_1 (Conv2D)            |(None, 40, 160, 24)       |624      |
|spatial_dropout2d_1 (Spatial |(None, 40, 160, 24)       |0        |
|conv2d_2 (Conv2D)            |(None, 20, 80, 36)        |21636    |
|spatial_dropout2d_2 (Spatial |(None, 20, 80, 36)        |0        |
|conv2d_3 (Conv2D)            |(None, 8, 38, 48)         |43248    |
|spatial_dropout2d_3 (Spatial |(None, 8, 38, 48)         |0        |
|conv2d_4 (Conv2D)            |(None, 6, 36, 64)         |27712    |
|spatial_dropout2d_4 (Spatial |(None, 6, 36, 64)         |0        |
|conv2d_5 (Conv2D)            |(None, 4, 34, 64)         |36928    |
|spatial_dropout2d_5 (Spatial |(None, 4, 34, 64)         |0        |
|flatten_1 (Flatten)          |(None, 8704)              |0        |
|dropout_1 (Dropout)          |(None, 8704)              |0        |
|dense_1 (Dense)              |(None, 300)               |2611500  |
|dropout_2 (Dropout)          |(None, 300)               |0        |
|dense_2 (Dense)              |(None, 100)               |30100    |
|dropout_3 (Dropout)          |(None, 100)               |0        |
|dense_3 (Dense)              |(None, 50)                |5050     |
|dropout_4 (Dropout)          |(None, 50)                |0        |
|dense_4 (Dense)              |(None, 1)                 |51       |

* Total params: 2,776,849.0
* Trainable params: 2,776,849.0
* Non-trainable params: 0.0

In the code, the function, normal_model(), defines this model using Keras.

#### 2. Attempts to reduce overfitting in the model

The model uses RELU for activation fuction and dropout 30% data every
after one layer to avoid overfitting. For convolutional layer,
'SpatialDropout2D' is used. Currently, this is not documented in Keras
documentation though, it is written in github and available in Keras
2.0.2.

The model was trained and validated on different data sets to ensure
that the model was not overfitting. The model was tested by running it
through the simulator and ensuring that the vehicle could stay on the
track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned
manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I
used a combination of center lane driving (3 laps), recovering from
the left and right sides of the road (1 laps), driving oppoisite way(1
lap), extremely low speed at the diffcult places, such as sharp corner
and missing lane (1 laps)

Plus, the data is augmented by using 3 different cameras and flipping.
Thus, I used 6 images in total at one sampling time.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try to
find the best performance by changing the following parameters, 
* model architecture (2 Conv + 2 FC, NVIDIA)
* model parameters (I try to use less than 10 million in total for computation speed)
* preprocessing pipiline (gray, blur, contrast, canny)
* mini-batch size (64, 128, 256, 512, 1024)
* the number of dropout and drop percentage
* epoch num (3 to see the performance)

What I fixed is
* resizing (deactivated)
* optimizer (Adam)
* Activation Function(ReLU)
* Loss Function (MSE)

Once GPU and general debugging is done, I started to use the NVIDIA
network.  The input size is quite similar to the data we use on this
simulation. The simulator has more limited information, thus, I
thought it reasonale to use more NVIDIA model to complete simulation
course.


In order to gauge how well the model was working, I split my image and
steering angle data into a training and validation set.

The both result can be interpreted to underfitting or overfitting.
Obviosly, when overfitting, I tweaked dropout and maxpooling. When
underfitting, I changed model architecture or added new dataset.

In my experience, if the validation loss (MAE, for easy
interpretation) is over 0.07, this won't complete. In my dataset,
sometime validation loss(MAE) shows 0.1228 constantly. If I run this,
car does not change the behavior and simply run straight forward. I
think this is the average loss when car does not change the
behavior. Thus, at least half of the performance, if the validation
loss is not less than 0.0614, I did not try to simulate them on test
data in simulator.


The final step was to run the simulator to see how well the car was
driving around track one. There were a few spots where the vehicle
fell off the track where lane line is missing. In order to improve the
driving behavior in these cases, first I tried to use edge
detection. To do it, I also added bluring function. However the
tweaking the parameter for edge detection also affect overall
performance. Thus, I came back to use gray scale image though,
instead, I increased the number of the data at the difficult places.
This is like collecting the corner case data.

At the end of the process, the vehicle is able to drive autonomously
around the track without leaving the road.


#### 2. Final Model Architecture

The final model architecture based on the NVIDIA's self-driving car
model is shown in previous table.


Here is a visualization of the architecture.

![alt text][img_model]


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track
one using center lane driving. Here is an example image of center lane
driving:

![alt text][image_center]

I then recorded the vehicle recovering from the left side and right
sides of the road back to center so that the vehicle would learn to
come back to center from the edge of the lane.

These images show what a recovery looks like starting from heading for right lane to coming back to center lane:

![alt text][image_recover1]
![alt text][image_recover2]
![alt text][image_recover3]

To augment the data sat, I also flipped images and angles thinking that this would gi
ve more flexibility to the model. Originally, there is only one right corner in dataset. This is imbalanced data. In order for the car to turn left and righ with nearly equal performance, flipping is the necessary processing.

.

For example, here is an image that has then been flipped:
![alt text][image_gray]
![alt text][image_gray_flip]

Also, I tried to use several preprocessing module.
Here is the contrast modification to make the image vivid.
![alt text][image_gray_contrast]

Here is the blur modification to remove the noisy road image.
![alt text][image_gray_blur]

Here is the canny edge detection to extract the lane.
![alt text][image_gray_canny]


After the collection process, I had 73398 number of images.

In the final model, I used simple gray scaling only for preprocessing.
Then, I applied the data augmentation mentioned above.
Then, crop the data to remove unnecessary background.

I finally randomly shuffled the data set and put 30% of the data into a
validation set.

I used this training data for training the model. The validation set
helped determine if the model was over or under fitting. The ideal
number of epochs was 20 as evidenced by the following figures.
![alt text][train_valid_MSE]

I used an adam optimizer so that manually training the learning rate
wasn't necessary.
