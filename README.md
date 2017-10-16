# CarND_Project02TrafficSignClassifier
Solution for Udacity Self driving car nano degree first project: Traffic Sign Classifier

---

Traffic Sign Classification.

[//]: # (Image References)

[DSV]: ./Doc_Images/DS_Vis.png
[DSH]: ./Doc_Images/DS_Hist.png
[PP1]: ./Doc_Images/PP1.png
[PP2]: ./Doc_Images/PP2.png
[MYI]: ./Doc_Images/MYI.png

---

### Used abbreviations

GRSD ==> German Road Sign Dataset

DN   ==> Deep Network

### Reflection

Udacity Self Driving Car Nanodegree second project is to build a traffic sign classifier using tensor flow to classify the German traffic sign dataset.

Before discussing the solution to achieve this goal, it is important to understand the problem and how to actually achieve the needed output.

---

### German Road Sign Dataset summary

---

The German Road Sign Dataset includes almost 52,000 images representing 43 different signs (i.e. classes).
* Training dataset contains almost 67% of the images, i.e. 34799 images.
* Validation dataset contains 8.5%, i.e. 4410 images.
* Testing dataset contains 24.5%, i.e. 12360 images.
* All images are 32*32 pixels in RGB format.

it is quite useful to visualize the input data to understand what would be the best way to process it.

![alt text][DSV]

Project rubric has suggested to augment the data be making random rotations, translations, etc.. transformations to the input image to increase input data, however looking at the above samples it is quite clear that the GRSD already has a high variance in lighting, contrast, position and other characteristics of the road signs to begin with, so no need for the augmentation mentioned for this case.

Ok so this is good news, but what about the image classes distribution ? 
After all we don't want to have a large number of inputs of a certain class and very few inputs for other classes as this would lead to a lot of false negatives for the low occurring classes as the network simply didn't have enough examples to learn them.

This is histogram showing the frequency of all 43 classes in the dataset.
![alt text][DSH]

It is clear that the image distribution is not good, with some classes reaching over 2000 examples and some less than 250 examples.
In this case it would be beneficial to actually augment the data of the small occurring class to provide a larger representation for them in the training data.

However for me, I have decided to try to have a small batch size that is continuously shuffled to try to achieve as much variation in the input as possible, p.s: it worked!

---

### Data preprocessing

---

Ok now we can begin to preprocess the GRSD images to ease the job of the DN as much as possible, looking back at the above images
i have decided to work on three points:
  * Sharpening the image
    * it is quite clear that in above images edges are very rough and close to each other, sharpening would help distinguish such features from each other more easily.
  * Down sampling to gray scale to reduce the computational bandwidth needed for the DN.
  * Image values normalization to rescale the pixel values to 0-1 range.
  * Image value standardization to shift the values to have a mean close to 0 and a unit variance distribution.

For the normalization I have used the suggested (pixel - 128) / 128 however calculating the actual mean of the output was quite high around 1.4-1.6 in my tests, using the sklearn normalize function proved to have a much better results with mean around 0.15.

Output of the above preprocessing is:
![alt text][PP1]

Ok they are kind of not that good, first the brightness of the images seems very inconsistent, also the images seems very distorted for actual usage.

Considering this I decided to
  * Add an adaptive histogram equalization step
    * This would help in making brightness as uniform as possible.
  * removing the standardization step from previous pipeline
    * By Testing this was the step causing the large distortions in the images.

Output of the second preprocessing pipe line is:
![alt text][PP2]

This is much better and is quite satisfying for this stage.

---

### Deep Network Design

---

The DN consists of multiple layers (some hidden) and a set of hyper parameters used to tune the network.

### Deep Network architecture

I have started out using LaNet as suggested in the lectures, however no matter how I tried changing the hyper parameters the accuracy would not increase over 92%, I had a theory that increasing the number of filters in the 2 convolutional layers in LaNet would lead to a better performance as the network would have more filters to apply to the input images and thus have a better perception.

This was kind of correct increasing the number of filters initially lead to an increase in accuracy, however after a certain threshold the accuracy would start to decrease again, it is not clear to me why but I think it is an overfitting issue, due to the small batch size I am using (refer reason above) increasing the input filters above a certain threshold (i.e. 32 filters) would lead to the network overfitting and memorizing the input training images.

Taking this in consideration I decided to increase the filter size to a mid range, I have also added a fourth fully connected layer to make sure the relative node counts difference between layers is reasonable, i.e. not having a very large layer fully connected to a very small one as this would result in each neuron in the small layer affecting a very large number of neurons in the large layer during backpropagation which would drastically increase network training time.

Going through the training I noticed that the network was overfitting very quickly due to the small batch size, I decided to add dropout to the network architecture.

With this in mind the final network architecture is as follows:

| Layer              |     Description                               |
|:------------------:|:---------------------------------------------:|
| Input              | 32x32x1 Gray image                            |
| Convolution 5x5    | 1x1 stride, valid padding, outputs 28x28x16   |
| RELU               |                                               |
| Max pooling        | 2x2 stride,  outputs 14x14x16                 |
| Convolution 5x5    | 1x1 stride, valid padding, outputs 10x10x64   |
| RELU               |                                               |
| Max pooling        | 2x2 stride,  outputs 5x5x64                   |
| Flattening         | Flattens the network to 1600                  |
| Fully connected    | Output 540                                    |
| RELU               |                                               |
| DROPOUT            | 0.5 Keep probability                          |
| Fully connected    | Output 120                                    |
| RELU               |                                               |
| DROPOUT            | 0.5 Keep probability                          |
| Fully connected    | Output 84                                     |
| RELU               |                                               |
| DROPOUT            | 0.5 Keep probability                          |
| Fully connected    | Output 43                                     |

I am using the Adam optimizer as suggested in the lectures.

With the above DN and the following set of hyper parameters:

| Parameter          |     Value                                     |
|:------------------:|:---------------------------------------------:|
| EPOCHS             | 50                                            |
| BATCH_SIZE         | 96                                            |
| keepprob           | 0.5                                           |
| Learning Rate      | 0.0008                                        |
| mu                 | 0                                             |
| sigma              | 0.1                                           |

I was able to achieve:

| Set                |     Accuracy                                  |
|:------------------:|:---------------------------------------------:|
| Validation         | 96.7%                                         |
| Test               | 94.7%                                         |
| Five Web Images    | 100%                                          |

Regarding the Five Web Images, they were quite not easy to obtain, I first downloaded some different images and tried to resize them manually using Microsoft paint select tool which proved to be very in-accurate as the images were very distorted to begin with and all of them were misclassified.

I used Gimp to resize the image, it has an automatic tool for that, and performed some manual changes to them to make them easier for classification as increasing contrast, etc..

Below are the 9 images I used:
![alt text][MYI]

All of them were successfully classified and the softmax probabilities were as follows:

| Image              |     Class (Probability                        |
|:------------------:|:---------------------------------------------:|
| 1.png              | 1  (100%)      14 (2.6e-20)    11 (1.5e-29)   |
| 19.png             | 19 (99.8%)     3  (7.3e-05)    29 (3.9e-05)   |
| 22.png             | 22 (99.2%)     25 (6.7e-04)    29 (5.3e-05)   |
| 27.png             | 27 (99.9%)     18 (1.6e-05)    24 (6.2e-07)   |
| 3.png              | 3  (100%)      19 (7.8e-21)    9  (1.6e-21)   |
| 31.png             | 31 (100%)      29 (1.3e-11)    25 (1.5e-16)   |
| 36.png             | 36 (100%)      25 (3.7e-16)    0  (3.6e-22)   |
| 4.png              | 4  (100%)      7  (1.4e-23)    5  (2.1e-24)   |
| 9.png              | 9  (100%)      15 (7.4e-30)    10 (1.0e-31)   |

Although the above results seems very positive, it is to be noted that such very clear, bright, centered images are not considered to be the common input to the DN.

---
