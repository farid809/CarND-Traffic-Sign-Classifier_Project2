
# Self-Driving Car Engineer Nanodegree

## Project: Build a Traffic Sign Recognition Classifier


### Overview
In this project I'm using deep neural networks and convolutional neural networks to classify traffic signs. I provided below step by step guide showing the approach i used to to train and validate a model so it can classify traffic sign images using the German Traffic Sign Dataset.  


### Project Goals 

steps of this project are the following:

    * Load the data set (see below for links to the project data set)
    * Explore, summarize and visualize the data set
    * Design, train and test a model architecture
    * Use the model to make predictions on new images
    * Analyze the softmax probabilities of the new images
    * Summarize the results with a written report


```python
#imports

# Load pickled data
import pickle               
import csv
import pandas as pd
import numpy as np

import random
import math


import os
import os.path
import sys

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import ImageOps
from PIL import Image

from scipy.misc import imresize
from skimage import data
from skimage import color
from skimage import img_as_float



import tensorflow as tf
from tensorflow.contrib.layers import flatten

```

## Step 0: Load The Data


```python



# TODO: Fill this in based on where you saved the training and testing data

training_file = './traffic-signs-data/train.p'
validation_file='./traffic-signs-data/valid.p'
testing_file = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_validation, y_validation = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


```

---

## Step 1: Dataset Summary & Exploration

The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**



### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

### Question 1.1 -  Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy library to calculate summmary statistics of the traffic signs data set


* Number of training examples = 34799
* Number of testing examples = 12630
* Number of validation examples = 4410
* Image data shape = (32, 32, 3)
* Number of unique classes = 43



```python

assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))


# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_validation)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print()
print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Number of validation examples =", len(X_validation))
print("Image data shape =", image_shape)
print("Number of unique classes =", n_classes)
```

    
    Number of training examples = 34799
    Number of testing examples = 12630
    Number of validation examples = 4410
    Image data shape = (32, 32, 3)
    Number of unique classes = 43
    

### Question 1.2. Include an exploratory visualization of the dataset.

I implemented mutiple helper function to help with visualizing data set images and mapping image labels from the provided CSV file.

* displayCSV
* getSignNameById
* showSampleImages


```python
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.


# Visualizations will be shown in the notebook.
%matplotlib inline


#Get Sign Name By ClassID
def displayCSV(fileName):
    signs_dict={}
    with open(fileName, 'rt') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        print("Label ID         SignName");
        for row in reader:
            print(row['ClassId']+"         "+row['SignName'])
            #signs_dict[row['ClassId']]=row['SignName']
    #return signs_dict[Id]


#Get Sign Name By ClassID
def getSignNameById(Id):
    signs_dict={}
    with open('signnames.csv', 'rt') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            signs_dict[row['ClassId']]=row['SignName']
    return signs_dict[Id]




#Show random Traffic Sign images from the Train Data Set
def showSampleImage (imageList                   ,   # X_train
                     imageTitleList              ,   # y_train
                     number_of_samples  =  10    ,   # Limit number of visualized samples (Optional)
                     randomize          =  False ,   # Pick random images from list (Optional)
                     filter_class       = None   ,   # Filter the image list for specific class (Optional)
                     actual_title_list  = None   ,   # Used to determine the correct prediction and highlight image border (Green/Red) 
                     hSize              = 16     ,   # Image High (Optional)
                     wSize              = 16     ,   # Image Width (Optional)
                     n_row              = None   ,   # force number of rows (Optional)
                     image_title        = True   ) : # Disable/Enable image title
    hSize=16
    wSize=16
    GREEN = [0,255,0]
    RED = [255,0,0]


    
    if number_of_samples<= 5:
        number_of_col = number_of_samples
        number_of_row = 1
    else:
        number_of_col = 5
        number_of_row = math.ceil(number_of_samples/number_of_col)
        
    if n_row != None:
        number_of_row = n_row
        number_of_col = number_of_samples
        
        
    fig, axs= plt.subplots(number_of_row,
                           number_of_col, 
                           figsize=(hSize,wSize))
   
    fig.subplots_adjust(hspace = 0.1, 
                        wspace = 0.1 )
    axs = axs.ravel()

        
    
    cnt=0
    if filter_class !=None:
        if randomize == True :
            images_in_this_sign_class = imageList[imageTitleList == filter_class, ...]
            filtered_image_title_list = imageTitleList[imageTitleList == filter_class, ...]
            #print(len(images_in_this_sign_class))
            #print(getSignNameById(str(filter_class)))
            for i in range(0,number_of_samples):
                index = random.randint(0, len(images_in_this_sign_class)-1)
                image = images_in_this_sign_class[index]   
                axs[i].axis('off')
                axs[i].imshow(image)
                if image_title != False:
                    axs[i].set_title(getSignNameById(str(filtered_image_title_list[index])))
            
        else :    
            for i in range(0,len(imageList)):
                if imageTitleList[i] == filter_class:
                    image = imageList[i]
                    axs[cnt].axis('off')
                    axs[cnt].imshow(image)
                    #axs[cnt].set_title(getSignNameById(str(imageTitleList[index]))+' '+str(image.shape))
                    if image_title != False:
                        axs[cnt].set_title(getSignNameById(str(imageTitleList[i])))
                    cnt+=1
                    if cnt>number_of_samples-1:
                        break
    elif filter_class == None:             
        for i in range(0,number_of_samples):
            index = i
            if randomize:
                index = random.randint(0, len(imageList))
            image = imageList[index]
            axs[i].axis('off')
            if actual_title_list != None:
                if imageTitleList[index] == actual_title_list[index] :
                    image= cv2.copyMakeBorder(image,2,2,2,2,cv2.BORDER_CONSTANT,value=GREEN)
                else:
                    image= cv2.copyMakeBorder(image,2,2,2,2,cv2.BORDER_CONSTANT,value=RED)
        
            axs[i].imshow(image)
            #axs[i].set_title(getSignNameById(str(imageTitleList[index]))+' '+str(image.shape))
            axs[i].set_title(getSignNameById(str(imageTitleList[index])))
            





```

#### Visualizing random images from the dataset using the previously defined helper function "ShowSampleImage"


```python
#show  random Images from the data set            
showSampleImage(X_train,y_train,number_of_samples=20,randomize=True,filter_class=None)

```


![png](output_11_0.png)


#### show  random Images from the data set filter for a specific class   
Let's explore samples from each class


```python
#show  random Images from the data set filter for a specific class   
#Let's explore samples from each class
showSampleImage(X_train,y_train,10,filter_class=0,randomize=True,hSize=8,wSize=8,n_row=1,image_title=False)
showSampleImage(X_train,y_train,10,filter_class=5,randomize=True,hSize=8,wSize=8,n_row=1,image_title=False)
showSampleImage(X_train,y_train,10,filter_class=10,randomize=True,hSize=8,wSize=8,n_row=1,image_title=False)

showSampleImage(X_train,y_train,10,filter_class=15,randomize=True,hSize=8,wSize=8,n_row=1,image_title=False)
showSampleImage(X_train,y_train,10,filter_class=20,randomize=True,hSize=8,wSize=8,n_row=1,image_title=False)
showSampleImage(X_train,y_train,10,filter_class=25,randomize=True,hSize=8,wSize=8,n_row=1,image_title=False)

showSampleImage(X_train,y_train,10,filter_class=30,randomize=True,hSize=8,wSize=8,n_row=1,image_title=False)
showSampleImage(X_train,y_train,10,filter_class=35,randomize=True,hSize=8,wSize=8,n_row=1,image_title=False)
showSampleImage(X_train,y_train,10,filter_class=40,randomize=True,hSize=8,wSize=8,n_row=1,image_title=False)

```


![png](output_13_0.png)



![png](output_13_1.png)



![png](output_13_2.png)



![png](output_13_3.png)



![png](output_13_4.png)



![png](output_13_5.png)



![png](output_13_6.png)



![png](output_13_7.png)



![png](output_13_8.png)


#### Display CSV file content


```python
displayCSV('signnames.csv')

```

    Label ID         SignName
    0         Speed limit (20km/h)
    1         Speed limit (30km/h)
    2         Speed limit (50km/h)
    3         Speed limit (60km/h)
    4         Speed limit (70km/h)
    5         Speed limit (80km/h)
    6         End of speed limit (80km/h)
    7         Speed limit (100km/h)
    8         Speed limit (120km/h)
    9         No passing
    10         No passing for vehicles over 3.5 metric tons
    11         Right-of-way at the next intersection
    12         Priority road
    13         Yield
    14         Stop
    15         No vehicles
    16         Vehicles over 3.5 metric tons prohibited
    17         No entry
    18         General caution
    19         Dangerous curve to the left
    20         Dangerous curve to the right
    21         Double curve
    22         Bumpy road
    23         Slippery road
    24         Road narrows on the right
    25         Road work
    26         Traffic signals
    27         Pedestrians
    28         Children crossing
    29         Bicycles crossing
    30         Beware of ice/snow
    31         Wild animals crossing
    32         End of all speed and passing limits
    33         Turn right ahead
    34         Turn left ahead
    35         Ahead only
    36         Go straight or right
    37         Go straight or left
    38         Keep right
    39         Keep left
    40         Roundabout mandatory
    41         End of no passing
    42         End of no passing by vehicles over 3.5 metric tons
    


```python
# Train data set - frequency distribution
plt.subplot(121)
plt.hist(y_train, n_classes)
plt.xlabel('Label id')
plt.ylabel('Frequency')
plt.title('(Original) train data set')
plt.axis([-0.25, n_classes-.5, 0,2500])
plt.grid(False)

# Test data set - frequency distribution
plt.subplot(122)
plt.hist(y_test, n_classes)
plt.xlabel('Label id')
plt.title('Test data set')
plt.axis([-0.25, n_classes-.5, 0,800])
plt.grid(False)

plt.tight_layout()
print (" \n")
print("                   FREQUENCY DISTRIBUTION OF SIGN LABELS\n")
plt.show()
```

     
    
                       FREQUENCY DISTRIBUTION OF SIGN LABELS
    
    


![png](output_16_1.png)


## Data Set Augmentation


```python
## Todo
# Augment the dataset by rotating or shifting images or by changing colors

'''

[References for image effects]

http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
https://github.com/aleju/imgaug/tree/master/imgaug/augmenters
https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
https://www.pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/
https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/

'''

#Dictionary used to collect frequency of each applied effect during augmentation process
augmentation_stats={'Translate':0,'Rotate':0,'Affin_Transform':0,'Blur':0,'Gamma_Correction':0}


def image_effect(img, image_effect=None):
    
        '''
        -> Perform one of the image effects as specified by image_effect param. 
        -> choose a random effect if image_effect=None
        
        image_effect :
                Invert
                *Add
                *Multiply
                GaussianBlur
                MedianBlur
                Sharpen
                AdditiveGaussianNoise
                *Dropout
                Rotate
                Affin Transform
                Posterize
                Solarize
                Flip
                Equalize
                mirror
        '''
        
        #select a random effect if none specified
        #Active Random Image Effect
        if image_effect == None:
            image_effect = random.choice(['Translate','Rotate','Affin_Transform','g_Blur','Gamma_Correction'])
            
        #AutoContrast
        if image_effect == 'AutoContrast':
            pil_image=Image.fromarray(img)
            pil_image= ImageOps.autocontrast(pil_image)
            img= np.array(pil_image)
            
            
        #Posterize
        if image_effect == 'Posterize':
            pil_image=Image.fromarray(img)
            pil_image= ImageOps.posterize(pil_image,2)
            img= np.array(pil_image)

            
            
        #Invert
        if image_effect == 'Invert':
            pil_image=Image.fromarray(img)
            pil_image= ImageOps.invert(pil_image)
            img= np.array(pil_image)
            
        #Solarize
        if image_effect == 'Solarize':
            pil_image=Image.fromarray(img)
            pil_image= ImageOps.solarize(pil_image,threshold=128)
            img= np.array(pil_image)
            
        #flip Vertically
        if image_effect == 'Flip':
            pil_image=Image.fromarray(img)
            pil_image= ImageOps.flip(pil_image)
            img= np.array(pil_image)
            
        #Mirror
        if image_effect == 'Mirror':
            pil_image=Image.fromarray(img)
            pil_image= ImageOps.mirror(pil_image)
            img= np.array(pil_image)
            
            
        #equalize
        if image_effect in 'Equalize' :
            pil_image=Image.fromarray(img)
            pil_image= ImageOps.equalize(pil_image)
            img= np.array(pil_image)
            
            
        #Brightness
        if image_effect == 'Brightness':
            hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
            hsv[:,:,2] = hsv[:,:,2]*.25+np.random.uniform()
            img = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
            augmentation_stats['Brightness'] +=1
        
        #Translate
        #https://docs.opencv.org/3.1.0/da/d6e/tutorial_py_geometric_transformations.html
        if image_effect == 'Translate':
            x = img.shape[0]
            y = img.shape[1]

            x_trans = np.random.uniform(-0.2 * x, 0.2 * x)
            y_trans = np.random.uniform(-0.2 * y, 0.2 * y)

            trans_matrix = np.float32([[1, 0, x_trans], [0, 1, y_trans]])
            trans_img = cv2.warpAffine(img, trans_matrix, (x, y))
            img = trans_img
            augmentation_stats['Translate'] +=1
        
        #Rotate
        #http://answers.opencv.org/question/173844/rotation-bicubic-resampling-using-opencv-python/
        if image_effect == 'Rotate':
            rows, cols, ch = img.shape
            rotation_matrix_2d = cv2.getRotationMatrix2D((rows / 2, cols / 2), np.random.uniform(-45, 45), 1)
            dst = cv2.warpAffine(img, rotation_matrix_2d, (cols, rows))
            img = dst
            augmentation_stats['Rotate'] +=1
        

            
        #Affin_Transform
        if image_effect == 'Affin_Transform':
            rows, cols, ch = img.shape
            affin_trans = np.random.randint(5,15)
            pts1 = np.array([[5, 5], [20, 5], [5, 20]]).astype('float32')
            pt1 = 5 + affin_trans * np.random.uniform() - affin_trans / 2
            pt2 = 20 + affin_trans * np.random.uniform() - affin_trans / 2
            pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
            M = cv2.getAffineTransform(pts1,pts2)
            dst = cv2.warpAffine(img,M,(cols,rows))
            img = dst
            augmentation_stats['Affin_Transform'] +=1
                
        #Gaussian_Blur
        if image_effect == 'g_Blur':
            r_int = np.random.randint(0, 2)
            odd_size = 2 * r_int + 1
            img = cv2.GaussianBlur(img, (odd_size, odd_size), 0)
            augmentation_stats['Blur'] +=1
            
        #Median Blur
        if image_effect == 'm_Blur':
            r_int = np.random.randint(0, 2)
            odd_size = 2 * r_int + 1
            img = cv2.medianBlur(img, (odd_size, odd_size), 0)
            augmentation_stats['Blur'] +=1
        
        
        #Gamma
        # https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
        if image_effect == 'Gamma_Correction':
            gamma = np.random.uniform(0.1, 3.0)
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            img = cv2.LUT(img, table)
            augmentation_stats['Gamma_Correction'] +=1

        return img
```


```python
'''
Visualize different image effects.

'''


fig, axs= plt.subplots(4,3, figsize=(32,32))
fig.set_size_inches((8.27, 15))
fig.subplots_adjust(hspace = 0.8, wspace =  0.3)
axs = axs.ravel()

image_effects=['Original','Mirror','Flip','Invert','Solarize','Equalize','Translate','Rotate','Affin_Transform','g_Blur','Gamma_Correction','Posterize']

for i,fx in enumerate(image_effects):
    axs[i].imshow(image_effect(X_train[3000],image_effect=fx))
    axs[i].set_title(fx)
    
    
plt.show()
```


![png](output_19_0.png)


#### Image Augmentation Process

* Calculate an equal target count for all image classes
* For each Image class in the dataset augment the class set with new images using Random image from the original set and applying randome effect until total count for the class reach the target count calculated in step 1. 
* Concatenate the new generated data set to the original set.


```python


    
#Dictionary used to collect frequency of each applied effect during augmentation process
augmentation_stats={'Translate':0,'Rotate':0,'Affin_Transform':0,'Blur':0,'Gamma_Correction':0}


unique_ids,sign_counts=np.unique(y_train, return_counts=True)

#The final count of image labels = Max frequency + 2000`
#to avoid data set that result in opinionated model
traget_max_count=np.amax(sign_counts)


X_train_augmented = np.array(np.zeros((1, 32, 32, 3)))
y_train_augmented = np.array([0])

print("Ids   Cnt  Target   Aug_Stats")
print("===   ===  ======   ==========")

for sign_id in unique_ids:
    X_train_by_classId = X_train[y_train == sign_id, ...]
    
    batch_images = []

    
    #Count of augmented images to reach target_max_count
    aug_count=traget_max_count - len(X_train_by_classId)
    
    for i in range(aug_count):
            batch_images.append(image_effect(X_train_by_classId[random.randint(0,len(X_train_by_classId)-1)]))
            
    if len(batch_images) > 0:
        aug_images  = np.concatenate((X_train_by_classId, batch_images), axis=0)
        
    aug_labels  = np.full(len(aug_images), sign_id, dtype='uint8')
    
    
    X_train_augmented   = np.concatenate((X_train_augmented, aug_images), axis=0)
    y_train_augmented = np.concatenate((y_train_augmented, aug_labels), axis=0)
    
    
    print("{}    {}   {}    {}".format(sign_id,len(X_train_by_classId),len(aug_images),augmentation_stats))

    
X_train_augmented = X_train_augmented/255
       
```

    Ids   Cnt  Target   Aug_Stats
    ===   ===  ======   ==========
    0    180   2010    {'Affin_Transform': 352, 'Translate': 346, 'Gamma_Correction': 377, 'Blur': 365, 'Rotate': 390}
    1    1980   2010    {'Affin_Transform': 357, 'Translate': 351, 'Gamma_Correction': 384, 'Blur': 373, 'Rotate': 395}
    2    2010   2010    {'Affin_Transform': 357, 'Translate': 351, 'Gamma_Correction': 384, 'Blur': 373, 'Rotate': 395}
    3    1260   2010    {'Affin_Transform': 504, 'Translate': 505, 'Gamma_Correction': 534, 'Blur': 514, 'Rotate': 553}
    4    1770   2010    {'Affin_Transform': 556, 'Translate': 559, 'Gamma_Correction': 583, 'Blur': 564, 'Rotate': 588}
    5    1650   2010    {'Affin_Transform': 617, 'Translate': 625, 'Gamma_Correction': 657, 'Blur': 648, 'Rotate': 663}
    6    360   2010    {'Affin_Transform': 948, 'Translate': 941, 'Gamma_Correction': 981, 'Blur': 994, 'Rotate': 996}
    7    1290   2010    {'Affin_Transform': 1114, 'Translate': 1087, 'Gamma_Correction': 1119, 'Blur': 1145, 'Rotate': 1115}
    8    1260   2010    {'Affin_Transform': 1280, 'Translate': 1232, 'Gamma_Correction': 1250, 'Blur': 1297, 'Rotate': 1271}
    9    1320   2010    {'Affin_Transform': 1408, 'Translate': 1369, 'Gamma_Correction': 1394, 'Blur': 1442, 'Rotate': 1407}
    10    1800   2010    {'Affin_Transform': 1440, 'Translate': 1420, 'Gamma_Correction': 1444, 'Blur': 1472, 'Rotate': 1454}
    11    1170   2010    {'Affin_Transform': 1600, 'Translate': 1598, 'Gamma_Correction': 1608, 'Blur': 1646, 'Rotate': 1618}
    12    1890   2010    {'Affin_Transform': 1619, 'Translate': 1615, 'Gamma_Correction': 1633, 'Blur': 1672, 'Rotate': 1651}
    13    1920   2010    {'Affin_Transform': 1632, 'Translate': 1637, 'Gamma_Correction': 1647, 'Blur': 1695, 'Rotate': 1669}
    14    690   2010    {'Affin_Transform': 1890, 'Translate': 1892, 'Gamma_Correction': 1927, 'Blur': 1956, 'Rotate': 1935}
    15    540   2010    {'Affin_Transform': 2145, 'Translate': 2189, 'Gamma_Correction': 2211, 'Blur': 2272, 'Rotate': 2253}
    16    360   2010    {'Affin_Transform': 2473, 'Translate': 2516, 'Gamma_Correction': 2537, 'Blur': 2620, 'Rotate': 2574}
    17    990   2010    {'Affin_Transform': 2672, 'Translate': 2721, 'Gamma_Correction': 2748, 'Blur': 2814, 'Rotate': 2785}
    18    1080   2010    {'Affin_Transform': 2855, 'Translate': 2903, 'Gamma_Correction': 2943, 'Blur': 2990, 'Rotate': 2979}
    19    180   2010    {'Affin_Transform': 3227, 'Translate': 3258, 'Gamma_Correction': 3303, 'Blur': 3373, 'Rotate': 3339}
    20    300   2010    {'Affin_Transform': 3568, 'Translate': 3596, 'Gamma_Correction': 3635, 'Blur': 3734, 'Rotate': 3677}
    21    270   2010    {'Affin_Transform': 3943, 'Translate': 3936, 'Gamma_Correction': 3996, 'Blur': 4074, 'Rotate': 4001}
    22    330   2010    {'Affin_Transform': 4252, 'Translate': 4295, 'Gamma_Correction': 4327, 'Blur': 4416, 'Rotate': 4340}
    23    450   2010    {'Affin_Transform': 4566, 'Translate': 4587, 'Gamma_Correction': 4634, 'Blur': 4718, 'Rotate': 4685}
    24    240   2010    {'Affin_Transform': 4935, 'Translate': 4927, 'Gamma_Correction': 4968, 'Blur': 5098, 'Rotate': 5032}
    25    1350   2010    {'Affin_Transform': 5052, 'Translate': 5067, 'Gamma_Correction': 5103, 'Blur': 5240, 'Rotate': 5158}
    26    540   2010    {'Affin_Transform': 5361, 'Translate': 5340, 'Gamma_Correction': 5394, 'Blur': 5565, 'Rotate': 5430}
    27    210   2010    {'Affin_Transform': 5724, 'Translate': 5677, 'Gamma_Correction': 5740, 'Blur': 5940, 'Rotate': 5809}
    28    480   2010    {'Affin_Transform': 6017, 'Translate': 5967, 'Gamma_Correction': 6065, 'Blur': 6246, 'Rotate': 6125}
    29    240   2010    {'Affin_Transform': 6382, 'Translate': 6330, 'Gamma_Correction': 6397, 'Blur': 6606, 'Rotate': 6475}
    30    390   2010    {'Affin_Transform': 6719, 'Translate': 6664, 'Gamma_Correction': 6729, 'Blur': 6908, 'Rotate': 6790}
    31    690   2010    {'Affin_Transform': 6977, 'Translate': 6913, 'Gamma_Correction': 7021, 'Blur': 7181, 'Rotate': 7038}
    32    210   2010    {'Affin_Transform': 7331, 'Translate': 7277, 'Gamma_Correction': 7368, 'Blur': 7513, 'Rotate': 7441}
    33    599   2010    {'Affin_Transform': 7628, 'Translate': 7549, 'Gamma_Correction': 7646, 'Blur': 7785, 'Rotate': 7733}
    34    360   2010    {'Affin_Transform': 7930, 'Translate': 7901, 'Gamma_Correction': 7970, 'Blur': 8108, 'Rotate': 8082}
    35    1080   2010    {'Affin_Transform': 8103, 'Translate': 8105, 'Gamma_Correction': 8152, 'Blur': 8311, 'Rotate': 8250}
    36    330   2010    {'Affin_Transform': 8436, 'Translate': 8433, 'Gamma_Correction': 8507, 'Blur': 8651, 'Rotate': 8574}
    37    180   2010    {'Affin_Transform': 8801, 'Translate': 8843, 'Gamma_Correction': 8865, 'Blur': 9006, 'Rotate': 8916}
    38    1860   2010    {'Affin_Transform': 8837, 'Translate': 8861, 'Gamma_Correction': 8903, 'Blur': 9037, 'Rotate': 8943}
    39    270   2010    {'Affin_Transform': 9174, 'Translate': 9192, 'Gamma_Correction': 9268, 'Blur': 9401, 'Rotate': 9286}
    40    300   2010    {'Affin_Transform': 9543, 'Translate': 9524, 'Gamma_Correction': 9603, 'Blur': 9732, 'Rotate': 9629}
    41    210   2010    {'Affin_Transform': 9899, 'Translate': 9914, 'Gamma_Correction': 9986, 'Blur': 10061, 'Rotate': 9971}
    42    210   2010    {'Affin_Transform': 10250, 'Translate': 10285, 'Gamma_Correction': 10350, 'Blur': 10401, 'Rotate': 10345}
    


```python
#Print frequency of each applied effect during augmentation process
print(augmentation_stats)

plt.bar(augmentation_stats.keys(),augmentation_stats.values(),width=1.0,color='g',alpha=0.6,edgecolor='white')
```

    {'Affin_Transform': 10250, 'Translate': 10285, 'Gamma_Correction': 10350, 'Blur': 10401, 'Rotate': 10345}
    




    <Container object of 5 artists>




![png](output_22_2.png)



```python
#show  random Images from the data set filter for a specific class   
#Let's explore samples from each class
showSampleImage(X_train_augmented,y_train_augmented,10,filter_class=0,randomize=True,hSize=8,wSize=8,n_row=1,image_title=False)
showSampleImage(X_train_augmented,y_train_augmented,10,filter_class=3,randomize=True,hSize=8,wSize=8,n_row=1,image_title=False)
showSampleImage(X_train_augmented,y_train_augmented,10,filter_class=5,randomize=True,hSize=8,wSize=8,n_row=1,image_title=False)
showSampleImage(X_train_augmented,y_train_augmented,10,filter_class=8,randomize=True,hSize=8,wSize=8,n_row=1,image_title=False)
showSampleImage(X_train_augmented,y_train_augmented,10,filter_class=10,randomize=True,hSize=8,wSize=8,n_row=1,image_title=False)

showSampleImage(X_train_augmented,y_train_augmented,10,filter_class=15,randomize=True,hSize=8,wSize=8,n_row=1,image_title=False)
showSampleImage(X_train_augmented,y_train_augmented,10,filter_class=18,randomize=True,hSize=8,wSize=8,n_row=1,image_title=False)
showSampleImage(X_train_augmented,y_train_augmented,10,filter_class=20,randomize=True,hSize=8,wSize=8,n_row=1,image_title=False)
showSampleImage(X_train_augmented,y_train_augmented,10,filter_class=23,randomize=True,hSize=8,wSize=8,n_row=1,image_title=False)
showSampleImage(X_train_augmented,y_train_augmented,10,filter_class=25,randomize=True,hSize=8,wSize=8,n_row=1,image_title=False)

showSampleImage(X_train_augmented,y_train_augmented,10,filter_class=30,randomize=True,hSize=8,wSize=8,n_row=1,image_title=False)
showSampleImage(X_train_augmented,y_train_augmented,10,filter_class=33,randomize=True,hSize=8,wSize=8,n_row=1,image_title=False)
showSampleImage(X_train_augmented,y_train_augmented,10,filter_class=35,randomize=True,hSize=8,wSize=8,n_row=1,image_title=False)
showSampleImage(X_train_augmented,y_train_augmented,10,filter_class=38,randomize=True,hSize=8,wSize=8,n_row=1,image_title=False)
showSampleImage(X_train_augmented,y_train_augmented,10,filter_class=40,randomize=True,hSize=8,wSize=8,n_row=1,image_title=False)
showSampleImage(X_train_augmented,y_train_augmented,10,filter_class=42,randomize=True,hSize=8,wSize=8,n_row=1,image_title=False)
```


![png](output_23_0.png)



![png](output_23_1.png)



![png](output_23_2.png)



![png](output_23_3.png)



![png](output_23_4.png)



![png](output_23_5.png)



![png](output_23_6.png)



![png](output_23_7.png)



![png](output_23_8.png)



![png](output_23_9.png)



![png](output_23_10.png)



![png](output_23_11.png)



![png](output_23_12.png)



![png](output_23_13.png)



![png](output_23_14.png)



![png](output_23_15.png)



```python

# Visualize frequency of class IDs in the original data set
cls_id, count = np.unique(y_train, return_counts=True)
plt.figure(1)
plt.bar(cls_id, count, alpha=0.8)
plt.title('Traffic Sign Image Distribution Histogram before Augmentation')

# Visualize frequency of sign classes in the augment data set 
cls_id, count = np.unique(y_train_augmented, return_counts=True)
plt.figure(2)
plt.bar(cls_id, count, alpha=0.8)
plt.title('Traffic Sign Image Distribution after Augmentation')



plt.show()
```


![png](output_24_0.png)



![png](output_24_1.png)



```python

X_train = X_train_augmented
y_train = y_train_augmented

#show  random Images from the data set filter for a specific class   
#Let's explore samples from each class
showSampleImage(X_train,y_train,50,filter_class=18,randomize=True,image_title=False)
```


![png](output_25_0.png)



## Step 2: Design and Test a Model Architecture


### Pre-process the Data Set (normalization, grayscale, etc.)


```python
### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.


def conv2GrayScale(X_train):
    return np.sum(X_train/3, axis=3, keepdims=True)


# Converting to grayscale
X_train_rgb = X_train
X_train_gry = conv2GrayScale(X_train)

X_test_rgb = X_test
X_test_gry = conv2GrayScale(X_test)

X_validation_rgb = X_validation
X_validation_gry = conv2GrayScale(X_validation)

print('RGB shape:', X_train_rgb.shape)
print('Grayscale shape:', X_train_gry.shape)


X_train = X_train_gry
X_test = X_test_gry
X_validation = X_validation_gry

# TODO!! Visualize graysacle
```

    RGB shape: (86431, 32, 32, 3)
    Grayscale shape: (86431, 32, 32, 1)
    


```python
from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)

```


```python
## Normalize the train and test datasets to (-1,1)

def normalize(X_train):
    return (X_train-128)/128

X_train_normalized = normalize(X_train)
X_test_normalized = normalize(X_test)
X_validation_normalized = normalize(X_validation)

#X_train_normalized = (X_train-128)/128
#X_test_normalized = (X_test-128)/128

print(np.mean(X_train_normalized))
print(np.mean(X_test_normalized))
print(np.mean(X_validation_normalized))
```

    -0.9974813881028776
    -0.3582151534281105
    -0.3472154111278294
    


```python
print("Original shape:", X_train.shape)
print("Normalized shape:", X_train_normalized.shape)
fig, axs = plt.subplots(1,2, figsize=(10, 3))
axs = axs.ravel()

axs[0].axis('off')
axs[0].set_title('normalized')
axs[0].imshow(X_train_normalized[0].squeeze(), cmap='gray')

axs[1].axis('off')
axs[1].set_title('original')
axs[1].imshow(X_train_rgb[0])
```

    Original shape: (86431, 32, 32, 1)
    Normalized shape: (86431, 32, 32, 1)
    




    <matplotlib.image.AxesImage at 0x257b3baa550>




![png](output_31_2.png)


### Question 2.1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

First, I converted the images to grayscale. and I normalized the data set so that the data has mean zero and equal variance. this allow that each feature contirbutes approximately equally to the final distance and hence a model that is not opinionated (Well Conditioned)



### Model Architecture

### Question 2.2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

### Input
The LeNet (TsNet) architecture accepts a 32x32xC image as input, where C is the number of color channels. Since MNIST images are grayscale, C is 1 in this case.

### Architecture
**Layer 1: Convolutional.** Input = 32x32x1. Output = 28x28x6.  Padding: VALID

**Activation.** RELU Activation

**Pooling.** Pooling. Input = 28x28x6. Output = 14x14x6.



**Layer 2: Convolutional.**  Output = 10x10x16. 	Padding: VALID

**Activation.** RELU Activation

**Pooling.**  Input = 10x10x16. Output = 5x5x16.



**Flatten.**  Input = 5x5x16. Output = 400. 
Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using `tf.contrib.layers.flatten`



**Layer 3: Fully Connected.**  Input = 400. Output = 120.   	

**Activation.** Activation: RELU. 



**Layer 4: Fully Connected.** Fully Connected. Input = 120. Output = 84.

**Activation.** Activation: RELU. 


**Layer 5: Fully Connected (Logits).** Fully Connected. Input = 84. Output = 43

### Output
Return the result of the 2nd fully connected layer (Logits)
Return Conv1, Conv2 TF variables to be used with the FeatureMap visualization (Optional Step 5)

## Setup TensorFlow
The `EPOCH` and `BATCH_SIZE` values affect the training speed and model accuracy.

You do not need to modify this section.


```python



#Enhanced TsNet 
# Implemented:
#    - DropOut
# Todo:
#    - L2 Regularization
def TsNetEx(x):    
 with tf.device('/cpu:0'):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)
    
     # Dropout
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.add(tf.matmul(fc2, fc3_W), fc3_b)
    
    return logits, conv1, conv2
```

## Features and Labels
Train LeNet to classify [MNIST](http://yann.lecun.com/exdb/mnist/) data.

`x` is a placeholder for a batch of input images.
`y` is a placeholder for a batch of output labels.

You do not need to modify this section.


```python
#Clears the default graph stack and resets the global default graph.
tf.reset_default_graph() 

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))

# To Control DropOut Rate
keep_prob = tf.placeholder(tf.float32) 

#Returns a one-hot tensor.
one_hot_y = tf.one_hot(y, 43)


print('done')
```

    done
    

### Train, Validate and Test the Model

A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

## Training Pipeline
Create a training pipeline that uses the model to classify MNIST data.

You do not need to modify this section.


```python
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
```


```python
EPOCHS = 200
BATCH_SIZE = 128
rate= 0.001


logits,conv1,conv2 = TsNetEx(x)


#Distance between the SoftMaX(Y) and one_hot encoded labels.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)

#Average cross Entropy across the entire training set
loss_operation = tf.reduce_mean(cross_entropy)

#Training weights and biases
#optimizer = tf.train.AdamOptimizer(learning_rate = rate)
optimizer = tf.train.AdamOptimizer()
training_operation = optimizer.minimize(loss_operation)
```

## Model Evaluation
Evaluate how well the loss and accuracy of the model for a given dataset.

You do not need to modify this section.


```python
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        #accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

print('done')
```

    done
    

## Train the Model
Run the training data through the training pipeline to train the model.

Before each epoch, shuffle the training set.

After each epoch, measure the loss and accuracy of the validation set.

Save the model after training.

You do not need to modify this section.

### Question  2.3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

* Optimizer : AdamOptimizer
* Hyper Parameters :
   * EPOCHS = 200         OtherValuesUsed={10, 25, 50, 100, 125, 150, 200, 250, 275, 300, 500, 1000}
   * BATCH_SIZE = 128     OtherValuesUsed={100,128}
   * Learning Rate= 0.001 OtherValuesUsed={0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.005}
   * DropOut = 0.5        OtherValuesUsed={0.5, 0.6, 0.7}
   * mu = 0
   * sigma = 0.1


```python
.import time
#from tqdm import tqdm





# Measurements use for graphing loss and accuracy
batches = []
loss_batch = []
train_acc_batch = []
valid_acc_batch = []
epoch_progress = []
batch_progress=0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    start = time.time()
    print("Training...")
    print()
    for i in range(EPOCHS):
        
         
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            _, loss=sess.run([training_operation,loss_operation], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            
            #_, loss=sess.run([training_operation,loss_operation], feed_dict={x: batch_x, y: batch_y})
            
            
            loss_batch.append(loss)
            batches.append(batch_progress)
            batch_progress+=BATCH_SIZE
        
        training_accuracy = evaluate(X_train,y_train)
        train_acc_batch.append(training_accuracy)
        
        validation_accuracy = evaluate(X_validation, y_validation)
        valid_acc_batch.append(validation_accuracy)
        
        epoch_progress.append(i)
        print("EPOCH {} ".format((i+1)))
        print("...Training Accuracy = {:.3f} ...Validation Accuracy = {:.3f} ...Loss = {:.3f}".format(training_accuracy,validation_accuracy,loss))

    end = time.time()
    print("Training Time :"+str((end - start)/60)+" Minutes")
    saver.save(sess, './tsnet')
    print("Model saved")
    
  
```

    Training...
    
    EPOCH 1 
    ...Training Accuracy = 0.677 ...Validation Accuracy = 0.759 ...Loss = 1.664
    EPOCH 2 
    ...Training Accuracy = 0.773 ...Validation Accuracy = 0.808 ...Loss = 1.091
    EPOCH 3 
    ...Training Accuracy = 0.803 ...Validation Accuracy = 0.841 ...Loss = 0.744
    EPOCH 4 
    ...Training Accuracy = 0.828 ...Validation Accuracy = 0.878 ...Loss = 0.976
    EPOCH 5 
    ...Training Accuracy = 0.842 ...Validation Accuracy = 0.900 ...Loss = 0.402
    EPOCH 6 
    ...Training Accuracy = 0.852 ...Validation Accuracy = 0.895 ...Loss = 0.511
    EPOCH 7 
    ...Training Accuracy = 0.866 ...Validation Accuracy = 0.898 ...Loss = 0.598
    EPOCH 8 
    ...Training Accuracy = 0.873 ...Validation Accuracy = 0.912 ...Loss = 0.686
    EPOCH 9 
    ...Training Accuracy = 0.880 ...Validation Accuracy = 0.923 ...Loss = 0.566
    EPOCH 10 
    ...Training Accuracy = 0.883 ...Validation Accuracy = 0.900 ...Loss = 0.724
    EPOCH 11 
    ...Training Accuracy = 0.892 ...Validation Accuracy = 0.918 ...Loss = 1.104
    EPOCH 12 
    ...Training Accuracy = 0.898 ...Validation Accuracy = 0.922 ...Loss = 0.402
    EPOCH 13 
    ...Training Accuracy = 0.902 ...Validation Accuracy = 0.919 ...Loss = 0.658
    EPOCH 14 
    ...Training Accuracy = 0.904 ...Validation Accuracy = 0.916 ...Loss = 0.617
    EPOCH 15 
    ...Training Accuracy = 0.909 ...Validation Accuracy = 0.916 ...Loss = 0.254
    EPOCH 16 
    ...Training Accuracy = 0.913 ...Validation Accuracy = 0.915 ...Loss = 0.314
    EPOCH 17 
    ...Training Accuracy = 0.915 ...Validation Accuracy = 0.902 ...Loss = 0.539
    EPOCH 18 
    ...Training Accuracy = 0.922 ...Validation Accuracy = 0.921 ...Loss = 0.603
    EPOCH 19 
    ...Training Accuracy = 0.922 ...Validation Accuracy = 0.925 ...Loss = 0.422
    EPOCH 20 
    ...Training Accuracy = 0.924 ...Validation Accuracy = 0.923 ...Loss = 0.253
    EPOCH 21 
    ...Training Accuracy = 0.930 ...Validation Accuracy = 0.928 ...Loss = 0.381
    EPOCH 22 
    ...Training Accuracy = 0.933 ...Validation Accuracy = 0.927 ...Loss = 0.459
    EPOCH 23 
    ...Training Accuracy = 0.931 ...Validation Accuracy = 0.906 ...Loss = 0.583
    EPOCH 24 
    ...Training Accuracy = 0.935 ...Validation Accuracy = 0.917 ...Loss = 0.159
    EPOCH 25 
    ...Training Accuracy = 0.937 ...Validation Accuracy = 0.933 ...Loss = 0.458
    EPOCH 26 
    ...Training Accuracy = 0.940 ...Validation Accuracy = 0.924 ...Loss = 0.503
    EPOCH 27 
    ...Training Accuracy = 0.938 ...Validation Accuracy = 0.917 ...Loss = 0.288
    EPOCH 28 
    ...Training Accuracy = 0.944 ...Validation Accuracy = 0.925 ...Loss = 0.221
    EPOCH 29 
    ...Training Accuracy = 0.945 ...Validation Accuracy = 0.918 ...Loss = 0.244
    EPOCH 30 
    ...Training Accuracy = 0.949 ...Validation Accuracy = 0.931 ...Loss = 0.339
    EPOCH 31 
    ...Training Accuracy = 0.949 ...Validation Accuracy = 0.924 ...Loss = 0.337
    EPOCH 32 
    ...Training Accuracy = 0.950 ...Validation Accuracy = 0.912 ...Loss = 0.045
    EPOCH 33 
    ...Training Accuracy = 0.954 ...Validation Accuracy = 0.923 ...Loss = 0.323
    EPOCH 34 
    ...Training Accuracy = 0.955 ...Validation Accuracy = 0.925 ...Loss = 0.179
    EPOCH 35 
    ...Training Accuracy = 0.956 ...Validation Accuracy = 0.914 ...Loss = 0.076
    EPOCH 36 
    ...Training Accuracy = 0.956 ...Validation Accuracy = 0.929 ...Loss = 0.447
    EPOCH 37 
    ...Training Accuracy = 0.957 ...Validation Accuracy = 0.924 ...Loss = 0.180
    EPOCH 38 
    ...Training Accuracy = 0.960 ...Validation Accuracy = 0.919 ...Loss = 0.128
    EPOCH 39 
    ...Training Accuracy = 0.957 ...Validation Accuracy = 0.925 ...Loss = 0.251
    EPOCH 40 
    ...Training Accuracy = 0.960 ...Validation Accuracy = 0.931 ...Loss = 0.243
    EPOCH 41 
    ...Training Accuracy = 0.961 ...Validation Accuracy = 0.922 ...Loss = 0.342
    EPOCH 42 
    ...Training Accuracy = 0.960 ...Validation Accuracy = 0.920 ...Loss = 0.433
    EPOCH 43 
    ...Training Accuracy = 0.961 ...Validation Accuracy = 0.931 ...Loss = 0.463
    EPOCH 44 
    ...Training Accuracy = 0.961 ...Validation Accuracy = 0.926 ...Loss = 0.270
    EPOCH 45 
    ...Training Accuracy = 0.959 ...Validation Accuracy = 0.918 ...Loss = 0.170
    EPOCH 46 
    ...Training Accuracy = 0.967 ...Validation Accuracy = 0.927 ...Loss = 0.278
    EPOCH 47 
    ...Training Accuracy = 0.965 ...Validation Accuracy = 0.916 ...Loss = 0.336
    EPOCH 48 
    ...Training Accuracy = 0.966 ...Validation Accuracy = 0.916 ...Loss = 0.147
    EPOCH 49 
    ...Training Accuracy = 0.969 ...Validation Accuracy = 0.932 ...Loss = 0.102
    EPOCH 50 
    ...Training Accuracy = 0.970 ...Validation Accuracy = 0.930 ...Loss = 0.032
    EPOCH 51 
    ...Training Accuracy = 0.970 ...Validation Accuracy = 0.919 ...Loss = 0.230
    EPOCH 52 
    ...Training Accuracy = 0.971 ...Validation Accuracy = 0.929 ...Loss = 0.240
    EPOCH 53 
    ...Training Accuracy = 0.966 ...Validation Accuracy = 0.927 ...Loss = 0.099
    EPOCH 54 
    ...Training Accuracy = 0.970 ...Validation Accuracy = 0.932 ...Loss = 0.042
    EPOCH 55 
    ...Training Accuracy = 0.970 ...Validation Accuracy = 0.928 ...Loss = 0.204
    EPOCH 56 
    ...Training Accuracy = 0.972 ...Validation Accuracy = 0.927 ...Loss = 0.024
    EPOCH 57 
    ...Training Accuracy = 0.969 ...Validation Accuracy = 0.916 ...Loss = 0.110
    EPOCH 58 
    ...Training Accuracy = 0.969 ...Validation Accuracy = 0.932 ...Loss = 0.239
    EPOCH 59 
    ...Training Accuracy = 0.973 ...Validation Accuracy = 0.926 ...Loss = 0.229
    EPOCH 60 
    ...Training Accuracy = 0.974 ...Validation Accuracy = 0.932 ...Loss = 0.170
    EPOCH 61 
    ...Training Accuracy = 0.973 ...Validation Accuracy = 0.929 ...Loss = 0.231
    EPOCH 62 
    ...Training Accuracy = 0.972 ...Validation Accuracy = 0.935 ...Loss = 0.190
    EPOCH 63 
    ...Training Accuracy = 0.975 ...Validation Accuracy = 0.925 ...Loss = 0.169
    EPOCH 64 
    ...Training Accuracy = 0.976 ...Validation Accuracy = 0.925 ...Loss = 0.042
    EPOCH 65 
    ...Training Accuracy = 0.974 ...Validation Accuracy = 0.933 ...Loss = 0.012
    EPOCH 66 
    ...Training Accuracy = 0.977 ...Validation Accuracy = 0.930 ...Loss = 0.143
    EPOCH 67 
    ...Training Accuracy = 0.977 ...Validation Accuracy = 0.936 ...Loss = 0.110
    EPOCH 68 
    ...Training Accuracy = 0.979 ...Validation Accuracy = 0.938 ...Loss = 0.133
    EPOCH 69 
    ...Training Accuracy = 0.978 ...Validation Accuracy = 0.938 ...Loss = 0.129
    EPOCH 70 
    ...Training Accuracy = 0.980 ...Validation Accuracy = 0.927 ...Loss = 0.022
    EPOCH 71 
    ...Training Accuracy = 0.977 ...Validation Accuracy = 0.934 ...Loss = 0.108
    EPOCH 72 
    ...Training Accuracy = 0.980 ...Validation Accuracy = 0.929 ...Loss = 0.161
    EPOCH 73 
    ...Training Accuracy = 0.979 ...Validation Accuracy = 0.929 ...Loss = 0.293
    EPOCH 74 
    ...Training Accuracy = 0.979 ...Validation Accuracy = 0.926 ...Loss = 0.249
    EPOCH 75 
    ...Training Accuracy = 0.980 ...Validation Accuracy = 0.929 ...Loss = 0.088
    EPOCH 76 
    ...Training Accuracy = 0.981 ...Validation Accuracy = 0.941 ...Loss = 0.063
    EPOCH 77 
    ...Training Accuracy = 0.983 ...Validation Accuracy = 0.931 ...Loss = 0.303
    EPOCH 78 
    ...Training Accuracy = 0.982 ...Validation Accuracy = 0.932 ...Loss = 0.407
    EPOCH 79 
    ...Training Accuracy = 0.982 ...Validation Accuracy = 0.927 ...Loss = 0.140
    EPOCH 80 
    ...Training Accuracy = 0.978 ...Validation Accuracy = 0.926 ...Loss = 0.149
    EPOCH 81 
    ...Training Accuracy = 0.982 ...Validation Accuracy = 0.932 ...Loss = 0.216
    EPOCH 82 
    ...Training Accuracy = 0.982 ...Validation Accuracy = 0.931 ...Loss = 0.064
    EPOCH 83 
    ...Training Accuracy = 0.981 ...Validation Accuracy = 0.924 ...Loss = 0.282
    EPOCH 84 
    ...Training Accuracy = 0.983 ...Validation Accuracy = 0.934 ...Loss = 0.040
    EPOCH 85 
    ...Training Accuracy = 0.980 ...Validation Accuracy = 0.931 ...Loss = 0.034
    EPOCH 86 
    ...Training Accuracy = 0.983 ...Validation Accuracy = 0.931 ...Loss = 0.024
    EPOCH 87 
    ...Training Accuracy = 0.981 ...Validation Accuracy = 0.932 ...Loss = 0.299
    EPOCH 88 
    ...Training Accuracy = 0.984 ...Validation Accuracy = 0.930 ...Loss = 0.009
    EPOCH 89 
    ...Training Accuracy = 0.983 ...Validation Accuracy = 0.933 ...Loss = 0.067
    EPOCH 90 
    ...Training Accuracy = 0.983 ...Validation Accuracy = 0.929 ...Loss = 0.243
    EPOCH 91 
    ...Training Accuracy = 0.983 ...Validation Accuracy = 0.933 ...Loss = 0.191
    EPOCH 92 
    ...Training Accuracy = 0.983 ...Validation Accuracy = 0.934 ...Loss = 0.064
    EPOCH 93 
    ...Training Accuracy = 0.984 ...Validation Accuracy = 0.929 ...Loss = 0.055
    EPOCH 94 
    ...Training Accuracy = 0.984 ...Validation Accuracy = 0.937 ...Loss = 0.037
    EPOCH 95 
    ...Training Accuracy = 0.983 ...Validation Accuracy = 0.934 ...Loss = 0.030
    EPOCH 96 
    ...Training Accuracy = 0.987 ...Validation Accuracy = 0.942 ...Loss = 0.179
    EPOCH 97 
    ...Training Accuracy = 0.985 ...Validation Accuracy = 0.932 ...Loss = 0.116
    EPOCH 98 
    ...Training Accuracy = 0.984 ...Validation Accuracy = 0.937 ...Loss = 0.153
    EPOCH 99 
    ...Training Accuracy = 0.985 ...Validation Accuracy = 0.942 ...Loss = 0.031
    EPOCH 100 
    ...Training Accuracy = 0.986 ...Validation Accuracy = 0.943 ...Loss = 0.052
    EPOCH 101 
    ...Training Accuracy = 0.985 ...Validation Accuracy = 0.933 ...Loss = 0.290
    EPOCH 102 
    ...Training Accuracy = 0.987 ...Validation Accuracy = 0.939 ...Loss = 0.303
    EPOCH 103 
    ...Training Accuracy = 0.986 ...Validation Accuracy = 0.945 ...Loss = 0.095
    EPOCH 104 
    ...Training Accuracy = 0.987 ...Validation Accuracy = 0.938 ...Loss = 0.058
    EPOCH 105 
    ...Training Accuracy = 0.986 ...Validation Accuracy = 0.935 ...Loss = 0.093
    EPOCH 106 
    ...Training Accuracy = 0.988 ...Validation Accuracy = 0.941 ...Loss = 0.089
    EPOCH 107 
    ...Training Accuracy = 0.986 ...Validation Accuracy = 0.939 ...Loss = 0.022
    EPOCH 108 
    ...Training Accuracy = 0.984 ...Validation Accuracy = 0.936 ...Loss = 0.024
    EPOCH 109 
    ...Training Accuracy = 0.986 ...Validation Accuracy = 0.942 ...Loss = 0.186
    EPOCH 110 
    ...Training Accuracy = 0.987 ...Validation Accuracy = 0.939 ...Loss = 0.340
    EPOCH 111 
    ...Training Accuracy = 0.987 ...Validation Accuracy = 0.940 ...Loss = 0.015
    EPOCH 112 
    ...Training Accuracy = 0.985 ...Validation Accuracy = 0.937 ...Loss = 0.475
    EPOCH 113 
    ...Training Accuracy = 0.986 ...Validation Accuracy = 0.934 ...Loss = 0.069
    EPOCH 114 
    ...Training Accuracy = 0.985 ...Validation Accuracy = 0.932 ...Loss = 0.046
    EPOCH 115 
    ...Training Accuracy = 0.989 ...Validation Accuracy = 0.941 ...Loss = 0.045
    EPOCH 116 
    ...Training Accuracy = 0.989 ...Validation Accuracy = 0.940 ...Loss = 0.150
    EPOCH 117 
    ...Training Accuracy = 0.989 ...Validation Accuracy = 0.942 ...Loss = 0.134
    EPOCH 118 
    ...Training Accuracy = 0.988 ...Validation Accuracy = 0.934 ...Loss = 0.103
    EPOCH 119 
    ...Training Accuracy = 0.986 ...Validation Accuracy = 0.943 ...Loss = 0.399
    EPOCH 120 
    ...Training Accuracy = 0.987 ...Validation Accuracy = 0.933 ...Loss = 0.097
    EPOCH 121 
    ...Training Accuracy = 0.988 ...Validation Accuracy = 0.939 ...Loss = 0.019
    EPOCH 122 
    ...Training Accuracy = 0.988 ...Validation Accuracy = 0.934 ...Loss = 0.063
    EPOCH 123 
    ...Training Accuracy = 0.989 ...Validation Accuracy = 0.938 ...Loss = 0.275
    EPOCH 124 
    ...Training Accuracy = 0.987 ...Validation Accuracy = 0.942 ...Loss = 0.242
    EPOCH 125 
    ...Training Accuracy = 0.988 ...Validation Accuracy = 0.941 ...Loss = 0.016
    EPOCH 126 
    ...Training Accuracy = 0.985 ...Validation Accuracy = 0.932 ...Loss = 0.011
    EPOCH 127 
    ...Training Accuracy = 0.990 ...Validation Accuracy = 0.930 ...Loss = 0.064
    EPOCH 128 
    ...Training Accuracy = 0.989 ...Validation Accuracy = 0.938 ...Loss = 0.089
    EPOCH 129 
    ...Training Accuracy = 0.989 ...Validation Accuracy = 0.935 ...Loss = 0.138
    EPOCH 130 
    ...Training Accuracy = 0.989 ...Validation Accuracy = 0.936 ...Loss = 0.037
    EPOCH 131 
    ...Training Accuracy = 0.988 ...Validation Accuracy = 0.928 ...Loss = 0.047
    EPOCH 132 
    ...Training Accuracy = 0.991 ...Validation Accuracy = 0.940 ...Loss = 0.059
    EPOCH 133 
    ...Training Accuracy = 0.991 ...Validation Accuracy = 0.930 ...Loss = 0.088
    EPOCH 134 
    ...Training Accuracy = 0.989 ...Validation Accuracy = 0.934 ...Loss = 0.285
    EPOCH 135 
    ...Training Accuracy = 0.991 ...Validation Accuracy = 0.941 ...Loss = 0.028
    EPOCH 136 
    ...Training Accuracy = 0.992 ...Validation Accuracy = 0.939 ...Loss = 0.140
    EPOCH 137 
    ...Training Accuracy = 0.990 ...Validation Accuracy = 0.939 ...Loss = 0.114
    EPOCH 138 
    ...Training Accuracy = 0.990 ...Validation Accuracy = 0.940 ...Loss = 0.075
    EPOCH 139 
    ...Training Accuracy = 0.989 ...Validation Accuracy = 0.934 ...Loss = 0.425
    EPOCH 140 
    ...Training Accuracy = 0.991 ...Validation Accuracy = 0.934 ...Loss = 0.070
    EPOCH 141 
    ...Training Accuracy = 0.992 ...Validation Accuracy = 0.935 ...Loss = 0.098
    EPOCH 142 
    ...Training Accuracy = 0.992 ...Validation Accuracy = 0.945 ...Loss = 0.045
    EPOCH 143 
    ...Training Accuracy = 0.991 ...Validation Accuracy = 0.941 ...Loss = 0.021
    EPOCH 144 
    ...Training Accuracy = 0.990 ...Validation Accuracy = 0.948 ...Loss = 0.290
    EPOCH 145 
    ...Training Accuracy = 0.991 ...Validation Accuracy = 0.947 ...Loss = 0.061
    EPOCH 146 
    ...Training Accuracy = 0.991 ...Validation Accuracy = 0.943 ...Loss = 0.002
    EPOCH 147 
    ...Training Accuracy = 0.992 ...Validation Accuracy = 0.942 ...Loss = 0.153
    EPOCH 148 
    ...Training Accuracy = 0.990 ...Validation Accuracy = 0.939 ...Loss = 0.141
    EPOCH 149 
    ...Training Accuracy = 0.992 ...Validation Accuracy = 0.933 ...Loss = 0.068
    EPOCH 150 
    ...Training Accuracy = 0.993 ...Validation Accuracy = 0.944 ...Loss = 0.405
    EPOCH 151 
    ...Training Accuracy = 0.992 ...Validation Accuracy = 0.929 ...Loss = 0.098
    EPOCH 152 
    ...Training Accuracy = 0.992 ...Validation Accuracy = 0.946 ...Loss = 0.045
    EPOCH 153 
    ...Training Accuracy = 0.991 ...Validation Accuracy = 0.939 ...Loss = 0.034
    EPOCH 154 
    ...Training Accuracy = 0.990 ...Validation Accuracy = 0.935 ...Loss = 0.065
    EPOCH 155 
    ...Training Accuracy = 0.991 ...Validation Accuracy = 0.941 ...Loss = 0.112
    EPOCH 156 
    ...Training Accuracy = 0.991 ...Validation Accuracy = 0.940 ...Loss = 0.074
    EPOCH 157 
    ...Training Accuracy = 0.990 ...Validation Accuracy = 0.938 ...Loss = 0.199
    EPOCH 158 
    ...Training Accuracy = 0.991 ...Validation Accuracy = 0.936 ...Loss = 0.183
    EPOCH 159 
    ...Training Accuracy = 0.991 ...Validation Accuracy = 0.949 ...Loss = 0.068
    EPOCH 160 
    ...Training Accuracy = 0.992 ...Validation Accuracy = 0.938 ...Loss = 0.035
    EPOCH 161 
    ...Training Accuracy = 0.993 ...Validation Accuracy = 0.943 ...Loss = 0.008
    EPOCH 162 
    ...Training Accuracy = 0.992 ...Validation Accuracy = 0.938 ...Loss = 0.016
    EPOCH 163 
    ...Training Accuracy = 0.991 ...Validation Accuracy = 0.938 ...Loss = 0.107
    EPOCH 164 
    ...Training Accuracy = 0.991 ...Validation Accuracy = 0.933 ...Loss = 0.083
    EPOCH 165 
    ...Training Accuracy = 0.992 ...Validation Accuracy = 0.937 ...Loss = 0.091
    EPOCH 166 
    ...Training Accuracy = 0.991 ...Validation Accuracy = 0.946 ...Loss = 0.084
    EPOCH 167 
    ...Training Accuracy = 0.993 ...Validation Accuracy = 0.938 ...Loss = 0.016
    EPOCH 168 
    ...Training Accuracy = 0.992 ...Validation Accuracy = 0.944 ...Loss = 0.034
    EPOCH 169 
    ...Training Accuracy = 0.992 ...Validation Accuracy = 0.938 ...Loss = 0.073
    EPOCH 170 
    ...Training Accuracy = 0.993 ...Validation Accuracy = 0.943 ...Loss = 0.153
    EPOCH 171 
    ...Training Accuracy = 0.992 ...Validation Accuracy = 0.938 ...Loss = 0.174
    EPOCH 172 
    ...Training Accuracy = 0.989 ...Validation Accuracy = 0.939 ...Loss = 0.178
    EPOCH 173 
    ...Training Accuracy = 0.991 ...Validation Accuracy = 0.937 ...Loss = 0.292
    EPOCH 174 
    ...Training Accuracy = 0.992 ...Validation Accuracy = 0.941 ...Loss = 0.007
    EPOCH 175 
    ...Training Accuracy = 0.992 ...Validation Accuracy = 0.945 ...Loss = 0.021
    EPOCH 176 
    ...Training Accuracy = 0.993 ...Validation Accuracy = 0.942 ...Loss = 0.356
    EPOCH 177 
    ...Training Accuracy = 0.993 ...Validation Accuracy = 0.934 ...Loss = 0.075
    EPOCH 178 
    ...Training Accuracy = 0.994 ...Validation Accuracy = 0.940 ...Loss = 0.035
    EPOCH 179 
    ...Training Accuracy = 0.993 ...Validation Accuracy = 0.941 ...Loss = 0.001
    EPOCH 180 
    ...Training Accuracy = 0.993 ...Validation Accuracy = 0.949 ...Loss = 0.001
    EPOCH 181 
    ...Training Accuracy = 0.993 ...Validation Accuracy = 0.939 ...Loss = 0.180
    EPOCH 182 
    ...Training Accuracy = 0.993 ...Validation Accuracy = 0.939 ...Loss = 0.079
    EPOCH 183 
    ...Training Accuracy = 0.992 ...Validation Accuracy = 0.938 ...Loss = 0.209
    EPOCH 184 
    ...Training Accuracy = 0.991 ...Validation Accuracy = 0.938 ...Loss = 0.212
    EPOCH 185 
    ...Training Accuracy = 0.993 ...Validation Accuracy = 0.942 ...Loss = 0.001
    EPOCH 186 
    ...Training Accuracy = 0.993 ...Validation Accuracy = 0.947 ...Loss = 0.745
    EPOCH 187 
    ...Training Accuracy = 0.994 ...Validation Accuracy = 0.942 ...Loss = 0.000
    EPOCH 188 
    ...Training Accuracy = 0.993 ...Validation Accuracy = 0.944 ...Loss = 0.002
    EPOCH 189 
    ...Training Accuracy = 0.992 ...Validation Accuracy = 0.936 ...Loss = 0.044
    EPOCH 190 
    ...Training Accuracy = 0.992 ...Validation Accuracy = 0.944 ...Loss = 0.056
    EPOCH 191 
    ...Training Accuracy = 0.993 ...Validation Accuracy = 0.945 ...Loss = 0.034
    EPOCH 192 
    ...Training Accuracy = 0.994 ...Validation Accuracy = 0.943 ...Loss = 0.043
    EPOCH 193 
    ...Training Accuracy = 0.990 ...Validation Accuracy = 0.945 ...Loss = 0.030
    EPOCH 194 
    ...Training Accuracy = 0.993 ...Validation Accuracy = 0.937 ...Loss = 0.069
    EPOCH 195 
    ...Training Accuracy = 0.992 ...Validation Accuracy = 0.944 ...Loss = 0.193
    EPOCH 196 
    ...Training Accuracy = 0.993 ...Validation Accuracy = 0.937 ...Loss = 0.087
    EPOCH 197 
    ...Training Accuracy = 0.993 ...Validation Accuracy = 0.947 ...Loss = 0.006
    EPOCH 198 
    ...Training Accuracy = 0.995 ...Validation Accuracy = 0.943 ...Loss = 0.169
    EPOCH 199 
    ...Training Accuracy = 0.994 ...Validation Accuracy = 0.947 ...Loss = 0.041
    EPOCH 200 
    ...Training Accuracy = 0.993 ...Validation Accuracy = 0.951 ...Loss = 0.018
    Training Time :118.11126843690872 Minutes
    Model saved
    

## Training Log



```python
    #Visualizing loss across all data set during all EPOCHS
    loss_plot = plt.subplot(211)
    loss_plot.set_title('Loss')
    loss_plot.plot(batches,loss_batch,'g')
    loss_plot.set_ylim([0.0001,3])
    
    #Comparing accuracy for validation and training data during each EPOCH
    acc_plot = plt.subplot(212)
    acc_plot.set_title('Accuracy')
    acc_plot.plot(epoch_progress, train_acc_batch, 'r', label='Training Accuracy')
    acc_plot.plot(epoch_progress, valid_acc_batch, 'x', label='Validation Accuracy')
    acc_plot.set_ylim([0, 2.0])
    acc_plot.set_xlim([epoch_progress[0], epoch_progress[-1]])
    acc_plot.legend(loc=4)
    plt.tight_layout()
    plt.show()
   
```


![png](output_50_0.png)


### Question 2.4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

 - 1st Approach :
          * Processing : CPU0 I7-4930k @ 3.40GHZ (My GPU only support CUDA 2 while Tensor flow require CUDA 3 at minimum)
          * Model : Used Default LeNet Model with minor changes to fit new Input and ouput Labels  
          * Training DataSet: Used the Default data without augmentation
          * Sample Results :
                 EPOCH 100, BATCH_SIZE=128, rate=0.001, Kp=0.0 Accuracy on Extra Test Image : 0.083333
                 EPOCH 50, BATCH_SIZE=128, rate=0.0009, Kp=0.0 Accuracy on Extra Test Image : 0.0000
                 EPOCH 200, BATCH_SIZE=128, rate=0.0009, Kp=0.0 Accuracy on Extra Test Image : 0.0000 
                 EPOCH 50 BATCH_SIZE=128 rate=0.001   Validation Accuracy on Extra Test Image : 0.252
                 EPOCH 100 BATCH_SIZE=128 rate=0.0009 Validation Accuracy on Extra Test Image : 0.222
 - 2nd Approach :
         * Processing :  Amazon EC2 GPU Instance G2
         * Model : Introduced DropOut
                   Increased Output width at the Convolutional layers
         * Training DataSet: still no Augmenation
         * Best Result:
                 EPOCH 200, BATCH_SIZE=128, rate=0.0009, KP=0.5  Validation Accuracy = 0.8, Accuracy on Extra Test Images = 0.5
 - 3rd Approach :
         * Processing :  Amazon EC2 GPU Instance G2
         * Model : No change
         * Training DataSet: Image Augmentation (brightness, Rotation, PIL Effects (Invert, Posterize..etc)
         * Best Result:
                 EPOCH 200, BATCH_SIZE=128, rate=0.001, KP=0.5  Validation Accuracy = 0.94, Accuracy on Extra Test Images = 0.86 
                 
                 
Notes:
    * Learning rate and EPOCH# had the biggest impact on the model accuracy.
    * Setting large EPOCH number tends to yield a highly overfitted model that had very low accuracy against extra test images.
    * Data Set augmenation helped me with increasing model accuracy from 50% to almost 90%
    
                 

## Evaluate the Model


```python
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    
    print("\n  Training set accuracy = {:.3f}".format(evaluate(X_train, y_train)))
    print("Validation set accuracy = {:.3f}".format(evaluate(X_validation, y_validation)))
    print("      Test set accuracy = {:.3f}".format(evaluate(X_test, y_test)))
```

    INFO:tensorflow:Restoring parameters from .\tsnet
    
      Training set accuracy = 0.993
    Validation set accuracy = 0.951
          Test set accuracy = 0.916
    

---

## Step 3: Test a Model on New Images

To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

### Load and Output the Images


```python

def load_extra_images(path):
    file_list = os.listdir(path)
    #print(file_list)
    images=[]
    labels=[]
    for file in file_list:
        img=cv2.imread(path+'/'+file)
        #img=mpimg.imread(path+'/'+file)
        #img=tf.image.resize_images(img, [32, 32])
        img=imresize(img, [32,32])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        
        #images.append(img)
        labels.append(int(file.split('_')[1].split('.')[0]))
        #imgplot = plt.imshow(img)
        #plt.show()
        #print(int(file.split('_')[1].split('.')[0]))
    return images,labels
        

X_Extra=[]
y_extra=[]
X_Extra,y_extra=load_extra_images('./test_images')


showSampleImage(X_Extra,
                y_extra, 
                len(X_Extra),
                randomize=False,
                filter_class=None)
```

    c:\users\farid\miniconda3\envs\tensorflowgpu\lib\site-packages\ipykernel_launcher.py:16: DeprecationWarning: `imresize` is deprecated!
    `imresize` is deprecated in SciPy 1.0.0, and will be removed in 1.2.0.
    Use ``skimage.transform.resize`` instead.
      app.launch_new_instance()
    


![png](output_56_1.png)



```python
#Normalize Extra testing set

X_Extra=np.asarray(X_Extra)
X_Extra_grey = conv2GrayScale(X_Extra)
X_Extra_normalized = normalize(X_Extra_grey)
print(X_Extra.shape)
print(X_Extra_normalized.shape)

plt.imshow(X_Extra[1],cmap='gray')
print(y_extra[1])
```

    (15, 32, 32, 3)
    (15, 32, 32, 1)
    1
    


![png](output_57_1.png)


### Predict the Sign Type for Each Image


```python
### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    result=sess.run(logits, feed_dict={x: X_Extra_normalized, y: y_extra, keep_prob: 1.0})
    predicted_label = np.argmax(result, axis = 1)
    #print(predicted_label)
    
showSampleImage(X_Extra,predicted_label, len(X_Extra),randomize=False,filter_class=None,actual_title_list=y_extra)


#legend
# Green Border ... Correct Predictions :)
# Red Border ... Incorrect Predictions :(
```

    INFO:tensorflow:Restoring parameters from .\tsnet
    


![png](output_59_1.png)


### Analyze Performance


```python
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    #print("\n  Training set accuracy = {:.3f}".format(evaluate(X_train[1:2], y_train[1:2])))
    print("Extra Test set accuracy = {:.3f} %".format(evaluate(X_Extra_normalized, y_extra)*100))
    
```

    INFO:tensorflow:Restoring parameters from .\tsnet
    Extra Test set accuracy = 86.667 %
    

### Question 3.1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

1. Yield: The model was always successful with predicting this Sign.
2. Speed Limit (30km/h) : Speed limit signs were always difficult to predict since they are very similar (in this case the model confused 30 km/h with 80 km/h)
3. Road Work: None standard Road work sign was always hard to detect. DataSet augmentation helped with increasing the accuracy of predicting this sign.
4. Speed Limit (70km/h) : similar to #2 Speed limit signs were always difficult to predict since they are very similar
5. Road Work : DataSet augmentation helped with increasing the accuracy of predicting this sign.

#### Question 3.2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model prediction results was satisfactory on these new extra traffic signs (86.66%) and not far from the model accuracy on the test images (91% accuracy).  

### Output Top 5 Softmax Probabilities For Each Image Found on the Web

For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 

The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.

`tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.

Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:

```
# (5, 6) array
a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
         0.12789202],
       [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
         0.15899337],
       [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
         0.23892179],
       [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
         0.16505091],
       [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
         0.09155967]])
```

Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:

```
TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
       [ 0.28086119,  0.27569815,  0.18063401],
       [ 0.26076848,  0.23892179,  0.23664738],
       [ 0.29198961,  0.26234032,  0.16505091],
       [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
       [0, 1, 4],
       [0, 5, 1],
       [1, 3, 5],
       [1, 4, 3]], dtype=int32))
```

Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.


```python

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.



logits_k=tf.placeholder(tf.float32)
softmax = tf.nn.softmax(logits_k)
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    #Finds values and indices of the k largest entries for the last dimension.
    values, indices = sess.run(tf.nn.top_k(result, k=5))
    print('-------------------------------------------------------------------------------------------')
    print(' label indices (classes) for all 15 images')
    print(indices)
    print('-------------------------------------------------------------------------------------------')
    print(' Predicted Probabilities for all 15 images')
    print(values)
    
    softmax_prob = sess.run(softmax, feed_dict={logits_k: values})
    print('-------------------------------------------------------------------------------------------')
    print(' Softmax of Predicted Probabilities for all 15 images')
    print(softmax_prob)
     
for i in range(len(y_extra)):
    print('-------------------------------------------------------------------------------------------')
    print('**** Image {} :  '.format(i))
    #plt.imshow(imresize(X_Extra[i], [16,16]))
    #plt.show()
    print('                 Actual_Class    == ({})'.format(getSignNameById(str(y_extra[i]))))
    print('                 Predicted_Class == ({})'.format(getSignNameById(str(predicted_label[i]))))
    if y_extra[i]==predicted_label[i]:
            print('                       --> The Predicted label match actual class (Match) ')
    else:
            print('                       --> The Predicted label did NOT match actual class (No Match) ')
    
    print('                 Top 5 Prediction Probabilities : ')
    
    y_sorted = [indices[i] for _,x in sorted(zip(softmax_prob[i],indices[i]))]
    softmax_sorted=np.sort(softmax_prob[i])[::-1]
    for index in range(len(y_sorted[0])):
        print('                            {}. : {} , probability: {} %'.format(index,getSignNameById(str(y_sorted[0][index])),softmax_sorted[index]*100))
        
    #print(y_sorted[0])
```

    INFO:tensorflow:Restoring parameters from .\tsnet
    -------------------------------------------------------------------------------------------
     label indices (classes) for all 15 images
    [[13 12 38 34 29]
     [ 6  5  7  1  8]
     [25 30 20 23 22]
     [ 4  8 26 31  7]
     [14 17  8 34 38]
     [25 38 30 20 22]
     [12 41 15 18 42]
     [22 38 14 15 13]
     [40  1 39  7  0]
     [17 10 34 40 14]
     [22 29 38 26 28]
     [11 30 21 40 25]
     [ 1  4  0  2  8]
     [25 38 30 20 22]
     [14 38 34 15  4]]
    -------------------------------------------------------------------------------------------
     Predicted Probabilities for all 15 images
    [[  33.681328    20.78622     -1.4781467  -35.147373   -35.43019  ]
     [  13.128907   -11.306796   -11.487299   -12.219127   -42.24548  ]
     [ 131.42723     12.052686   -53.929985  -118.23535   -150.98221  ]
     [  34.347424   -35.989204   -49.570522   -56.95614    -60.018864 ]
     [  30.755413     1.4582164   -6.2949686   -8.302131   -20.008337 ]
     [  72.30081     -8.364271   -23.171984   -36.697723   -60.081665 ]
     [  14.931318   -33.3814     -36.077507   -41.974113   -43.969337 ]
     [  10.635921     4.8857756    3.4924083   -1.491667    -4.561928 ]
     [  14.358021     3.9154408    1.7060323   -3.2812707   -6.0160785]
     [  77.43073      0.7437843  -49.01974    -56.496643   -62.08064  ]
     [  84.23002     39.563953   -32.833813   -41.665638   -41.874207 ]
     [ 200.01268    115.358574    46.932117   -63.342003  -133.85388  ]
     [ 101.05747    -43.015163   -46.362335   -60.822945   -62.865356 ]
     [  72.30081     -8.364271   -23.171984   -36.697723   -60.081665 ]
     [  21.47879     -9.026398   -16.024107   -20.05142    -23.461004 ]]
    -------------------------------------------------------------------------------------------
     Softmax of Predicted Probabilities for all 15 images
    [[9.9999750e-01 2.5102963e-06 5.3756830e-16 1.2825438e-30 9.6660186e-31]
     [1.0000000e+00 2.4417933e-11 2.0385299e-11 9.8059286e-12 8.9373710e-25]
     [1.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00]
     [1.0000000e+00 2.8391757e-31 3.5883861e-37 0.0000000e+00 0.0000000e+00]
     [1.0000000e+00 1.8896825e-13 8.1137961e-17 1.0902444e-17 8.9863702e-23]
     [1.0000000e+00 9.2810854e-36 0.0000000e+00 0.0000000e+00 0.0000000e+00]
     [1.0000000e+00 1.0424416e-21 7.0331165e-23 1.9332328e-25 2.6288719e-26]
     [9.9603784e-01 3.1697105e-03 7.8684260e-04 5.3868093e-06 2.4999665e-07]
     [9.9996758e-01 2.9162917e-05 3.2010828e-06 2.1844334e-08 1.4178387e-09]
     [1.0000000e+00 4.9577025e-34 0.0000000e+00 0.0000000e+00 0.0000000e+00]
     [1.0000000e+00 3.9973700e-20 0.0000000e+00 0.0000000e+00 0.0000000e+00]
     [1.0000000e+00 1.7186555e-37 0.0000000e+00 0.0000000e+00 0.0000000e+00]
     [1.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00]
     [1.0000000e+00 9.2810854e-36 0.0000000e+00 0.0000000e+00 0.0000000e+00]
     [1.0000000e+00 5.6463162e-14 5.1605721e-17 9.1972486e-19 3.0401456e-20]]
    -------------------------------------------------------------------------------------------
    **** Image 0 :  
                     Actual_Class    == (Yield)
                     Predicted_Class == (Yield)
                           --> The Predicted label match actual class (Match) 
                     Top 5 Prediction Probabilities : 
                                0. : Yield , probability: 99.99974966049194 %
                                1. : Priority road , probability: 0.0002510296326363459 %
                                2. : Keep right , probability: 5.3756829530463924e-14 %
                                3. : Turn left ahead , probability: 1.2825438001652313e-28 %
                                4. : Bicycles crossing , probability: 9.66601857675514e-29 %
    -------------------------------------------------------------------------------------------
    **** Image 1 :  
                     Actual_Class    == (Speed limit (30km/h))
                     Predicted_Class == (End of speed limit (80km/h))
                           --> The Predicted label did NOT match actual class (No Match) 
                     Top 5 Prediction Probabilities : 
                                0. : End of speed limit (80km/h) , probability: 100.0 %
                                1. : Speed limit (80km/h) , probability: 2.4417932953380017e-09 %
                                2. : Speed limit (100km/h) , probability: 2.0385299351333153e-09 %
                                3. : Speed limit (30km/h) , probability: 9.805928638528805e-10 %
                                4. : Speed limit (120km/h) , probability: 8.937371029493639e-23 %
    -------------------------------------------------------------------------------------------
    **** Image 2 :  
                     Actual_Class    == (Road work)
                     Predicted_Class == (Road work)
                           --> The Predicted label match actual class (Match) 
                     Top 5 Prediction Probabilities : 
                                0. : Road work , probability: 100.0 %
                                1. : Beware of ice/snow , probability: 0.0 %
                                2. : Dangerous curve to the right , probability: 0.0 %
                                3. : Slippery road , probability: 0.0 %
                                4. : Bumpy road , probability: 0.0 %
    -------------------------------------------------------------------------------------------
    **** Image 3 :  
                     Actual_Class    == (Speed limit (70km/h))
                     Predicted_Class == (Speed limit (70km/h))
                           --> The Predicted label match actual class (Match) 
                     Top 5 Prediction Probabilities : 
                                0. : Speed limit (70km/h) , probability: 100.0 %
                                1. : Speed limit (120km/h) , probability: 2.839175737320734e-29 %
                                2. : Traffic signals , probability: 3.588386140803613e-35 %
                                3. : Wild animals crossing , probability: 0.0 %
                                4. : Speed limit (100km/h) , probability: 0.0 %
    -------------------------------------------------------------------------------------------
    **** Image 4 :  
                     Actual_Class    == (Stop)
                     Predicted_Class == (Stop)
                           --> The Predicted label match actual class (Match) 
                     Top 5 Prediction Probabilities : 
                                0. : Stop , probability: 100.0 %
                                1. : No entry , probability: 1.8896825293782116e-11 %
                                2. : Speed limit (120km/h) , probability: 8.113796099686927e-15 %
                                3. : Turn left ahead , probability: 1.0902443959879593e-15 %
                                4. : Keep right , probability: 8.986370217431401e-21 %
    -------------------------------------------------------------------------------------------
    **** Image 5 :  
                     Actual_Class    == (Road work)
                     Predicted_Class == (Road work)
                           --> The Predicted label match actual class (Match) 
                     Top 5 Prediction Probabilities : 
                                0. : Road work , probability: 100.0 %
                                1. : Keep right , probability: 9.281085369902231e-34 %
                                2. : Beware of ice/snow , probability: 0.0 %
                                3. : Dangerous curve to the right , probability: 0.0 %
                                4. : Bumpy road , probability: 0.0 %
    -------------------------------------------------------------------------------------------
    **** Image 6 :  
                     Actual_Class    == (Priority road)
                     Predicted_Class == (Priority road)
                           --> The Predicted label match actual class (Match) 
                     Top 5 Prediction Probabilities : 
                                0. : Priority road , probability: 100.0 %
                                1. : End of no passing , probability: 1.0424416442212737e-19 %
                                2. : No vehicles , probability: 7.03311653731307e-21 %
                                3. : General caution , probability: 1.9332328242173194e-23 %
                                4. : End of no passing by vehicles over 3.5 metric tons , probability: 2.6288718792268265e-24 %
    -------------------------------------------------------------------------------------------
    **** Image 7 :  
                     Actual_Class    == (Turn left ahead)
                     Predicted_Class == (Bumpy road)
                           --> The Predicted label did NOT match actual class (No Match) 
                     Top 5 Prediction Probabilities : 
                                0. : Bumpy road , probability: 99.60378408432007 %
                                1. : Keep right , probability: 0.31697105150669813 %
                                2. : Stop , probability: 0.07868426037020981 %
                                3. : No vehicles , probability: 0.0005386809334595455 %
                                4. : Yield , probability: 2.499966456070979e-05 %
    -------------------------------------------------------------------------------------------
    **** Image 8 :  
                     Actual_Class    == (Roundabout mandatory)
                     Predicted_Class == (Roundabout mandatory)
                           --> The Predicted label match actual class (Match) 
                     Top 5 Prediction Probabilities : 
                                0. : Roundabout mandatory , probability: 99.99675750732422 %
                                1. : Speed limit (30km/h) , probability: 0.0029162916689529084 %
                                2. : Keep left , probability: 0.00032010827908379724 %
                                3. : Speed limit (100km/h) , probability: 2.1844334341381e-06 %
                                4. : Speed limit (20km/h) , probability: 1.4178387353069866e-07 %
    -------------------------------------------------------------------------------------------
    **** Image 9 :  
                     Actual_Class    == (No entry)
                     Predicted_Class == (No entry)
                           --> The Predicted label match actual class (Match) 
                     Top 5 Prediction Probabilities : 
                                0. : No entry , probability: 100.0 %
                                1. : No passing for vehicles over 3.5 metric tons , probability: 4.957702525890692e-32 %
                                2. : Turn left ahead , probability: 0.0 %
                                3. : Roundabout mandatory , probability: 0.0 %
                                4. : Stop , probability: 0.0 %
    -------------------------------------------------------------------------------------------
    **** Image 10 :  
                     Actual_Class    == (Bumpy road)
                     Predicted_Class == (Bumpy road)
                           --> The Predicted label match actual class (Match) 
                     Top 5 Prediction Probabilities : 
                                0. : Bumpy road , probability: 100.0 %
                                1. : Bicycles crossing , probability: 3.99737002032554e-18 %
                                2. : Keep right , probability: 0.0 %
                                3. : Traffic signals , probability: 0.0 %
                                4. : Children crossing , probability: 0.0 %
    -------------------------------------------------------------------------------------------
    **** Image 11 :  
                     Actual_Class    == (Right-of-way at the next intersection)
                     Predicted_Class == (Right-of-way at the next intersection)
                           --> The Predicted label match actual class (Match) 
                     Top 5 Prediction Probabilities : 
                                0. : Right-of-way at the next intersection , probability: 100.0 %
                                1. : Beware of ice/snow , probability: 1.7186554601110528e-35 %
                                2. : Double curve , probability: 0.0 %
                                3. : Roundabout mandatory , probability: 0.0 %
                                4. : Road work , probability: 0.0 %
    -------------------------------------------------------------------------------------------
    **** Image 12 :  
                     Actual_Class    == (Speed limit (30km/h))
                     Predicted_Class == (Speed limit (30km/h))
                           --> The Predicted label match actual class (Match) 
                     Top 5 Prediction Probabilities : 
                                0. : Speed limit (30km/h) , probability: 100.0 %
                                1. : Speed limit (70km/h) , probability: 0.0 %
                                2. : Speed limit (20km/h) , probability: 0.0 %
                                3. : Speed limit (50km/h) , probability: 0.0 %
                                4. : Speed limit (120km/h) , probability: 0.0 %
    -------------------------------------------------------------------------------------------
    **** Image 13 :  
                     Actual_Class    == (Road work)
                     Predicted_Class == (Road work)
                           --> The Predicted label match actual class (Match) 
                     Top 5 Prediction Probabilities : 
                                0. : Road work , probability: 100.0 %
                                1. : Keep right , probability: 9.281085369902231e-34 %
                                2. : Beware of ice/snow , probability: 0.0 %
                                3. : Dangerous curve to the right , probability: 0.0 %
                                4. : Bumpy road , probability: 0.0 %
    -------------------------------------------------------------------------------------------
    **** Image 14 :  
                     Actual_Class    == (Stop)
                     Predicted_Class == (Stop)
                           --> The Predicted label match actual class (Match) 
                     Top 5 Prediction Probabilities : 
                                0. : Stop , probability: 100.0 %
                                1. : Keep right , probability: 5.646316205386304e-12 %
                                2. : Turn left ahead , probability: 5.160572053948466e-15 %
                                3. : No vehicles , probability: 9.197248557024244e-17 %
                                4. : Speed limit (70km/h) , probability: 3.0401456294866656e-18 %
    

### Question 3.3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The model had a high accuracy with detecting images that is very similar to the original dataset. However image augementation helped with increased the model accuracy up to 90%
-------------------------------------------------------------------------------------------
**** Image 0 :  
                 * Actual_Class    == (Yield)
                 * Predicted_Class == (Yield) 
                 * Top 5 Prediction Probabilities : 
                            0. : Yield , probability: 99.99974966049194 %
                            1. : Priority road , probability: 0.0002510296326363459 %
                            2. : Keep right , probability: 5.3756829530463924e-14 %
                            3. : Turn left ahead , probability: 1.2825438001652313e-28 %
                            4. : Bicycles crossing , probability: 9.66601857675514e-29 %
-------------------------------------------------------------------------------------------
**** Image 1 :  
                 * Actual_Class    == (Speed limit (30km/h))
                 * Predicted_Class == (End of speed limit (80km/h))
                 * Top 5 Prediction Probabilities : 
                            0. : End of speed limit (80km/h) , probability: 100.0 %
                            1. : Speed limit (80km/h) , probability: 2.4417932953380017e-09 %
                            2. : Speed limit (100km/h) , probability: 2.0385299351333153e-09 %
                            3. : Speed limit (30km/h) , probability: 9.805928638528805e-10 %
                            4. : Speed limit (120km/h) , probability: 8.937371029493639e-23 %
-------------------------------------------------------------------------------------------
**** Image 2 :  
                 * Actual_Class    == (Road work)
                 * Predicted_Class == (Road work)
                 * Top 5 Prediction Probabilities : 
                            0. : Road work , probability: 100.0 %
                            1. : Beware of ice/snow , probability: 0.0 %
                            2. : Dangerous curve to the right , probability: 0.0 %
                            3. : Slippery road , probability: 0.0 %
                            4. : Bumpy road , probability: 0.0 %
-------------------------------------------------------------------------------------------
**** Image 3 :  
                 * Actual_Class    == (Speed limit (70km/h))
                 * Predicted_Class == (Speed limit (70km/h))
                 * Top 5 Prediction Probabilities : 
                            0. : Speed limit (70km/h) , probability: 100.0 %
                            1. : Speed limit (120km/h) , probability: 2.839175737320734e-29 %
                            2. : Traffic signals , probability: 3.588386140803613e-35 %
                            3. : Wild animals crossing , probability: 0.0 %
                            4. : Speed limit (100km/h) , probability: 0.0 %
-------------------------------------------------------------------------------------------
**** Image 4 :  
                 * Actual_Class    == (Stop)
                 * Predicted_Class == (Stop)
                 * Top 5 Prediction Probabilities : 
                            0. : Stop , probability: 100.0 %
                            1. : No entry , probability: 1.8896825293782116e-11 %
                            2. : Speed limit (120km/h) , probability: 8.113796099686927e-15 %
                            3. : Turn left ahead , probability: 1.0902443959879593e-15 %
                            4. : Keep right , probability: 8.986370217431401e-21 %
-------------------------------------------------------------------------------------------



## Step 4 (Optional): Visualize the Neural Network's State with Test Images


### Question 4.1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Visualizing the weights of different conv layers served as a great tool to understand how an input image preceived during the forward pass. in the below example Conv1 feature map visualized high level detail of the input sign (Sign Text, Border shape ..etc). while Conv2 feature maps are more focused into a smaller subset of the feature maps in conv1. 

This was surprising to me since i thought that the first layer is the lowest level in the hierarchy and it should detect more premitive shapes like edges and curves. 

"https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/da532f3b-29e8-43d1-9590-aa58909c28d1"



```python
### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    #print('  Number of Feature Maps  : {}'.format(int(len(featuremaps))))
    #print('  Feature Maps Dimensions : {}'.format(featuremaps[0].shape))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")

            
image_list=[]
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    image = X_Extra_normalized[14]
    image_list.append(image)
    print(image.shape)
  
    plt.imshow(X_Extra[14])
    plt.show()
    print('>>> conv1 Feature Maps Visualization (conv1 output = 14x14x6) :')
    print ('  Number of Feature Maps  : 6')
    print('  Feature Maps Dimensions : 14x14')
    outputFeatureMap(image_list, conv1, plt_num=1)
    plt.show()
    print('>>> conv2 Feature Maps Visualization (conv2 output = 5x5x16):')
    print ('  Number of Feature Maps  : 16')
    print('  Feature Maps Dimensions : 5x5')
    outputFeatureMap(image_list, conv2, plt_num=2)
    plt.show()    
        
```

    INFO:tensorflow:Restoring parameters from .\tsnet
    (32, 32, 1)
    


![png](output_69_1.png)


    >>> conv1 Feature Maps Visualization (conv1 output = 14x14x6) :
      Number of Feature Maps  : 6
      Feature Maps Dimensions : 14x14
    


![png](output_69_3.png)


    >>> conv2 Feature Maps Visualization (conv2 output = 5x5x16):
      Number of Feature Maps  : 16
      Feature Maps Dimensions : 5x5
    


![png](output_69_5.png)

