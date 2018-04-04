# A Homemade Teachable Machine

[Demo](https://youtu.be/4G_cuvfPGxA)

A Program that can be trained to recognize images captured by a webcam.

For example:

- It might be trained to recognize when the user is smiling, frowning, or making a silly face, or
- It might be trained to identify the type of object they’re holding, say - a book, a coffee cup, or a tennis ball, or
- To recognize whether a cat appears in the video with you.

The program does not know what types of images it has been trained to recognize in advance.

To collect training data, the program captures a few seconds of video in which the user demonstrates each scene they want to recognize.

For example, to train a classifier for smiling vs. frowning vs. silly faces - the user captures a few seconds of video of each pose. Then, the program extract frames from the video, use them to train an image classification model and begin using it to classify images it receives from the webcam.

Inspired by https://teachablemachine.withgoogle.com/

## Getting Started

Hardware ​required:​

- A Laptop
- A Webcam

After activating a new virtual environment, run these commands inside it:

```bash
git clone https://github.com/vmvargas/cs632/tree/master/project
cd project
pip install -r requirements.txt
```

## Deployment

The file ```CONST.py``` only contains global variables used by all files.

### 1. Collect the training data

Run ```collect_training_data.py``` this script allows the user will to collect training data for each of the
categories they want to recognize. The program limits the user to use **3 categories**.

To collect training data, the user types the name of the label he is about to record. Then, the webcam is open and the program asks the user to save **50 frames** in real-time whenever the user presses the key **r**. Note: keeping 'r' held will record consecutive frames.

The training data is saved inside ```./dataset``` folder. Each sub-folder has the name of the label the user gave and contains the **50 frames**.

### 2. Train and evaluate the model

I trained two image classifiers, a ```simple CNN from scratch``` (as a baseline) and one using the bottleneck features of ```a pre-trained VGG16 CNN```, both follow the same preliminary steps:

- Load the frames recorded.
- Preprocess the images by:
  - Reducing its resolution,
  - Using one-hot-encoding,
  - Scaling the data points from [0,255] to [0,1],
  - Spitting the dataset in train and validation 75-25 ratio and in stratified chunks

The main difference relies on their architecture:

#### 2.1 A simple CNN trained from scratch

Execute ```$ python train-crappy-model.py``` to train this model.

You can also find the model saved here: ```saved_models/crappy_model.h5```.

Nowadays, the right tool for an image classification job is a CNN, so I tried to train one simple to start. Since I only have few examples, my first concern was overfitting or memorizing the data.

My main focus for degrading overfitting was how much information my model was allowed to store, because, a model that can only store a few features will have to focus on the most significant features found in the data, and these are more likely to be truly relevant and to generalize better.

That is why this model consists of a simple stack of 3 convolution layers with a ReLU activation, followed by max-pooling layers. On top of it, I ended the model with a single unit and a sigmoid activation.

For a dataset of 150 images, with EPOCHS = 15 and BATCH_SIZE = 15, this CNN gave me a validation accuracy between 75-95% in less than 1 minute of training on CPU.

EPOCHS value was picked arbitrarily because the model is small and uses aggressive dropout, it does not seem to be overfitting too much at that point.

##### Approaches that might benefit this model

- Adding data augmentation is one way to reduce overfitting, but it could not be enough since the augmented samples could be still highly correlated.

- A way to accentuate the entropic capacity of the model to generalize better is the use of weight regularization or "weight decay" to force model weights to taker smaller values.

- The variance of the validation accuracy is quite high because accuracy is a high-variance metric but mainly because I only used 38 validation samples. A good validation strategy in these cases could be k-fold cross-validation, but this would require training k models for every evaluation round.

#### 2.2 A pre-trained VGG16 CNN

Run ```train-vgg16.py``` to train this model.

You can also find the model saved here: ```saved_models/bottleneck_model.h5```.

Trying to refined my previous approach, I developed the suggestion of experimenting with Transfer Learning to leverage a model pre-trained on a much larger dataset.

I used the VGG16 architecture, pre-trained on the ImageNet dataset (that certainly contains similar images to my dataset). I believe it is possible that merely recording the softmax predictions of the model over my data would have been sufficient but maybe not for the grade this project. Nevertheless, I believe the method I used here is likely to generalize well to a broad range of scenarios as well.

##### Procedure

- I only instantiate the convolutional part of the model, that is, everything up to the fully-connected layers.
- Ran this model on my training and validation data only one time.
- Recorded the outputs of the last activation maps before the fully-connected layers in two numpy arrays (i.e. bottleneck features of the VGG16 model).
- Used this output to train a small fully-connected model on top of the stored features.

I stored the features is merely for computational efficiency. This way I only had to run it once.

For the same dataset of 150 images, with EPOCHS = 50 and BATCH_SIZE = 10, this model gave me a validation accuracy of 99% in less than 3 minutes of training on CPU.

EPOCHS value was picked arbitrarily because it does not seem to be overfitting too much at that point.

Validation accuracy was great due to the fact that the base model was trained on a dataset that already featured the objects that I showed (face, hands, and so on).

##### Approaches that might benefit this model

- Fine-tuning one or two convolutional blocks at the end (alongside with greater regularization).

- Adding an aggressive data augmentation (because of the same reason as the previous approach).

- Increasing dropouts values, so that the network becomes less sensitive to the specific weights of neurons.

- Using L1 and L2 regularization, a.k.a. weight decay (because of the same reason as the previous approach).

- Training a linear classifier (e.g. SVM) on top of the bottleneck features. SVM is particularly good at drawing decision boundaries on a small dataset.

### 3. Classify Images

Once a model is trained, execute ```$ python test.py``` to classify images from the webcam in real-time.

This script follows these steps:

1. Load the **pre-trained CNN** model and its weights.
2. Prepare the webcam to record.
3. Preprocess each frame it receives in the same way as the training data.
4. Generates a prediction of the frame preprocessed.
5. Give a visual and **auditive feedback** to indicate which category the current frame is classified as.

## Considerations

### Extract​ing frames

Frames are extracted every time the user presses the key 'r' and until the programs reach **50 frames**. I choose this number arbitrarily because I want the program to learn only one label per sequence of frames, thus I make sure that these frames were somehow different yet highly correlated.

This program takes the image as it is recorded by the webcam - a 640x480px color picture - and save it in the corresponding folder.

The program treats the output of the webcam as a collection of independent images.

### Input​ ​preprocessing

Training, validation and testing dataset were preprocessed in the same way.

Considering the tiny amount of data I had, each frame was downsized to **150x150 px**. I started using half of this size and still got a good accuracy but I wanted the model to be a bit sensitive to subtle changes among labels such as the different facial expression case.

### Transfer​ ​learning​

I trained two models to evaluate the difference between a model from scratch vs transfer learning on top of a pre-train model.

I previously explain both procedures. Now, I'll highlight some considerations:

- I expect my dataset won't represents very specific domain features, like medical images or Chinese handwritten characters. If that were the case. I should definitely prefer the CNN from scratch.

- Taking the output of the intermediate layer prior to the fully connected layers as features (bottleneck features) of a pre-trained model (VGG16 in this case) and then training a small fully-connected model on top of the stored feature will take advantage of the knowledge gained by the pre-trained network (like curves and edges) combine with the fine-tune of my own fully-connected classifier at the end to customize the output.

### Underfitting​ ​vs.​ ​Overfitting​

Training a CNN on a small dataset (one that is smaller than the number of parameters) greatly affects the CNN ability to generalize, often result in overfitting. That is why fine-tuning existing networks that are trained on a large dataset like the ImageNet (1.2M labeled images) yield to better accuracy and generalization.

Also due to the fact that my dataset is not drastically different in context to the original dataset (ImageNet), the pre-trained model has already learned features that are relevant to our own classification problem.

## Built With

- tensorflow 1.3
- keras 2
- opencv 3
- pil
- gtts
- pygame
- sklearn
- numpy
- matplotlib
- imutils
- pandas