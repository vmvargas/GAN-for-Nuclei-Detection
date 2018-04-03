from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import sys
import numpy as np

class DCGAN():
    def __init__(self):
        # MNIST input shape is 28x28x1
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1

        # While previous GAN work has used momentum to accelerate training, we used the Adam optimizer
        # (Kingma & Ba, 2014) with tuned hyperparameters. We found the suggested learning rate of 0.001,
        # to be too high, using 0.0002 instead. Additionally, we found leaving the momentum term β1 at the
        # suggested value of 0.9 resulted in training oscillation and instability while reducing it to 0.5 helped 
        # stabilize training
        optimizer = Adam(0.0002, 0.5)

        # Build discriminator's model
        self.discriminator = self.build_discriminator()
        # Compile discriminator's model, i.e. define its learning process
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generates imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        # Replace any pooling layers with strided convolutions.
        # Allowing it to learn its own spatial upsampling
        
        noise_shape = (100,)

        model = Sequential()
        # 128*7*7
        model.add(Dense(128 * 7 * 7, activation="relu", input_shape=noise_shape))
        model.add(Reshape((7, 7, 128)))
        model.add(BatchNormalization(momentum=0.8))

        # fractionally-strided convolution
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        # using a bounded activation allowed the model to learn more quickly to saturate and cover the color space of the training distribution
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        #upsampling is the opposite to pooling. Repeats the rows and columns of the data
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        
        #flatten to the amount of channels
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        # Replace any pooling layers with strided convolutions
        # Allowing it to learn its own spatial downsampling

        img_shape = (self.img_rows, self.img_cols, self.channels)
        
        # A Sequential model is a linear stack of layers.
        model = Sequential()
        # Create a Sequential model by simply adding layers via the .add() method
        # 32 filters, 3x3 kernel size, stride 2, input_shape is 28x28x1, same: pad so the output and input size are equal 
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        # f(x) = alpha * x for x < 0, f(x) = x for x >= 0.
        # Leaky rectified activation worked well, especially for higher resolution modeling.
        # This is in contrast to the original GAN paper, which used the maxout activation
        model.add(LeakyReLU(alpha=0.2))
        # drops 25% of the input units
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        #A zero-padding layer. Adds rows and columns of zeros to the image
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        # Normalize the activations of the previous layer at each batch to reduce its covariance shift,
        # i.e., the amount that the distribution of each layer shift around.

        # This helps deal with training problems that arise due to poor initialization and helps gradient flow in deeper models. 
        # This proved critical to get deep generators to begin learning, preventing the generator from collapsing all samples
        # to a single point which is a common failure mode observed in GANs. 
        # 
        # Directly applying batchnorm to all layers, however, resulted in sample oscillation and model instability. 
        # This was avoided by not applying batchnorm to the generator output layer and the discriminator input layer
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()
        # instantiate a Keras tensor
        img = Input(shape=img_shape)
        validity = model(img)
        
        # Given an img (x) and a validity (y), instantiate a Model.
        # This model will include all layers required in the computation of b given a.
        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Normalize values from -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            # Draw random samples from a Gaussian distribution.
            noise = np.random.normal(0, 1, (half_batch, 100))
            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            # train_on_batch: Single gradient update over one batch of samples
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        #fig.suptitle("DCGAN: Generated digits", fontsize=12)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("./dcgan-images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=4000, batch_size=32, save_interval=50)