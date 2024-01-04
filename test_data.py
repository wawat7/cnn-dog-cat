import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from google.colab import drive
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = '/Users/wawatganteng/Development/kuliah/Cats-vs-dogs-classification-computer-vision-/images'
train_dir = '/Users/wawatganteng/Development/kuliah/Cats-vs-dogs-classification-computer-vision-/images/train'
validation_dir = '/Users/wawatganteng/Development/kuliah/Cats-vs-dogs-classification-computer-vision-/images/validation'
test_dir = '/Users/wawatganteng/Development/kuliah/Cats-vs-dogs-classification-computer-vision-/images/test'

train_cats_dir = '/Users/wawatganteng/Development/kuliah/Cats-vs-dogs-classification-computer-vision-/images/train/cats'
train_dogs_dir = '/Users/wawatganteng/Development/kuliah/Cats-vs-dogs-classification-computer-vision-/images/train/dogs'

validation_cats_dir = '/Users/wawatganteng/Development/kuliah/Cats-vs-dogs-classification-computer-vision-/images/validation/cats'
validation_dogs_dir ='/Users/wawatganteng/Development/kuliah/Cats-vs-dogs-classification-computer-vision-/images/validation/dogs'

test_cats_dir = '/Users/wawatganteng/Development/kuliah/Cats-vs-dogs-classification-computer-vision-/images/test/cats'
test_dogs_dir = '/Users/wawatganteng/Development/kuliah/Cats-vs-dogs-classification-computer-vision-/images/test/dogs'

num_cats_train = len(os.listdir(train_cats_dir))
num_dogs_train = len(os.listdir(train_dogs_dir))

num_cats_validation = len(os.listdir(validation_cats_dir))
num_dogs_validation = len(os.listdir(validation_dogs_dir))

num_cats_test = len(os.listdir(test_cats_dir))
num_dogs_test = len(os.listdir(test_dogs_dir))

BATCH_SIZE = 120
IMG_SHAPE  = 150

"""###### Pre-Processing for testing data"""

# No data augmentation for testing
image_gen_test = ImageDataGenerator(rescale=1./255)

test_data_gen = image_gen_test.flow_from_directory(batch_size=BATCH_SIZE,
                                                   directory=test_dir,
                                                   target_size=(IMG_SHAPE, IMG_SHAPE),
                                                   class_mode='binary')

"""###### Testing model"""

# Load the saved model
loaded_model = tf.keras.models.load_model('Cat_vs_dogs_classification.h5')

# Evaluate the model on the test set
results = loaded_model.evaluate(test_data_gen)
print("Test loss, test accuracy:", results)

# Note: The loaded model is used for evaluation, so there's no need to compile it again.

# ... (existing code)
