from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

loaded_model = load_model('Cat_vs_dogs_classification.h5')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

def classify_image(img_path):
    preprocessed_img = preprocess_image(img_path)
    prediction = loaded_model.predict(preprocessed_img)
    print('kucing : ', prediction[0][0])
    print('anjing : ', prediction[0][1])
    if prediction[0][0] > prediction[0][1]:
        return "Cat"
    else:
        return "Dog"

test_image_path = './image/test/cats/cat.4005.jpg'
result = classify_image(test_image_path)

img = image.load_img(test_image_path)
plt.imshow(img)
plt.title(f"Predicted: {result}")
plt.show()

