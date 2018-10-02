from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.utils import plot_model

model = ResNet50(weights='imagenet')
model_vgg16 = VGG16(weights='imagenet', include_top=False)

# Plot model to image file
plot_model(model, to_file="ResNet50.png", show_shapes=True)
plot_model(model_vgg16, to_file="VGG16_withoutTop.png", show_shapes=True)

img_path = 'data/cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)

print("Predicted:", decode_predictions(preds, top=10)[0])

features = model_vgg16.predict(x)

print (features.shape)