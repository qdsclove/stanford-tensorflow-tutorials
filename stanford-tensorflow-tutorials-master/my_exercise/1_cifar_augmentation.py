from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

img_generator = ImageDataGenerator(
                rotation_range=90,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range=0.3 )

img_path = "data/kite.jpg"
img = image.load_img(img_path)
plt.imshow(img)


x = image.img_to_array(img)
print (x.shape)
x = np.expand_dims(x, axis=0)

print (x.shape)

gen = img_generator.flow(x, batch_size=1)


plt.figure()
for i in range(3):
    for j in range(3):
        x_batch = next(gen)
        print ("x_batch.shape:", x_batch.shape)
        idx = (3*i) + j
        plt.subplot(3, 3, idx+1)
        plt.imshow(x_batch[0]/256)
plt.show()