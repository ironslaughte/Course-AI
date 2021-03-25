from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2


def show_loss(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='swish', input_shape=(28, 28, 1)))
model.add(layers.Conv2D(64, (3, 3), activation='swish', kernel_regularizer=tf.keras.regularizers.l2(1e-5)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='swish', activity_regularizer=tf.keras.regularizers.l2(1e-4)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation='softmax'))

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255


test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

test_labels = to_categorical(test_labels)
train_labels = to_categorical(train_labels)

validation_images = train_images[:1000]
validation_labels = train_labels[:1000]

train_labels = train_labels[1000:]
train_images = train_images[1000:]
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)  # случайное сальnо. Поворот изображения на
# n*90 градусов, n - целое число
datagen_train = ImageDataGenerator(horizontal_flip=True)  # случайный поворот до 90 градусов
datagen_train.fit(train_images)

datagen_val = ImageDataGenerator(horizontal_flip=True)
datagen_val.fit(validation_images)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(datagen_train.flow(train_images, train_labels, batch_size=256), epochs=20,
                     validation_data=datagen_val.flow(validation_images, validation_labels, batch_size=256))

show_loss(history)
model.evaluate(test_images, test_labels)
print(model.summary())


img = test_images[18]
plt.imshow(img.reshape(28, 28)*255, cmap='binary')
x = img.reshape(1, 28, 28, 1)


preds = model.predict(x)
cur_class = np.argmax(preds)
print("Класс: ", class_names[cur_class], "\nИндекс:", cur_class)

# This is the "african elephant" entry in the prediction vector
cur_class_output = model.output[:, cur_class]

# The is the output feature map of the `block5_conv3` layer,
# the last convolutional layer in VGG16
last_conv_layer = model.get_layer('conv2d_2')

# This is the gradient of the "african elephant" class with regard to
# the output feature map of `block5_conv3`
grads = K.gradients(cur_class_output, last_conv_layer.output)[0]

# This is a vector of shape (64,), where each entry
# is the mean intensity of the gradient over a specific feature map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# This function allows us to access the values of the quantities we just defined:
# `pooled_grads` and the output feature map of `block5_conv3`,
# given a sample image
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

# These are the values of these two quantities, as Numpy arrays,
# given our sample image of two elephants
pooled_grads_value, conv_layer_output_value = iterate([x])

# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the elephant class
for i in range(64):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# The channel-wise mean of the resulting feature map
# is our heatmap of class activation
heatmap = np.mean(conv_layer_output_value, axis=-1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()


img = img*255
rgbimg = cv2.merge([img, img, img])

# We resize the heatmap to have the same size as the original image
heatmap = cv2.resize(heatmap, (rgbimg.shape[1], rgbimg.shape[0]))

# We convert the heatmap to RGB
heatmap = np.uint8(255 * heatmap)

# We apply the heatmap to the original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 0.7 here is a heatmap intensity factor
superimposed_img = heatmap*0.7 + rgbimg

# Save the image to disk
superimposed_img = cv2.resize(superimposed_img, (540, 540))
cv2.imwrite("result.jpg", superimposed_img)
