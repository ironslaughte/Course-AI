import tensorflow.keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()


def show_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss',color='b')
    plt.plot(epochs, val_loss, 'b', label='Validation loss', color='r')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

    mae = history.history['mae']
    val_mae = history.history['val_mae']
    plt.plot(epochs, mae, 'bo', label='Training mae',color='b')
    plt.plot(epochs, val_mae, 'b', label='Validation mae',color='r')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(256, activation="swish", kernel_regularizer=tensorflow.keras.regularizers.l2(l2=1e-4),
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(256, activation="swish", kernel_regularizer=tensorflow.keras.regularizers.l2(l2=1e-4)))
    model.add(layers.Dense(1))
    # задача регрессии
    model.compile(optimizer='adam', loss="mse", metrics=["mae"])
    return model


print("Сколько мы имеем данных:", train_data.shape, test_data.shape, sep="\n")
print("Тип данных", type(train_data), type(train_targets))
print("Как вылядят данные:", train_data[1], sep="\n")
print("Как выглядят наши ответы:", train_targets[1], sep="\n")

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

# наивный вариант разделения данных на тренировочные и валидационные
val_data = train_data[:60]
val_targets = train_targets[:60]
train_data = train_data[60:]
train_targets = train_targets[60:]
model = build_model()

history = model.fit(train_data, train_targets,validation_data=(val_data,val_targets), epochs=30, batch_size=10)

#print(history.history.keys())

show_history(history)

val_mse, val_mae = model.evaluate(test_data, test_targets)

# разделение на k блоков и оценка сети
#k = 4
#num_val_samples = len(train_data) // k
#all_scores = []
#for i in range(k):
#    print('processing fold #', i)
#    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
#    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
#    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
#                                         train_data[(i + 1) * num_val_samples:]], axis=0)
#    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
#                                            train_targets[(i + 1) * num_val_samples:]],
#                                           axis=0)
#    model = build_model()
#    history = model.fit(partial_train_data, partial_train_targets,
#                        epochs=100, validation_data=(val_data, val_targets), batch_size=5, verbose=0)
    # show_history(history)
#    val_mse, val_mae = model.evaluate(test_data, test_targets, verbose=0)
#    all_scores.append(val_mae)

#print(np.mean(all_scores))
