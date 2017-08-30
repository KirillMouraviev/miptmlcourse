from .settings  import *
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam


def initialize_model():
    
    model = Sequential()
    model.add(Conv2D(16, (4, 4), strides=(2, 2), padding='valid', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.1))

    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))

    model.add(Flatten())
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(NB_CLASSES))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    opt = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss = "mse", optimizer = opt, metrics=['accuracy'])

    print(model.summary())

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    return model