from .settings  import *
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, Activation, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.applications.vgg16 import VGG16


def initialize_model():
    model = VGG16(include_top=False, input_shape=(512, 512, 3), weights=None, classes=5)
    model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    
    for layer in model.layers:
        layer.trainable = False
        
    x = model.layers[-1].output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(rate=0.3)(x)
    x = Dense(NB_CLASSES)(x)
    x = Activation('softmax')(x)
    model = Model(model.input, x, model.weights)

    print(model.summary())
    
    
    '''print(model.summary())

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)'''

    return model