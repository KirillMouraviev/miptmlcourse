import os
import numpy as np
import pandas as pd
from models.settings import *
from models.utils import quadratic_weighted_kappa, get_class_weights 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from models.VGG_model import initialize_model
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, Activation, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.applications.vgg16 import VGG16

seed = 9
np.random.seed(seed)
model = initialize_model()
for layer in model.layers:
    layer.trainable = True

last_checkpoint = 'weights-improvement-13-0.55.hdf5'
model.load_weights('./weights_new/' + last_checkpoint)

#opt = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
opt = SGD(lr=1e-4)
model.compile(loss = "mse", optimizer = opt, metrics=['accuracy'])

# Initialize generators
train_datagen = ImageDataGenerator()
test_datagen  = ImageDataGenerator()

# Generator data flows
train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    class_mode='categorical', shuffle=True,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE)

validation_generator = test_datagen.flow_from_directory(
    VALIDATION_DATA_DIR,
    class_mode='categorical', shuffle=True,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE)

# get the number of training examples
#NB_TRAIN_SMPL = sum([len(files) for r, d, files in os.walk(TRAIN_DATA_DIR)])
NB_TRAIN_SMPL = 3000

# get the number of validation examples
NB_VAL_SMPL = sum([len(files) for r, d, files in os.walk(VALIDATION_DATA_DIR)])

# get class weights for dealing with class imbalances
correct_train_labels = pd.read_csv('trainLabels.csv')
labels_list = correct_train_labels['level'].values.tolist()
class_weights = get_class_weights(labels_list)

# configure checkpoints
filepath = WEIGHTS_DIR + "weights-improvement-{:02d}-{:.2f}.hdf5"
#model_checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#callbacks_list = [ model_checkpoint]
n_epochs = 10

for epoch in range(n_epochs):
    history = model.fit_generator(
        train_generator,
        steps_per_epoch= NB_TRAIN_SMPL//BATCH_SIZE,
        epochs=1,
        shuffle=True,
        validation_data=validation_generator,
        validation_steps=NB_VAL_SMPL//BATCH_SIZE,
        class_weight=class_weights)
        #callbacks = callbacks_list)
    val_acc = history.history['val_acc'][0]
    print('val_acc:', val_acc)
    model.save_weights(filepath.format(epoch, float(val_acc)))

# save history
pd.DataFrame(history.history).to_csv("history.csv")