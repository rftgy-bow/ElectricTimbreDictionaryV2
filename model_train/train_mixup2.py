# -*- coding=utf-8 -*-
import numpy as np
from tensorflow import keras, saved_model, distribute
#from keras.models import Model
#from keras.layers import Input, Dense, Dropout, Activation
#from keras.layers import Conv2D, GlobalAveragePooling2D
#from keras.layers import BatchNormalization, Add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

import os
from matplotlib import pyplot as plt
from new_model import CBACNN
from mixup import MixupGenerator
import gc
import time as tm

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#if len(physical_devices) > 0:
#    for k in range(len(physical_devices)):
#        tf.config.experimental.set_memory_growth(physical_devices[k], True)
#        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
#else:
#    print("Not enough GPU hardware devices available")
   
strategy = distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))    


# IMPORT DATASET
# dataset files
train_files = ["mel_npz/esc_melsp_train_raw.npz", 
               "mel_npz/esc_melsp_train_ss.npz",
               "mel_npz/esc_melsp_train_st.npz", 
               "mel_npz/esc_melsp_train_wn.npz",
               "mel_npz/esc_melsp_train_com.npz"]
test_file = "mel_npz/esc_melsp_test.npz"

train_num = 1500
test_num = 500

freq = 128 # melsp
time = 862 # 5 * 44100 / 256

# define dataset placeholders
print("making placeholders...")
x_train = np.zeros(freq*time*train_num*len(train_files)).reshape(
    train_num*len(train_files), freq, time)
y_train = np.zeros(train_num*len(train_files))

# load dataset
for i in range(len(train_files)):
    print("loading datasets...")
    data = np.load(train_files[i])
    x_train[i*train_num:(i+1)*train_num] = data["x"]
    y_train[i*train_num:(i+1)*train_num] = data["y"]

# load test dataset
test_data = np.load(test_file)
x_test = test_data["x"]
y_test = test_data["y"]

# convert target data into one-hot vector
classes = 50
print("converting datasets...")
y_train = keras.utils.to_categorical(y_train, classes)
y_test = keras.utils.to_categorical(y_test, classes)

# reshape source data
x_train = x_train.reshape(train_num*5, freq, time, 1)
x_test = x_test.reshape(test_num, freq, time, 1)
print("finished")


# DEFINE MODEL
with strategy.scope():
    model = CBACNN.exportModel(x_train.shape[1:], classes)
    opt = keras.optimizers.Adam(lr=0.00001, decay=1e-6, amsgrad=True)
    model.compile(
        loss=['categorical_crossentropy', None], # train output "final" only
        optimizer=opt, metrics=['accuracy'])
        
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, profile_batch = 100000000)
]

# TRAIN MODEL
start = tm.time()

print("***start training***")        
model.fit(x_train,y_train,
    validation_data=(x_test,y_test),
    epochs=10, verbose=1, batch_size=32,
    callbacks=callbacks)

print("training finished!")
print("***exec time:", tm.time() - start, "sec***")

# make SavedModel
model.save("./saved_model2")
print("stored trained model as <<saved_model>>")

# CALLBACKS
# early stopping
es_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

# TRAIN WITH MIXUP
batch_size = 32
epochs = 10
training_generator = MixupGenerator(x_train, y_train)()
print("***start MIXUP training***")  
model.fit_generator(generator=training_generator,
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    validation_data=(x_test, y_test),
                    epochs=epochs, 
                    verbose=1,
                    shuffle=True,
                    callbacks=[es_cb])

print("training finished!")


# make SavedModel
saved_model.save(model, "./saved_model2_mixup")
print("stored trained model as <<saved_model>>")


# EVALUATION
evaluation = model.evaluate(x_test, y_test)
print(evaluation)
print("evaluation finished!")

keras.backend.clear_session()
gc.collect()
