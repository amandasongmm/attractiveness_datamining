import os
import h5py
import numpy as np
import face_autoencoder
from scipy.misc import imread
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger

WD = os.getcwd()
BATCH = 200
VBATCH = 50
BATCHES_PER = 10
EPOCHS = 100
img_width = 100
img_height = 100

train_log = CSVLogger(WD + '/results.csv')

train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_gen = train_gen.flow_from_directory(
        WD+'/data/train/',
        target_size = (img_width, img_height),
        batch_size = BATCH,
        shuffle = True,
        seed=1,
        class_mode = None)

val_gen = val_gen.flow_from_directory(
        WD+'/data/val',
        target_size = (img_width, img_height),
        batch_size = VBATCH,
        shuffle = True,
        seed=1,
        class_mode = None)

autoencoder = face_autoencoder.build_model()

#stats = []
for e in range(EPOCHS):
    print 'Running Epoch #' + str(e)
    i = 0
    for X_batch in train_gen:
        #for i in range(BATCHES_PER):
        #X_batch = train_gen.flow_from_directory()
        #V_batch = val_gen.flow_from_directory()
        X_batch -= .5
        train = autoencoder.train_on_batch(X_batch, X_batch)
        print train
        #val = autoencoder.train_on_batch(V_batch, V_batch)
        #stats.append((train,val))
        i += 1
        if (i >= BATCHES_PER):
            print 'End Epoch #' + str(e) +'\n'
            #stats.append(train)
            if (e % 5 == 0):
                autoencoder.save('beauty_autoencoder_' + str(e) + '.h5')
            break

#np.save(WD+'/stats',stats)

#autoencoder.save('beauty_autoencoder.h5')

'''
autoencoder.fit_generator(
        (train_gen, train_gen),
        samples_per_epoch = BATCH,
        nb_epoch = EPOCHS,
        validation_data = (val_gen, val_gen),
        nb_val_samples = VBATCH,
        callbacks = [train_log])
'''
