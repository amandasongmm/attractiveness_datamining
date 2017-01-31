import os
import numpy as np
from PIL import Image
from keras.models import load_model

WD = os.getcwd()
DIMS = (100, 100)
NUM = 95

autoencoder = load_model('beauty_autoencoder_' + str(NUM) + '.h5')

x = Image.open(WD+'/beauty.jpg')
x.thumbnail(DIMS)
background = Image.new('RGB', DIMS)
background.paste( x, (int((DIMS[0] - x.size[0]) / 2), int((DIMS[1] - x.size[1]) / 2)))
x = np.array(background, np.float32)/255.0 - .5
#x = np.reshape(x, (3, 100, 100))
x = np.array([x])

result = autoencoder.predict(x)

x = result[0] + .5
x *= 255
#x = np.reshape(x, (100, 100, 3))
x = np.uint8(x)
img = Image.fromarray(x)
img.save(WD+'/corrected_autoencoder.jpg')
