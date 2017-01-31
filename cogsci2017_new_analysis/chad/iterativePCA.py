import os
import glob
import random
import numpy as np
from PIL import Image
from scipy.misc import imsave, imread
from sklearn.externals import joblib
from sklearn.decomposition import IncrementalPCA, PCA

WD = os.getcwd()
BATCH = 1800
N_COMP = 1700
DIM = 140
DIMS = (DIM, DIM)
EPOCHS = 1
ITERS = 8


def run():
    imgs = compile_list()
    for iter in range(ITERS):
        random.shuffle(imgs)
        images = imgs[:10000]
        pca = inc_pca(images)
        joblib.dump(pca, WD+'/pca_file')
        project_face(WD+'/beauty.jpg', pca, WD+'/corrected_' + str(iter) + '.jpg')
        #project_face(WD+'/corrected.jpg', pca, WD+'/corrected_l2.jpg')
    average_faces(ITERS, WD+'/averaged_' + str(ITERS) + '.jpg')


def compile_list():
    """Compile list of jpgs in dataset."""
    image_list = []
    for filename in glob.glob(WD + '/data/*/*/*.jpg'):
        image_list.append(filename)
    #-----REMOVE------------
    #image_list = image_list[:1500]
    #-----REMOVE------------
    return image_list


def subsample(images):
    """Subsample images to uniform size."""

    inMemory = list(images)
    inMemory = [Image.open(path) for path in images]
    for i in range(len(inMemory)):
            x = inMemory[i]
            x.thumbnail(DIMS)
            background = Image.new('RGB', DIMS)
            background.paste( x, (int((DIMS[0] - x.size[0]) / 2), int((DIMS[1] - x.size[1]) / 2)))
            inMemory[i] = background
    inMemory = [np.array(x, np.float32)/255. - .5 for x in inMemory]
    return inMemory


def flatten(inMemory):
    """Flatten image to 1d."""
    inMemory = [x.flatten() for x in inMemory]
    return inMemory


def inc_pca(images):
    """Perform PCA on all images."""

    total = len(images)
    groupPCA = IncrementalPCA(n_components=N_COMP)

    # Fit on all in chunks
    for e in range(EPOCHS):
        start = 0
        random.shuffle(images)
        while start < total:
            end = min(start+BATCH, total)
            if (end-start < N_COMP):
                break
            print "Processing batch from " + str(start) + " to " + str(end) + ".\n"
            subset = images[start:end]
            inMemory = subsample(subset)
            inMemory = flatten(inMemory)
            groupPCA.partial_fit(inMemory)
            start = end

    return groupPCA


def project_face(path, pca, dest):
    im = subsample([path])
    im = flatten(im)[0]

    tr = pca.transform(im)
    x = pca.inverse_transform(tr)

    x = np.reshape(x, (3, DIM, DIM))
    x = [(j +.5)*255 for j in x]
    x = np.reshape(x, (DIM, DIM, 3))
    x = np.uint8(x)
    img = Image.fromarray(x)
    img.save(dest)

def average_faces(cnt, dest):
    ims = []
    for c in range(cnt):
        ims.append(np.array(Image.open(WD+'/corrected_' + str(c) + '.jpg'),
        np.uint32))
    acc = ims[0]
    for c in range(1,cnt):
        acc += ims[c]
    acc /= cnt
    acc = np.array(acc, np.uint8)

    img = Image.fromarray(acc)
    img.save(dest)


run()
