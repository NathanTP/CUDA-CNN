import ctypes
import ctypes.util
import sys
import pathlib
from mnist import MNIST
import numpy as np

from . import __state

def loadMnist(path, dataset='test'):
    mnistData = MNIST(str(path))

    if dataset == 'train':
        images, labels = mnistData.load_training()
    else:
        images, labels = mnistData.load_testing()

    images = np.asarray(images).astype(np.float32)
    labels = np.asarray(labels).astype(np.uint32)

    return images, labels

def loadModel(path):
    # While we could do more abstraction in C-land to make this easier from
    # python, we don't because we want to make each step very clear, some of them
    # will probably be factored out as we move toward KaaS.

    # host layers
    hlayers = [
            __state.cnnLib.layerParamsFromFile(str(path / 'l_input').encode('utf-8')),
            __state.cnnLib.layerParamsFromFile(str(path / 'l_c1').encode('utf-8')),
            __state.cnnLib.layerParamsFromFile(str(path / 'l_s1').encode('utf-8')),
            __state.cnnLib.layerParamsFromFile(str(path / 'l_f').encode('utf-8')),
           ]

    if None in hlayers:
        raise RuntimeError("Failed to load layers")

    # device layers
    dlayers = [ __state.cnnLib.layerParamsToDevice(l) for l in hlayers ]

    if None in dlayers:
        raise RuntimeError("Failed to load layers onto device")

    m = __state.cnnLib.newModel(dlayers[0], dlayers[1], dlayers[2], dlayers[3])
    if m is None:
        raise RuntimeError("Failed to initialize model")

    return m

def classify(model, imgBuf):
    return __state.cnnLib.classify(model, imgBuf)
