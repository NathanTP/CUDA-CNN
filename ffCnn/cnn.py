import ctypes
import ctypes.util
import sys
import pathlib
from mnist import MNIST
import numpy as np
import types

import libff as ff
import libff.kv
import libff.invoke
import kaasServer as kaas

redisPwd = "Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy"

modelLibPath = pathlib.Path("../libkaascnn/libkaascnn.so").resolve()
modelDir = pathlib.Path("../model").resolve()
dataDir = pathlib.Path("../data").resolve()

class layerParams(ctypes.Structure):
    _fields_ = [    # Vector of N floats
                    ('bias', ctypes.POINTER(ctypes.c_float)),

                    # Matrix of NxM floats
                    ('weight', ctypes.POINTER(ctypes.c_float)),

                    # Vector of O floats
                    ('output', ctypes.POINTER(ctypes.c_float)),

                    # Vector of O floats (only ever exists on the device)
                    ('preact', ctypes.POINTER(ctypes.c_float)),

                    ('M', ctypes.c_int),
                    ('N', ctypes.c_int),
                    ('O', ctypes.c_int),
                    ('onDevice', ctypes.c_bool)
            ]


def getLibCnnHandle():
    """Imports libCnn for local usage. As a KaaS client, we're only interested
    in the helpers for parsing saved models."""

    s = types.SimpleNamespace()

    packagePath = pathlib.Path(__file__).parent.parent
    # libcnnPath = packagePath / "libkaascnn" / "libkaascnn.so"

    s.cnnLib = ctypes.cdll.LoadLibrary(modelLibPath)
    
    # Function signatures
    s.cnnLib.initLibkaascnn.restype = ctypes.c_bool

    s.cnnLib.layerParamsFromFile.restype = ctypes.POINTER(layerParams)
    s.cnnLib.layerParamsFromFile.argtypes = [ ctypes.c_char_p ]

    s.cnnLib.layerParamsToDevice.restype = ctypes.POINTER(layerParams)
    s.cnnLib.layerParamsToDevice.argtypes = [ctypes.POINTER(layerParams)]

    s.cnnLib.newModel.restype = ctypes.c_void_p
    s.cnnLib.newModel.argtypes = [ctypes.c_void_p]*4

    s.cnnLib.printLayerWeights.argtypes = [ ctypes.POINTER(layerParams) ]

    s.cnnLib.classify.restype = ctypes.c_uint32
    s.cnnLib.classify.argtypes = [ ctypes.c_void_p, np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS") ]

    return s


def getCtx(remote=False):
    if remote:
        objStore = ff.kv.Redis(pwd=redisPwd, serialize=True)
    else:
        objStore = ff.kv.Local(copyObjs=False, serialize=False)

    return libff.invoke.RemoteCtx(None, objStore)


def loadMnist(path, dataset='test'):
    mnistData = MNIST(str(path))

    if dataset == 'train':
        images, labels = mnistData.load_training()
    else:
        images, labels = mnistData.load_testing()

    images = np.asarray(images).astype(np.float32)
    labels = np.asarray(labels).astype(np.uint32)

    return images, labels


def loadLayersLocal(path):
    # host layers
    hlayers = [
            __state.cnnLib.layerParamsFromFile(str(path / 'l_input').encode('utf-8')),
            __state.cnnLib.layerParamsFromFile(str(path / 'l_c1').encode('utf-8')),
            __state.cnnLib.layerParamsFromFile(str(path / 'l_s1').encode('utf-8')),
            __state.cnnLib.layerParamsFromFile(str(path / 'l_f').encode('utf-8')),
           ]

    if None in hlayers:
        raise RuntimeError("Failed to load layers")


class kaasLayer():
    def __init__(self, name, inputs, temps, outputs, N, M, O):
        self.name = name
        self.inputs = inputs
        self.temps = temps
        self.outputs = outputs

        # Layer Dimensions (in number of floats)
        self.N = N
        self.M = M
        self.O = O


    def getKernel(self, kernName, inputSpec, outputSpec=None):
        """This creates the local kernel description from this layer. inputSpec
        is a kaas.bufferSpec. outputSpec will store the output layer. If
        outputSpec is None, the output will be considered intermediate and
        ephemeral."""

        if outputSpec is not None:
            outputs = [ outputSpec ]
        else:
            outputs = self.outputs

        return kaas.kernelSpec(modelLibPath, kernName,
                64, 64,
                inputs=[inputSpec] + self.inputs, temps=self.temps, outputs=outputs)


def prepareLayer(layer, name, ctx, intermediate):
    """Prepare the layer for KaaS execution. Stores the parameters in layer in
    the libff context and returns the parameter inputs and temporaries lists.
    The caller must add input buffers. If intermediate is true, the
    output buffer is considered ephemeral."""

    layer = layer.contents
    N = layer.N
    M = layer.M
    O = layer.O

    # Upload Model. We convert to numpy arrays for pickling.
    ctx.kv.put(name+"-bias", np.ctypeslib.as_array(layer.bias, shape=(layer.N,)))
    ctx.kv.put(name + "-weight", np.ctypeslib.as_array(layer.weight, shape=(layer.N * layer.M,)))

    inputs = [
            kaas.bufferSpec(name+"-weight", N*M*4, const=True),
            kaas.bufferSpec(name+"-bias", N*4, const=True)
            ]

    temps = [ kaas.bufferSpec(name+"-preact", O*4, ephemeral=True) ]

    if intermediate:
        outputs = [ kaas.bufferSpec(name+"-out", O*4, ephemeral=True) ]
    else:
        outputs = []

    return kaasLayer(name, inputs, temps, outputs, N, M, O)


class kaasModel():
    def __init__(self, layers, ctx, kaasHandle):
        self.layers = layers
        self.ctx = ctx
        self.kaasHandle = kaasHandle


    def classify(self, inputName, outputName):
        """Invoke the model on inputName (must already be in kv store). The
        output will be available in outputName."""
        
        kerns = [ self.layers[0].getKernel('kaasLayerCForward', kaas.bufferSpec(inputName,
                    self.layers[0].N*4)),
                  self.layers[1].getKernel('kaasLayerSForward', self.layers[0].outputs[0]),
                  self.layers[2].getKernel('kaasLayerFForward', self.layers[1].outputs[0],
                    kaas.bufferSpec(outputName, self.layers[2].O*4))
                ]

        req = kaas.kaasReq(kerns)
        self.kaasHandle.Invoke(req.toDict())


def prepareModel(path, libffCtx, kaasHandle, libHandle):
    """Load the model into the KV store and prepare a template for execution."""
    hlayers = [ libHandle.cnnLib.layerParamsFromFile(str(path / 'l_c1').encode('utf-8')),
                libHandle.cnnLib.layerParamsFromFile(str(path / 'l_s1').encode('utf-8')),
                libHandle.cnnLib.layerParamsFromFile(str(path / 'l_f').encode('utf-8')) ]

    if None in hlayers:
        raise RuntimeError("Failed to load layers")

    klayers = []

    klayers.append(prepareLayer(hlayers[0], 'l_c1', libffCtx, intermediate=True))
    klayers.append(prepareLayer(hlayers[1], 'l_s1', libffCtx, intermediate=True))
    klayers.append(prepareLayer(hlayers[2], 'l_f', libffCtx, intermediate=False))

    return kaasModel(klayers, libffCtx, kaasHandle)


if __name__ == '__main__':
    ffCtx = getCtx(remote=False)
    kaasHandle = kaas.getHandle('direct', ffCtx)
    libHandle = getLibCnnHandle()

    model = prepareModel(modelDir, ffCtx, kaasHandle, libHandle)

    imgs, lbls = loadMnist(dataDir)
    
    print("Trying to predict ", lbls[0])
    ffCtx.kv.put('testInput', imgs[0])

    model.classify('testInput', 'testOutput')

    pred = ffCtx.kv.get('testOutput')
    print(np.argmax(np.frombuffer(pred.data, dtype=np.float32)))
