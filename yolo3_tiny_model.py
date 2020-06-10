import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
import struct

np.set_printoptions(threshold=sys.maxsize)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
tf.keras.backend.set_session(tf.Session(config=config))

argparser = argparse.ArgumentParser(
    description='yolov3 network with coco weights')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to weights file')


class WeightReader:
    def __init__(self, weight_file):
        with open(weight_file, 'rb') as w_f:
            major,    = struct.unpack('i', w_f.read(4))
            minor,    = struct.unpack('i', w_f.read(4))
            revision, = struct.unpack('i', w_f.read(4))

            if (major*10 + minor) >= 2 and major < 1000 and minor < 1000:
                w_f.read(8)
            else:
                w_f.read(4)

            binary = w_f.read()

        self.offset = 0
        self.all_weights = np.frombuffer(binary, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]

    def load_weights(self, model):
        for i in range(23):
            try:
                conv_layer = model.get_layer('conv_' + str(i))
                print("loading weights of convolution #" + str(i))

                if i not in [15, 22]:
                    norm_layer = model.get_layer('bnorm_' + str(i))

                    size = np.prod(norm_layer.get_weights()[0].shape)

                    beta  = self.read_bytes(size) # bias
                    gamma = self.read_bytes(size) # scale
                    mean  = self.read_bytes(size) # mean
                    var   = self.read_bytes(size) # variance            

                    weights = norm_layer.set_weights([gamma, beta, mean, var])  

                if len(conv_layer.get_weights()) > 1:
                    bias   = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2,3,1,0])
                    conv_layer.set_weights([kernel, bias])
                else:
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2,3,1,0])
                    conv_layer.set_weights([kernel])
            except ValueError:
                print("no convolution #" + str(i))     
    
    def reset(self):
        self.offset = 0

def _conv(inp, conv):
    x = inp
    
    x = Conv2D(filters=conv['filter'],
        kernel_size=conv['kernel'],
        strides=conv['stride'],
        padding='same',
        name='conv_'+str(conv['layer_idx']),
        use_bias=False if conv['bnorm'] else True)(x)
    if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
    if conv['activation']=='leaky': x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)

    return x

def make_yolov3_tiny_model():
    input_image = Input(shape=(None, None, 3))

    x0 = _conv(input_image, {'layer_idx': 0, 'bnorm': True, 'filter': 16, 'kernel': 3, 'stride': 1, 'activation': 'leaky'})

    x1 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same', name='maxpool_1')(x0)
    x2 = _conv(x1, {'layer_idx': 2, 'bnorm': True, 'filter': 32, 'kernel': 3, 'stride': 1, 'activation': 'leaky'})
    x3 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same', name='maxpool_3')(x2)
    x4 = _conv(x3, {'layer_idx': 4, 'bnorm': True, 'filter': 64, 'kernel': 3, 'stride': 1, 'activation': 'leaky'})
    x5 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same', name='maxpool_5')(x4)
    x6 = _conv(x5, {'layer_idx': 6, 'bnorm': True, 'filter': 128, 'kernel': 3, 'stride': 1, 'activation': 'leaky'})
    x7 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same', name='maxpool_7')(x6)
    x8 = _conv(x7, {'layer_idx': 8, 'bnorm': True, 'filter': 256, 'kernel': 3, 'stride': 1, 'activation': 'leaky'})
    x9 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same', name='maxpool_9')(x8)
    x10 = _conv(x9, {'layer_idx': 10, 'bnorm': True, 'filter': 512, 'kernel': 3, 'stride': 1, 'activation': 'leaky'})
    x11 = MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same', name='maxpool_11')(x10)
    x12 = _conv(x11, {'layer_idx': 12, 'bnorm': True, 'filter': 1024, 'kernel': 3, 'stride': 1, 'activation': 'leaky'})
    x13 = _conv(x12, {'layer_idx': 13, 'bnorm': True, 'filter': 256, 'kernel': 1, 'stride': 1, 'activation': 'leaky'})
    x14 = _conv(x13, {'layer_idx': 14, 'bnorm': True, 'filter': 512, 'kernel': 3, 'stride': 1, 'activation': 'leaky'})
    yolo_16 = _conv(x14, {'layer_idx': 15, 'bnorm': False, 'filter': 255, 'kernel': 1, 'stride': 1, 'activation': 'linear'})

    x17 = x13
    x18 = _conv(x17, {'layer_idx': 18, 'bnorm': True, 'filter': 128, 'kernel': 1, 'stride': 1, 'activation': 'leaky'})
    x19 = UpSampling2D(2, name='upsample_19')(x18)
    x20 = Concatenate()([x19, x8])
    x21 = _conv(x20, {'layer_idx': 21, 'bnorm': True, 'filter': 256, 'kernel': 3, 'stride': 1, 'activation': 'leaky'})
    yolo_23 = _conv(x21, {'layer_idx': 22, 'bnorm': False, 'filter': 255, 'kernel': 1, 'stride': 1, 'activation': 'linear'})
    
    model = Model(input_image, [yolo_16, yolo_23])    
    return model

if __name__ == '__main__':
    args = argparser.parse_args()
    weights_path = args.weights

    # make the yolov3 tiny model to predict 80 classes on COCO
    yolov3_tiny = make_yolov3_tiny_model()

    # load the weights trained on COCO into the model
    weight_reader = WeightReader(weights_path)
    weight_reader.load_weights(yolov3_tiny)
    #yolov3_tiny.summary()
    #tf.keras.utils.plot_model(yolov3_tiny, to_file=os.path.splitext(weights_path)[0]+'.png', show_shapes=True)
    yolov3_tiny.save(os.path.splitext(weights_path)[0]+'.h5')
