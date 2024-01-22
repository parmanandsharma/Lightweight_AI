# This is a Python implementation of deep learning models based on
# Lightweight bottleneck narrowing with attention (LWBNA) reported by
# Sharma etal. Scientific Reports, volume 12, Article number: 8508 (2022)
# The Architecture of a lightweight Deep Learning model (LWBNA_Unet) has been patented, and any commercial use require permissions.

import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow import keras
#----Loss_functions ----
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

def dice_p_bce(in_gt, in_pred):
    return 1 - dice_coef(in_gt, in_pred)

#Attention-Block (AB)
def Attention(input, filter):
    # Squeeze
    x = layers.GlobalAveragePooling2D()(input)
    # Excitation
    x = layers.Dense(filter,kernel_initializer="he_normal")(x)
    x=layers.Activation(activation='relu')(x)
    x = layers.Activation(activation='sigmoid')(x)
    x = layers.Multiply()([input, x])
    return (x)

# Two CONV-blocks
def Convolution_block(input, filter=64, kernel=3,AB_layer=False):
    x = layers.Conv2D(filter, kernel, kernel_initializer="he_normal",padding='same')(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filter, kernel, kernel_initializer="he_normal",padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if AB_layer==True:
        x = Attention(x,filter)
    return(x)

# Encoder block
def Encoder(input_data,depth_layer=4, f=16,dropout=0.3,AB_block=True, fixed_filter=False):
    skip_layers=[] # list to store input of convolution is each layer depth
    for i in range(0,depth_layer,1):
        if fixed_filter==True:
            filter_multiplier = 1
        else:
            filter_multiplier=2**i
        layer_filter=f*filter_multiplier # number of filters/channels with depth
        conv = Convolution_block(input_data, filter=layer_filter, kernel=3, AB_layer=AB_block)
        skip_layers.append(conv) #store the convolution output
        output = layers.MaxPooling2D(2)(conv) # maxpooling to reduce the dimension
        encoder_output = layers.Dropout(dropout)(output) # randomly drop
        input_data = encoder_output
    return encoder_output, skip_layers, layer_filter

# Decoder block
def Decoder(input,skip_layers, depth_layer = 4,f=16,dropout=0.3,AB_block=True ):
    output=input
    for i in range(0,depth_layer,1):
        up = layers.UpSampling2D(2)(output) # upsampling
        up = layers.add([up, skip_layers[len(skip_layers)-1-i]])  # skip layer adding from encoder
        up = layers.Dropout(dropout)(up) # random drop out
        output = Convolution_block(up, filter=f, kernel=3, AB_layer=AB_block)  # convolution two times
    return output

# Mid block
def Mid_block(input,filter,depth_layer=4,r_factor=2, segmentation=False ):
    x=input;f_reduction=1
    output=[]
    for i in range(0, depth_layer):
        x = layers.Conv2D(filter/f_reduction, 3, activation='relu', kernel_initializer="he_normal", padding='same')(x)
        if i==0:
            xe1=x
        x = Attention(x, filter=int(filter/f_reduction))
        f_reduction = f_reduction*r_factor
    output.append(x)
    if segmentation==True:
        x = layers.Conv2D(filter, 3, activation='relu', kernel_initializer="he_normal", padding='same')(x)  # best128
        x = layers.add([x, xe1]) # skip connection
        output.append(x)
    return output

# LWBNA_unet model for segmentation tasks
def LWBNA_unet_segmentation_model(img_shape=(320,320,3),depth_layer=4,f=128,dropout=0.3,AB_block=True):
    _,_,output_channel = img_shape
    inputs = keras.Input(shape=img_shape)
    encoder_output, skip_layers,encoder_last_layer_filter= Encoder(inputs, depth_layer, f, dropout, AB_block, fixed_filter=True)
    bottleneck = Mid_block(encoder_output, encoder_last_layer_filter, depth_layer, r_factor=2,segmentation=True)
    bottleneck = Convolution_block(bottleneck[1], encoder_last_layer_filter, kernel=3, AB_layer=AB_block)
    decoder_output = Decoder(bottleneck,skip_layers,depth_layer,encoder_last_layer_filter,dropout,AB_block)
    output = layers.Conv2D(output_channel, 3, activation='sigmoid', padding='same',name="Seg_out")(decoder_output)
    model = keras.Model(inputs, output, name = 'LWBNA_unet_seg_model')
    model.summary()
    return model
