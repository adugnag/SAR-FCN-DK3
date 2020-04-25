"""
@author: Adugna Mullissa
@software: Spyder
@file: FCN.py
@time: 2020/02/01 16:54
@desc: This function implements a fully convolutional network with dilated kernels 
to train a model for a semantic segmentation task.
"""
from keras import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Reshape

def FCN_DK3(nClasses, input_height, input_width):

    img_input = Input(shape=( input_height, input_width,10))


    conv_1 = Conv2D(16, (3, 3), dilation_rate=(1,1), padding="same")(img_input)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    
    conv_2 = Conv2D(32, (3, 3), dilation_rate=(2,2), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)
    
    conv_3 = Conv2D(32, (3, 3) , dilation_rate=(3,3), padding="same")(conv_2)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)

    conv_4 = Conv2D(32, (3, 3) , dilation_rate=(3,3), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    conv_5 = Conv2D(32, (3, 3) , dilation_rate=(2,2), padding="same")(conv_4)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)

    conv_6 = Conv2D(32, (3, 3) , dilation_rate=(1,1), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    
    conv_7 = Conv2D(64, (3, 3), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)
    
    conv_8 = Conv2D(nClasses, (1, 1), padding="same")(conv_7)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)

    conv_9 = Reshape((-1, nClasses))(conv_8)
    y = Activation("softmax")(conv_9)

    model=Model(inputs=img_input,outputs=y)
    
    return model


if __name__ == '__main__':
    m = FCN_DK3(13, 125, 125)
    # print(m.get_weights()[2]) 
    from keras.utils import plot_model
    plot_model(m, show_shapes=True, to_file='model_FCN.png')
    print(len(m.layers))
    m.summary()
