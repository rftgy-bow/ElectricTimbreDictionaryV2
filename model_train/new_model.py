# from https://qiita.com/shinmura0/items/858214154f889c05e4f4
#import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Add

class CBACNN:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        
    
    def __call__(self):
        pass
    
    def exportModel(input_shape, output_classes):
        # redefine target data into one hot vector
        # classes = 50
        classes = output_classes

        def cba(inputs, filters, kernel_size, strides):
            x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            return x

        # define CNN
        inputs = Input(shape=input_shape)

        x_1 = cba(inputs, filters=32, kernel_size=(1,8), strides=(1,2))
        x_1 = cba(x_1, filters=32, kernel_size=(8,1), strides=(2,1))
        x_1 = cba(x_1, filters=64, kernel_size=(1,8), strides=(1,2))
        x_1 = cba(x_1, filters=64, kernel_size=(8,1), strides=(2,1))

        x_2 = cba(inputs, filters=32, kernel_size=(1,16), strides=(1,2))
        x_2 = cba(x_2, filters=32, kernel_size=(16,1), strides=(2,1))
        x_2 = cba(x_2, filters=64, kernel_size=(1,16), strides=(1,2))
        x_2 = cba(x_2, filters=64, kernel_size=(16,1), strides=(2,1))

        x_3 = cba(inputs, filters=32, kernel_size=(1,32), strides=(1,2))
        x_3 = cba(x_3, filters=32, kernel_size=(32,1), strides=(2,1))
        x_3 = cba(x_3, filters=64, kernel_size=(1,32), strides=(1,2))
        x_3 = cba(x_3, filters=64, kernel_size=(32,1), strides=(2,1))

        x_4 = cba(inputs, filters=32, kernel_size=(1,64), strides=(1,2))
        x_4 = cba(x_4, filters=32, kernel_size=(64,1), strides=(2,1))
        x_4 = cba(x_4, filters=64, kernel_size=(1,64), strides=(1,2))
        x_4 = cba(x_4, filters=64, kernel_size=(64,1), strides=(2,1))

        x = Add()([x_1, x_2, x_3, x_4])

        x = cba(x, filters=128, kernel_size=(1,16), strides=(1,2))
        x = cba(x, filters=128, kernel_size=(16,1), strides=(2,1))

        x = GlobalAveragePooling2D()(x) # 128dim
        
        # Bypass of feature layer
        feature_vector = x
        
        x = Dense(classes)(x)
        final = Activation("softmax")(x)

        model = Model(
            inputs = inputs,
            outputs = [final, feature_vector])
        
        model.summary()
        
        # The output of model.predict() is now a list 
        # containing the classification score and feature_vector.
        
        #model.compile(
        #    loss=['categorical_crossentropy', None],# train output "final x" only
        #    optimizer='adam',
        #    metrics=['accuracy'])
        
        return model
        

if __name__ == '__main__':
    pass
    




