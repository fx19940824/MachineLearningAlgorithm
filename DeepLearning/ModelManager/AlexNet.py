import tensorflow as tf
import numpy as np
from collections import defaultdict
import keras 
from keras.layers import Input
from keras import optimizers
from keras.datasets import mnist
from keras.models import Model,Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout,Dense,BatchNormalization,Flatten
from keras.utils.generic_utils import Progbar

def Alexnet(input_shape=(227,227,3),num_classes=5):
    cnn=Sequential()
    cnn.add(Conv2D(96,11,activation='relu',padding='same',strides=4,input_shape=input_shape))
    cnn.add(MaxPooling2D(pool_size=2))
    cnn.add(Conv2D(256,5,padding='same'))
    cnn.add(MaxPooling2D(pool_size=2))
    cnn.add(Conv2D(384,3,padding='same'))
    cnn.add(Conv2D(384,3,padding='same'))
    cnn.add(Conv2D(256,3,padding='same'))
    cnn.add(Flatten())
    cnn.add(Dense(4096,activation='relu'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(4096,activation='relu'))
    cnn.add(Dropout(0.5))

    input=Input(input_shape)
    features=cnn(input)
    output=Dense(num_classes,activation='softmax')(features)
    return Model(inputs=input,outputs=output)

if __name__ =='__main__':
    #Training parameters
    batch_size=128
    input_shape=(28,28,1)
    num_classes=10
    epochs=10
    learning_rate=0.1

    #load data
    (x_train,y_train),(x_test,y_test)=mnist.load_data()
    x_train=np.expand_dims(x_train,axis=-1)
    x_test=np.expand_dims(x_test,axis=-1)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    alexnet=Alexnet(input_shape=input_shape,num_classes=num_classes)
    alexnet.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.SGD(lr=learning_rate),
        metrics=['accuracy']
    )
    alexnet.summary()

    train_history=defaultdict(list)
    test_history=defaultdict(list)
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch,epochs))
        num_batches=int(np.ceil(x_train.shape[0]/float(batch_size)))
        progress_bar=Progbar(target=num_batches)
        
        epoch_loss=[]
        for index in range(num_batches):
            image_batch=x_train[index*batch_size:(index+1)*batch_size]
            label_batch=y_train[index*batch_size:(index+1)*batch_size]
            #image_batch=image_batch.resize(image_batch.shape[0],input_shape[0],input_shape[1],input_shape[2])
            epoch_loss.append(alexnet.train_on_batch(image_batch,label_batch))
            progress_bar.update(index+1)
        
        print('Test for epoch {}:'.format(epoch))
        
        train_loss=np.mean(np.array(epoch_loss))
        train_history['trainloss'].append(train_loss)
        
        test_loss=alexnet.evaluate(x_test,y_test)
        test_history['testloss'].append(test_loss)

        print('loss on training is {}'.format(train_loss))
        print('loss on test is {}'.format(test_loss))


