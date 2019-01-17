import tensorflow as tf
import numpy as np
import keras
from collections import defaultdict
from keras.datasets import mnist
from keras.layers import Input
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.models import Model,Sequential
from keras import optimizers
from keras.utils.generic_utils import Progbar

def LeNet_5(input_shape=(32,32,1)):
    model=Sequential()
    model.add(Conv2D(6,5,padding='valid',activation='relu',input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(16,5,padding='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(120,activation='tanh'))
    model.add(Dense(84,activation='tanh'))

    input=Input(input_shape)
    features=model(input)
    output=Dense(10,activation='softmax')(features)

    return Model(inputs=input,outputs=output)

if __name__=='__main__':
    #Training parameters
    batch_size=128
    epochs=10
    num_classes=10
    learning_rate=0.1

    #load data
    (x_train,y_train),(x_test,y_test)=mnist.load_data()
    x_train=np.expand_dims(x_train,axis=-1)
    x_test=np.expand_dims(x_test,axis=-1)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    input_shape=(28,28,1)
    lenet=LeNet_5(input_shape=input_shape)
    lenet.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        #optimizer=optimizers.SGD(lr=learning_rate),
        metrics=['accuracy']
    )
    lenet.summary()

    train_history=defaultdict(list)
    test_history=defaultdict(list)
    for epoch in range(1,epochs+1):
        print('Epoch {}/{}'.format(epoch,epochs))
        num_batches=int(np.ceil(x_train.shape[0]/float(batch_size)))
        progress_bar=Progbar(target=num_batches)

        epoch_loss=[]
        for index in range(num_batches):
            image_batch=x_train[index*batch_size:(index+1)*batch_size]
            label_batch=y_train[index*batch_size:(index+1)*batch_size]

            epoch_loss.append(lenet.train_on_batch(image_batch,label_batch))
            progress_bar.update(index+1)
        print('Test for epoch {}'.format(epoch))

        train_loss=np.mean(np.array(epoch_loss))
        train_history['trainloss'].append(train_loss)

        test_loss=lenet.evaluate(x_test,y_test)
        test_history['testloss'].append(test_loss)

        print('loss on training is {}'.format(train_loss))
        print('loss on test is {}'.format(test_loss))


