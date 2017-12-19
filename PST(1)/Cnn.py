from __future__ import print_function
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from LoadCK import buildDataSetCK
from LoadCK import ShuffleDataSet
from sklearn.metrics import confusion_matrix
from ConfusionMatrixBuild import plot_confusion_matrix
import itertools
import matplotlib.pyplot as plt
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

batch_size = 32
num_classes = 7
epochs = 35
k = 5

# input image dimensions
img_rows, img_cols = 32, 32

DataSet = buildDataSetCK()

k_folders = ShuffleDataSet(DataSet,k,num_classes)

result=[]
y=[]
mean = 0.

# Load dataset
for n in range(k):
    trainSet=[]
    for i in range (k):
        if (i == n):
            testSet = k_folders[n]
        else:
            trainSet.append(k_folders[i])     

    x_train=[]
    y_train=[]
    for j in range (len(trainSet)):
        for i in range (len(trainSet[j])):
            x_train.append(trainSet[j][i][0])
            y_train.append(trainSet[j][i][1])

    x_test=[]
    y_test=[]

    for i in range (len(testSet)):
        x_test.append(testSet[i][0])
        y_test.append(testSet[i][1])

    for i in range (len(x_train)):
        x_train[i]=np.array(x_train[i])
    for i in range(len(x_test)):
        x_test[i]=np.array(x_test[i])

    x_train=np.array(x_train)
    x_test=np.array(x_test)
    print(x_train.shape)
    
    y.extend(y_test)

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))

    #y=model.predict(x_test)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    mean = mean + score[1]
    
    #y_pred = model.predict(x_test)
    y_pred = model.predict_classes(x_test)
    result.extend(y_pred)

mean = mean/k

print('Total Test accuracy:', mean)
print(result)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y, result)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[0, 1, 2, 3, 4, 5, 6], normalize=True,
                      title='Normalized confusion matrix')

plt.show()