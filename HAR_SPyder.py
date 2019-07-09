# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 00:27:08 2019

@author: DELL
"""
#pip install hyperopt
import hyperas
import hyperopt
import pickle

def data():
  '''
  Data providing function:
  This function is separated from model() so that hyperopt
  won't reload data for each evaluation run.
  '''
  pickle_in = open("C:\\Users\\DELL\\Machine Learning\\Deep Learning Assignment\\21 Human Activity Detection\\HAR\\X_train.pickle","rb")
  X_train = pickle.load(pickle_in)
  pickle_in.close()

  pickle_in = open("C:\\Users\\DELL\\Machine Learning\\Deep Learning Assignment\\21 Human Activity Detection\\HAR\\X_test.pickle","rb")
  X_test = pickle.load(pickle_in)
  pickle_in.close()

  pickle_in = open("C:\\Users\\DELL\\Machine Learning\\Deep Learning Assignment\\21 Human Activity Detection\\HAR\\Y_train.pickle","rb")
  Y_train = pickle.load(pickle_in)
  pickle_in.close()

  pickle_in = open("C:\\Users\\DELL\\Machine Learning\\Deep Learning Assignment\\21 Human Activity Detection\\HAR\\Y_test.pickle","rb")
  Y_test = pickle.load(pickle_in)
  pickle_in.close()
  X_train = X_train.reshape(len(X_train), len(X_train[0]), len(X_train[0][0]),1)
  X_test = X_test.reshape(len(X_test),len(X_test[0]),len(X_test[0][0]),1)
  return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = data()

def _count_classes(y):
    return len(set([tuple(category) for category in y]))

timesteps = len(X_train[0])
input_dim = len(X_train[0][0])
n_classes = _count_classes(Y_train)

print(timesteps)
print(input_dim)
print(len(X_train))
print(n_classes)

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

# Initiliazing the sequential model
def model(X_train, Y_train, X_test, Y_test):
    from keras.models import Sequential
    import keras
    from keras.layers import LSTM, Conv1D, TimeDistributed, MaxPooling1D
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.layers.normalization import BatchNormalization
    model = Sequential()
    # Configuring the parameters
    model.add(TimeDistributed(Conv1D(filters = {{choice([64,128, 256, 512, 1024])}}, kernel_size={{choice([2,3,4,5])}},
                     activation={{choice(['relu', 'sigmoid'])}},padding = 'same'), input_shape = (128,9,1)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    # model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Dropout({{uniform(0, 1)}})))

    model.add(TimeDistributed(Conv1D(filters = {{choice([64,128, 256, 512, 1024])}}, kernel_size={{choice([2,3,4,5])}},
                     activation={{choice(['relu', 'sigmoid'])}},padding = 'same')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Dropout({{uniform(0, 1)}})))

    if {{choice(['two', 'three'])}} == 'three':
        model.add(TimeDistributed(Conv1D(filters = {{choice([64,128, 256, 512, 1024])}}, kernel_size={{choice([2,3,4,5])}},
                     activation={{choice(['relu', 'sigmoid'])}},padding = 'same')))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Dropout({{uniform(0, 1)}})))
    
    
        model.add(TimeDistributed(Conv1D(filters = {{choice([64,128, 256, 512, 1024])}}, kernel_size={{choice([2,3,4,5])}},
                     activation={{choice(['relu', 'sigmoid'])}},padding = 'same')))
#         model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    # model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Dropout({{uniform(0, 1)}})))
        
    if {{choice(['three','four'])}} == 'four':
      
        model.add(TimeDistributed(Conv1D(filters = {{choice([64,128, 256, 512, 1024])}}, kernel_size={{choice([2,3,4,5])}},
                     activation={{choice(['relu', 'sigmoid'])}},padding = 'same')))
#         model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Dropout({{uniform(0, 1)}})))
    
    
        model.add(TimeDistributed(Conv1D(filters = {{choice([64,128, 256, 512, 1024])}}, kernel_size={{choice([2,3,4,5])}},
                     activation={{choice(['relu', 'sigmoid'])}},padding = 'same')))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    # model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Dropout({{uniform(0, 1)}})))

    model.add(TimeDistributed(Flatten()))
    model.add(LSTM({{choice([64,128, 256, 512, 1024])}}, return_sequences=True))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(LSTM({{choice([64,128, 256, 512, 1024])}},return_sequences=True))
    # Adding a dropout layer
    model.add(Dropout({{uniform(0, 1)}}))
    # Adding a dropout layer
    model.add(LSTM({{choice([64,128, 256, 512, 1024])}},return_sequences=True))
    model.add(Dropout({{uniform(0, 1)}}))
        
    model.add(Flatten())
    
    model.add(Dense({{choice([64,128, 256, 512, 1024])}}, activation={{choice(['softmax', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))
    
    # Adding a dense output layer with sigmoid activation
    model.add(Dense(6, activation='softmax'))

    adam = keras.optimizers.Adam(lr={{choice([10**-3, 10**-2, 10**-1])}})
    rmsprop = keras.optimizers.RMSprop(lr={{choice([10**-3, 10**-2, 10**-1])}})
    sgd = keras.optimizers.SGD(lr={{choice([10**-3, 10**-2, 10**-1])}})

    choiceval = {{choice(['adam', 'sgd', 'rmsprop'])}}
    if choiceval == 'adam':
        optim = adam
    elif choiceval == 'rmsprop':
        optim = rmsprop
    else:
        optim = sgd

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer=optim)
    model.fit(X_train, Y_train,
              batch_size={{choice([16,64,128,256,512])}},
              nb_epoch=30,
              verbose=2,
              validation_data=(X_test, Y_test))
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
  best_run, best_model = optim.minimize(model=model,
                                        data=data,
                                        algo=tpe.suggest,
                                        max_evals=30,
                                        notebook_name='HAR_SPyder',
                                        trials=Trials())
  
  print(best_run)
  print(best_model)