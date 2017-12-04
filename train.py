from __future__ import print_function

import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import CSVLogger
from model import get_model
from utils import crps, real_to_cdf, preprocess, rotation_augmentation, shift_augmentation
from datetime import datetime
import os

os.environ['THEANO_FLAGS'] = 'device=gpu'

def load_train_data():
    """
    Load training data from .npy files.
    """
    X = np.load('data/X_train.npy')
    y = np.load('data/y_train.npy')

    X = X.astype(np.float32)
    X /= 255

    seed = np.random.randint(1, 10e6)
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    return X, y


def split_data(X, y, split_ratio=0.2):
    """
    Split data into training and testing.

    :param X: X
    :param y: y
    :param split_ratio: split ratio for train and test data
    """
    split = X.shape[0] * split_ratio
    X_test = X[:split]
    y_test = y[:split, :]
    X_train = X[split:, :, :, :]
    y_train = y[split:, :]

    return X_train, y_train, X_test, y_test


def train():
    """
    Training systole and diastole models.
    """
    print('Loading and compiling models...')
    model_systole = get_model()
    model_diastole = get_model()

    print('Loading training data...')
    X, y = load_train_data()

    print('Pre-processing images...')
    X = preprocess(X)

    # split to training and test
    X_train, y_train, X_test, y_test = split_data(X, y, split_ratio=0.2)

    #Iteraties was 200
    nb_iter = 250

    epochs_per_iter = 1

    ## Batch-size was 32, ivm processing op laptop heb ik hiervan 12 gemaakt voor test #3
    batch_size = 12
    calc_crps = 1  # calculate CRPS every n-th iteration (set to 0 if CRPS estimation is not needed)

    # remember min val. losses (best iterations), used as sigmas for submission
    min_val_loss_systole = sys.float_info.max
    min_val_loss_diastole = sys.float_info.max

    print('-'*50)
    print('Training...')
    start=datetime.now()

    print('-'*50)

    for i in range(nb_iter):
        print('-'*50)
        print('Iteration {0}/{1}'.format(i + 1, nb_iter))
        print('-'*50)

        print('Augmenting images - rotations')
        X_train_aug = rotation_augmentation(X_train, 15)
        print('Augmenting images - shifts')
        X_train_aug = shift_augmentation(X_train_aug, 0.1, 0.1)

        csv_logger_diastole = CSVLogger('training_diastole.log', append=True, separator=';')
        csv_logger_systole = CSVLogger('training_systole.log', append=True, separator=';')


        print('Fitting systole model...')
        hist_systole = model_systole.fit(X_train_aug, y_train[:, 0], shuffle=True, nb_epoch=epochs_per_iter,
                                         batch_size=batch_size, validation_data=(X_test, y_test[:, 0]), callbacks=[csv_logger_systole])

        print('Fitting diastole model...')
        hist_diastole = model_diastole.fit(X_train_aug, y_train[:, 1], shuffle=True, nb_epoch=epochs_per_iter,
                                           batch_size=batch_size, validation_data=(X_test, y_test[:, 1]), callbacks=[csv_logger_diastole])

        dialoss_history = hist_diastole.history["loss"]
        sysloss_history = hist_systole.history["loss"]

        import numpy
        numpy_dialoss_history = numpy.array(dialoss_history)
        numpy_sysloss_history = numpy.array(sysloss_history)
        numpy.savetxt("dialoss_history.txt", numpy_dialoss_history, delimiter=",")
        numpy.savetxt("sysloss_history.txt", numpy_sysloss_history, delimiter=",")

        # sigmas for predicted data, actually loss function values (RMSE)
        loss_systole = hist_systole.history['loss'][-1]
        loss_diastole = hist_diastole.history['loss'][-1]
        val_loss_systole = hist_systole.history['val_loss'][-1]
        val_loss_diastole = hist_diastole.history['val_loss'][-1]

        if calc_crps > 0 and i % calc_crps == 0:
            print('Evaluating CRPS...')
            pred_systole = model_systole.predict(X_train, batch_size=batch_size, verbose=1)
            pred_diastole = model_diastole.predict(X_train, batch_size=batch_size, verbose=1)
            val_pred_systole = model_systole.predict(X_test, batch_size=batch_size, verbose=1)
            val_pred_diastole = model_diastole.predict(X_test, batch_size=batch_size, verbose=1)

            ## DEZE BOVENSTAANDE VALUES ZIJN DE RESULTATEN ##

            ## try

        ##    accuracy_systole = pred_systole - val_pred_systole

            print("Pred_diastole:")
            print(pred_diastole)

            print("Pred_systole:")
            print(pred_systole)

            print("Val_pred_sysyole:")
            print(val_pred_systole)

            print("Val_pred_diastole:")
            print(val_pred_diastole)

            # CDF for train and test data (actually a step function)
            cdf_train = real_to_cdf(np.concatenate((y_train[:, 0], y_train[:, 1])))
            cdf_test = real_to_cdf(np.concatenate((y_test[:, 0], y_test[:, 1])))

            # CDF for predicted data
            cdf_pred_systole = real_to_cdf(pred_systole, loss_systole)
            cdf_pred_diastole = real_to_cdf(pred_diastole, loss_diastole)
            cdf_val_pred_systole = real_to_cdf(val_pred_systole, val_loss_systole)
            cdf_val_pred_diastole = real_to_cdf(val_pred_diastole, val_loss_diastole)


            # evaluate CRPS on training data
            crps_train = crps(cdf_train, np.concatenate((cdf_pred_systole, cdf_pred_diastole)))
            print('CRPS(train) = {0}'.format(crps_train))

            # evaluate CRPS on test data
            crps_test = crps(cdf_test, np.concatenate((cdf_val_pred_systole, cdf_val_pred_diastole)))
            print('CRPS(test) = {0}'.format(crps_test))


            """ BEGIN PLOTTING RESULTS """

            import matplotlib.pyplot as plt
            import numpy

        #    score = model_systole.evaluate(X_test, Y_test, verbose=0)
        #    print("Score systole")
        #    print(score)




        """ EIND PLOTTING RESULT """

        print('Saving weights...')
        # save weights so they can be loaded later
        model_systole.save_weights('weights_systole.hdf5', overwrite=True)
        model_diastole.save_weights('weights_diastole.hdf5', overwrite=True)

        # for best (lowest) val losses, save weights
        if val_loss_systole < min_val_loss_systole:
            min_val_loss_systole = val_loss_systole
            model_systole.save_weights('weights_systole_best.hdf5', overwrite=True)

        if val_loss_diastole < min_val_loss_diastole:
            min_val_loss_diastole = val_loss_diastole
            model_diastole.save_weights('weights_diastole_best.hdf5', overwrite=True)

            ##Start accuracy plot for systole and diastole

            ## pyplot.plot(history.history['acc'])
            ##pyplot.show()

        # save best (lowest) val losses in file (to be later used for generating submission)
        with open('val_loss.txt', mode='w+') as f:
            f.write(str(min_val_loss_systole))
            f.write('\n')
            f.write(str(min_val_loss_diastole))

        #    print datetime.now()-start


train()
