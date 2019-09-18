# Imports
import sys
import argparse
import glob
import os
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import backend as K
import tensorflow as tf
from tensorflow.python.util import deprecation
import mlflow
import mlflow.keras

# Acknowledgement for the dataset included in this package:
# @misc{UCRArchive2018,
#     title = {The UCR Time Series Classification Archive},
#     author = {Dau, Hoang Anh and Keogh, Eamonn and Kamgar, Kaveh and Yeh, Chin-Chia Michael and Zhu, Yan
#               and Gharghabi, Shaghayegh and Ratanamahatana, Chotirat Ann and Yanping and Hu, Bing
#               and Begum, Nurjahan and Bagnall, Anthony and Mueen, Abdullah and Batista, Gustavo},
#     year = {2018},
#     month = {October},
#     note = {\url{https://www.cs.ucr.edu/~eamonn/time_series_data_2018/}}
# }

# Silence tensorflow warnings and deprecation messages
#import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Constants
loss_func_list = ['mse', 'crossentropy']
optimizer_list = ['Adam', 'RMSprop', 'SGD']
trainingset_ext = "_TRAIN.tsv"
testset_ext = "_TEST.tsv"
model_dir = "models"
now = datetime.datetime.utcnow()
raw_now = datetime.datetime.utcnow()
now = raw_now.strftime("%Y-%m-%d__%H-%M-%S")

# Define the command line interface and return the parsed arguments
def parse_command_args():
    parser = argparse.ArgumentParser(description='Train or predict time series classes.')
    parser.add_argument('--op', choices = ['train', 'classify'], default='train', help="Operation to perform", type= str)
    parser.add_argument('--dataset', default='ChlorineConcentration', help="Name of dataset to use (ChlorineConcentration)", type=str)
    parser.add_argument('--datadir', default='data', help="Dataset directory (data)", type=str)
    parser.add_argument('--format', choices=['UCR'], default='UCR', help="Dataset files format (UCR)", type= str)
    parser.add_argument('--loss', choices=loss_func_list, default=loss_func_list[0],
            help="Loss function, train only (%s)" % loss_func_list[0], type=str)
    parser.add_argument('--optimizer', choices=optimizer_list, default=optimizer_list[0],
            help="Optimizer, train only (%s)" % optimizer_list[0], type=str)
    parser.add_argument('--epochs', default=300, help="Number of epochs to train, train only (300)", type=int)
    parser.add_argument('--model', default='latest', help="Pre-trained model filename in 'models' directory, classify only (latest)", type= str)
    return parser.parse_args()

# Load dataset named base_name that is in the UCR Time Series Classification Archive format
# see https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/
def load_UCR_dataset(directory, base_name):
    train_filename = os.path.join(directory, base_name, base_name + trainingset_ext)
    test_filename = os.path.join(directory, base_name, base_name + testset_ext)
    print ("Loading trainind and test data from:\n" + train_filename + "\n" + test_filename)
    train_raw = pd.read_csv(train_filename, sep='\t', header=None)
    test_raw = pd.read_csv(test_filename, sep='\t', header=None)

    train_yset = train_raw[0]
    train_Xset = train_raw.iloc[:,1:]
    test_yset = test_raw[0]
    test_Xset = test_raw.iloc[:,1:]
    train_yset -= train_yset.min()
    train_Y = keras.utils.to_categorical(train_yset)
    test_yset -= test_yset.min()
    test_Y = keras.utils.to_categorical(test_yset)

    CNN_train_Xset = train_Xset.to_numpy()
    CNN_train_Xset3d = CNN_train_Xset.reshape((CNN_train_Xset.shape[0], CNN_train_Xset.shape[1], 1))
    CNN_test_Xset = test_Xset.to_numpy()
    CNN_test_Xset3d = CNN_test_Xset.reshape((CNN_test_Xset.shape[0], CNN_test_Xset.shape[1], 1))
    print("Training shape=" + str(CNN_train_Xset3d.shape) + "\nTest shape=" + str(CNN_test_Xset3d.shape))
    return CNN_train_Xset3d, train_Y, CNN_test_Xset3d, test_Y, test_yset

# Define and compile a CNN model with specified loss functioni and optimizer.
# Uses "accuracy" metric.
def compiled_CNN_model(loss_function, optimizer, train_X, train_y):
    CNN_model = keras.models.Sequential()
    CNN_model.add(keras.layers.Conv1D(128, 3, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
    CNN_model.add(keras.layers.Conv1D(128, 3, activation='relu'))
    CNN_model.add(keras.layers.MaxPooling1D(3))
    CNN_model.add(keras.layers.Conv1D(256, 3, activation='relu'))
    CNN_model.add(keras.layers.Conv1D(256, 3, activation='relu'))
    CNN_model.add(keras.layers.GlobalAveragePooling1D())
    CNN_model.add(keras.layers.Dropout(0.5))
    CNN_model.add(keras.layers.Dense(train_y.shape[1], activation='sigmoid'))
    if loss_function == 'crossentropy':
        if train_y.shape[1] == 1:
            loss_function = 'binary_crossentropy'
        else:
            loss_function = 'categorical_crossentropy'
    CNN_model.compile(loss=loss_function,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print("Model compiled")
    return CNN_model

    ## These lines are an example of building the model with the keras API:
    # test_input = keras.layers.Input(shape=(CNN_train_Xset3d.shape[1], CNN_train_Xset3d.shape[2]))
    # test_output = keras.layers.Conv1D(128, 3, activation='relu')(test_input)
    # test_output = keras.layers.Conv1D(128, 3, activation='relu')(test_output)
    # test_output = keras.layers.MaxPooling1D(3)(test_output)
    # ...etc.
    # test_model = keras.models.Model(inputs=test_input, outputs=test_output)

if __name__ == '__main__':
    # Clear old session
    # TODO: Check that --help wasn't used on command line first?
    try:
        K.clear_session()
    except:
        # Only tensorflow backend has a clear_session.
        pass
    # Initialize random seed?
    command_args = parse_command_args()
    train_X, train_y, test_X, test_y, test_y_original = load_UCR_dataset(
            command_args.datadir, command_args.dataset
    )
    if command_args.op == 'train':
        print ("Training " + command_args.dataset)
        CNN_model = compiled_CNN_model(
                command_args.loss, command_args.optimizer,
                train_X, train_y
        )
        train_result = CNN_model.fit(train_X, train_y, epochs=command_args.epochs)
        try:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
        except OSError:
            print ('Error: Creating directory %s' % model_dir)
        model_filepath = os.path.join(model_dir, '%s_%s_%d.hf5' % (command_args.dataset, now, command_args.epochs))
        CNN_model.save(filepath=model_filepath)
        eval_result = CNN_model.evaluate(test_X, test_y)
        print ("Test result [loss, accuracy]:\n%s" % str(eval_result))

        with mlflow.start_run(run_name='Train %s' % command_args.dataset):
            run_uuid = mlflow.active_run().info.run_uuid
            print("MLflow Run ID: %s" % run_uuid)
            mlflow.keras.log_model(CNN_model, "models")
            mlflow.log_artifact(model_filepath, model_dir)
#            mlflow.log_artifact(image_dir +  city + '_Loss_Diag.png', "images")
#            mlflow.log_artifact(image_dir +  city + '_Daily_Temp_Predicted.png', "images")
            mlflow.log_param('Operation', command_args.op)
            mlflow.log_param('Date__Time', now)
            mlflow.log_param('Dataset Name', command_args.dataset)
            mlflow.log_param('Training Epochs', command_args.epochs)
            for i, loss in enumerate(train_result.history['loss'], start=1):
                mlflow.log_metric('Training Loss', loss, step=i)
            for i, acc in enumerate(train_result.history['acc'], start=1):
                mlflow.log_metric('Training Accuracy', acc, step=i)

    elif command_args.op == 'classify':
        if command_args.model == 'latest':
            model_files_list = glob.glob(os.path.join(model_dir, '%s_*.hf5' % command_args.dataset))
            model_filepath = max(model_files_list, key=os.path.getctime)
        else:
            model_filepath = os.path.join(model_dir, command_args.model)

        if os.path.isfile(model_filepath) == True:
            print("Loading model %s\n" % model_filepath)
            CNN_model = keras.models.load_model(model_filepath)
            print("Model loaded:\n" + str(CNN_model.summary()))
            pred_classes_raw = CNN_model.predict(test_X)
            pred_classes = CNN_model.predict_classes(test_X)
            print ("prediction shape=%s" % str(pred_classes.shape))

            with mlflow.start_run(run_name='Predict %s' % command_args.dataset):
                run_uuid = mlflow.active_run().info.run_uuid
                print("MLflow Run ID: %s" % run_uuid)
                mlflow.keras.log_model(CNN_model, "models")
                mlflow.log_artifact(model_filepath, model_dir)
                mlflow.log_param('Operation', command_args.op)
                mlflow.log_param('Date__Time', now)
                mlflow.log_param('Dataset Name', command_args.dataset)
                pred_classes += 1
                for i, y in enumerate(pred_classes, start=1):
                    mlflow.log_metric('Predicted class', y, step=i)
                test_y_original += 1
                for i, y in enumerate(test_y_original, start=1):
                    mlflow.log_metric('Actual class', y, step=i)
                pred_classes -= test_y_original
                for i, y in enumerate(pred_classes, start=1):
                    mlflow.log_metric('Discrepancy', y, step=i)
                error_count = np.count_nonzero(pred_classes)
                mlflow.log_metric('Error Count', error_count)
    else:
        print("Unsupported operation '%s'" % command_args.op)
