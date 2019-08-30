# Time Series Classification

Train a Convolutional Neural Network (CNN) on multiple samples of a classified univariate time series, then classify new time series samples with the trained network.

## Data

The current version works with datasets in the format provided by The UCR Time Series Classification Archive of univariate time series datasets.  Please see (https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/).
The Chlorine Concentration dataset from that archive is included in the data/ folder of this package.  See http://www.cs.cmu.edu/~leili/pubs/leili-thesis.pdf and http://www.timeseriesclassification.com/description.php?Dataset=ChlorineConcentration for information on this dataset.  These references are also in the readme file in the dataset folder.

I recommend downloading the archive and trying out other datasets with this classifier.

Archive reference from the python source:
@misc{UCRArchive2018,
    title = {The UCR Time Series Classification Archive},
    author = {Dau, Hoang Anh and Keogh, Eamonn and Kamgar, Kaveh and Yeh, Chin-Chia Michael and Zhu, Yan
              and Gharghabi, Shaghayegh and Ratanamahatana, Chotirat Ann and Yanping and Hu, Bing
              and Begum, Nurjahan and Bagnall, Anthony and Mueen, Abdullah and Batista, Gustavo},
    year = {2018},
    month = {October},
    note = {\url{https://www.cs.ucr.edu/~eamonn/time_series_data_2018/}}
}

## CNN Network

The Convolution Neural Network for this task is built with keras using tensorflow as the backend.
A CNN was chosen for this task based on the conclusions in the paper "Deep learning for time series classification: a review" by Hassan Ismail Fawaz, Germain Forestier, Jonathan Weber, Lhassane Idoumghar, and Pierre-Alain Muller.  Their paper can be found at https://arxiv.org/pdf/1809.04356.pdf.
Their conclusion was that the best performing network was a ResNet, but that the CNN performed nearly as well and is much simpler.  I've used here a simplified version of their CNN based on the example for "Sequence classification with 1D convolutions" found in the keras documentation at https://keras.io/getting-started/sequential-model-guide/, but with higher kernal counts based on the numbers used in the CNN portion of the network described in https://devpost.com/software/lstm-fcn.

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv1d_1 (Conv1D)            (None, 164, 128)          512
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 162, 128)          49280
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 54, 128)           0
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 52, 256)           98560
_________________________________________________________________
conv1d_4 (Conv1D)            (None, 50, 256)           196864
_________________________________________________________________
global_average_pooling1d_1 ( (None, 256)               0
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 771
=================================================================
Total params: 345,987
Trainable params: 345,987
Non-trainable params: 0
_________________________________________________________________


Ideally the full network including the LSTM portion described in that paper would be implemented here.  At present the network will train to 100% accuracy on the training data, but still achieves only about 75% accuracy on untrained test data.  Time permitting, I would experiment with model configurations and hyperparameters to improve on that.  In particular I would try the existing network with fewer convolution kernels, since these results suggest the model is over-fitting the training set.

## Training vs. Test data

When using the "train" option below, the network will be trained on the data contained in the file <dataset name>\_TRAIN.tsv.
When using the "classify" option, the network will classify (predict) on the time-series data in the file <dataset name>\_TEST.tsv.
To change the division of training and prediction examples, lines can be moved between the files.

## Usage

This project uses MLflow and has two entry-points:

### Training (on training data)
```bash
mlflow run . -e train -P <options>
```
Supported options:
  dataset=<dataset name> (default "ChlorineConcentration")
  datadir=<directory of datasets> (default "data/", and it is assumed that there will be a folder in there called <dataset name>)
  format=<dataset format> (default and only option is "UCR")
  loss=<loss function name> (default "mse"; "crossentropy" is also supported)
  optimizer=<optimizer name> (default "Adam"; "RMSprop" and "SGD" are also supported)
  epochs=<number of epochs to train> (default 300)

### Classifier prediction (on test data)
```bash
mlflow run . -e classify -P <options>
```
Supported options:
  dataset=<dataset name> (default "ChlorineConcentration")
  datadir=<directory of datasets> (default "data/", and it is assumed that there will be a folder in there called <dataset name>)
  format=<dataset format> (default and only option is "UCR")
  model=<model file name> (default "latest" will use the newest .hf5 file in the models/ directory)

## MLFlow

The "train" operation logs metrics for loss and accuracy per epoch, in mlflow.
The "classify" operation logs metrics for predicted class, actual class if provided, and the discrepancy between the two, per data entry, in mlflow.  Also the total of mismatching predictions is entered as a single metric.

Start MLflow UI from the command line after training/classifying with:
```bash
mlflow ui
```
In browser enter:
```bash
localhost:5000
```

## Todo
- Normalize input data to -1,+1 range.
- Experiment with deeper networks, particularly LSTM+FCN
- Experiment with hyperparameters such as: loss function, optimizer, kernel counts.  Possibly use a grid search.
- Add support for additional dataset formats.
- Separate out a portion of the dataset for validation on training.  Add an option for specifying validation portion.
