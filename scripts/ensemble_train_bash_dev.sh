#!/bin/bash


echo "Beginning ensemble training"

echo $CONDA_DEFAULT_ENV

python model_train.py
python lstm_model_train.py
python lstm_deep_model_train.py
python ensemble_evaluation.py
python cnn_lstm_model.py
echo 'Finished training multiple models'
