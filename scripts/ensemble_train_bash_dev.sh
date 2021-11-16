#!/bin/bash


echo "Beginning ensemble training"

echo $CONDA_DEFAULT_ENV

ls -l ../..

ls -l /analysis

cd ../data


### Downloads data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1DoyextA0q4mxumMAhBvqZbfZriIM9A-Y' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1DoyextA0q4mxumMAhBvqZbfZriIM9A-Y" -O "Dataset_Competition_Zip_File.zip" && rm -rf /tmp/cookies.txt

unzip Dataset_Competition_Zip_File.zip
rm Dataset_Competition_Zip_File.zip

cp Dataset_Competition/clusterID_genotype.npy clusterID_genotype.npy 

cp Dataset_Competition/Training/* . 

ls -l

rm -rf Dataset_Competition

cd ../scripts

python handle_data.py

echo "Successfully handled data"

python model_train.py
python lstm_model_train.py
python lstm_deep_model_train.py
python ensemble_evaluation.py
python cnn_lstm_model.py
echo 'Finished training multiple models'
