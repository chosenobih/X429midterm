# X429midterm
ISTA429 Midterm project for Team X for MCLAS2021

Collaborators:
- Ryan Papetti
- Jordan Elliott
- Nicholas Stout
- Chosen Obih
- Jake Newton 


## Getting Started
In order to get started, you must set up a `data` directory. 

First, make sure you have anaconda on your machine. Next, make a conda env using the `environment.yml` file with `conda env create -f environment.yml`. 
If needed, activate the env with `conda activate info529midterm`. Once here, we can run the scripts to organize data.


data
├── clusterID_genotype.npy
├── inputs_others_train.npy
├── inputs_weather_train.npy
└── yield_train.npy

These are necessary for `handle_data.py` to work. This script will set up all other data for you.

Final data dir
 
data
├── clusterID_genotype.npy
├── combined_data_train.npy
├── combined_data_validation.npy
├── combined_weather_mgcluster_214_development.npy
├── inputs_others_train.npy
├── inputs_weather_train.npy
├── scaled_yield_development.npy
├── scaled_yield_train.npy
├── scaled_yield_validation.npy
└── yield_train.npy


As you can see, there is training data and validation data created for us. Also, we can run this script, minus the data splitting, to format our test data as well.
Development data is the entire "training" set before it was split

##
