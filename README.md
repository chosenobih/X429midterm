# X429midterm
ISTA429 Midterm project for Team X

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

```bash
data
├── clusterID_genotype.npy
├── inputs_others_train.npy
├── inputs_weather_train.npy
└── yield_train.npy
```
These are necessary for `handle_data.py` to work. This script will set up all other data for you.

Final data dir structure:
```
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
```

As you can see, there is training data and validation data created for us. Also, we can run this script, minus the data splitting, to format our test data as well.
Development data is the entire "training" set before it was split

### Running existing model_train scripts

If you have the environment set up on your machine, you can run the script like any other Python script. This is the same for if you are in an interactive session on the HPC.

If you are trying to train the model through a slurm script, please run `sbatch model_train.script`. To see if it's running, I like to use `squeue -u [netID]`.

Interested in writing your own script? That's okay - just make sure all your log files are pointed to a logs directory. 






## Ryan Directory set up
```bash
X429midterm
├── data
│   ├── clusterID_genotype.npy
│   ├── combined_data_train.npy
│   ├── combined_data_validation.npy
│   ├── combined_weather_mgcluster_214_development.npy
│   ├── inputs_others_train.npy
│   ├── inputs_weather_train.npy
│   ├── scaled_yield_development.npy
│   ├── scaled_yield_train.npy
│   ├── scaled_yield_validation.npy
│   └── yield_train.npy
├── environment.yml
├── LICENSE
├── logs
│   ├── model_train.log
├── README.md
├── results
│   ├── actual_pred_plot.png
│   ├── loss.csv
│   ├── metrics_Evaluation.csv
│   ├── scatter_plot.png
│   ├── val_loss.csv
│   └── yield_scaler.sav
└── scripts
    ├── conda_test.script
    ├── CropData.py
    ├── handle_data.py
    ├── handle_data.script
    ├── model_train.py
    ├── model_train.script
    └── recent_model
        ├── assets
        ├── saved_model.pb
        └── variables
            ├── variables.data-00000-of-00001
            └── variables.index

7 directories, 33 files
```