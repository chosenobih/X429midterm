import os, logging, joblib, csv, numpy as np, matplotlib.pyplot as plt, pandas as pd
from typing import Dict, Tuple
from keras.layers import Concatenate, Dot, Input, LSTM, Dense
from keras.layers import Dropout, Flatten, Activation
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.activations import softmax
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.optimizers import Adam
from math import sqrt

logging.basicConfig(filename='../logs/model_train.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

allow_pickle_flag = True

TRAIN_DATA = np.load('../data/combined_data_train.npy', allow_pickle=allow_pickle_flag)
TRAIN_LABELS = np.load("../data/scaled_yield_train.npy", allow_pickle=allow_pickle_flag)
results_dir = '../results'
YIELD_SCALER = joblib.load(results_dir + '/yield_scaler.sav')

VALIDATION_DATA = np.load('../data/combined_data_validation.npy', allow_pickle=allow_pickle_flag) 
VALIDATION_LABELS = np.load("../data/scaled_yield_validation.npy", allow_pickle=allow_pickle_flag)

class EnsembleModel:
    def __init__(self, compiled_models, yield_scaler) -> None:
        self.ensemble = compiled_models
        self.yield_scaler = yield_scaler
    



    def predict(self,X_data, batch_size):
        final_results = np.array([model.predict(X_data,batch_size = batch_size) for model in self.ensemble])
        return final_results.mean(axis=0)





    def evaluate_ensemble(self,X_data,yield_data, batch_size, dataset):
        yield_data_hat = self.ensemble.predict(X_data, batch_size = batch_size)
        yield_data_hat = YIELD_SCALER.inverse_transform(yield_data_hat)
        
        yield_data = YIELD_SCALER.inverse_transform(yield_data)
        
        metric_dict = {}  # Dictionary to save the metrics
        
        data_rmse = sqrt(mean_squared_error(yield_data, yield_data_hat))
        metric_dict ['rmse'] = data_rmse 
        logging.info('%s RMSE: %.3f' %(dataset, data_rmse))
        
        data_mae = mean_absolute_error(yield_data, yield_data_hat)
        metric_dict ['mae'] = data_mae
        logging.info('%s MAE: %.3f' %(dataset, data_mae))
        
        data_r2score = r2_score(yield_data, yield_data_hat)
        metric_dict ['r2_score'] = data_r2score
        logging.info('%s r2_score: %.3f' %(dataset, data_r2score))


        #make plots
        self.actual_pred_plot(yield_data, yield_data_hat, n_samples = 69)
        self.scatter_plot(yield_data, yield_data_hat)
        
        
        # Save metrics
        with open('%s/ensemble_metrics_%s.csv' %(results_dir, dataset), 'w', newline="") as csv_file:  
            writer = csv.writer(csv_file)
            for key, value in metric_dict.items():
                writer.writerow([key, value])    
            
        return metric_dict





    # Plot Ground Truth, Model Prediction
    def actual_pred_plot (self, y_actual: np.ndarray, y_pred: np.ndarray, n_samples: int = 60):
        """[summary]

        Args:
            y_actual (np.ndarray): [description]
            y_pred (np.ndarray): [description]
            n_samples (int, optional): [description]. Defaults to 60.
        """    
        # Shape of y_actual, y_pred: (10337, 1)
        fig, ax = plt.subplots()
        ax.plot(y_actual[ : n_samples])  # n_samples examples
        ax.plot(y_pred[ : n_samples])    # n_samples examples
        ax.legend(['Ground Truth', 'Model Prediction'], loc='upper right')
        fig.savefig('%s/lstm_attention_actual_pred_plot.png'%(results_dir))
        logging.info("Saved actual vs pred plot to disk")
        plt.close(fig)

    # Correlation Scatter Plot
    def scatter_plot (self, y_actual: np.ndarray, y_pred: np.ndarray):
        """[summary]

        Args:
            y_actual (np.ndarray): [description]
            y_pred (np.ndarray): [description]
        """    
        # Shape of y_actual, y_pred: (10337, 1)
        fig, ax = plt.subplots()
        ax.scatter(y_actual[:], y_pred[:])
        ax.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'k--', lw=4)
        fig.suptitle('Predicted Value Vs Actual Value')
        ax.set_ylabel('Predicted')
        ax.set_xlabel('Actual')
        fig.savefig('%s/lstm_attention_scatter_plot.png'%(results_dir))
        logging.info("Saved scatter plot to disk")
        plt.close(fig)






