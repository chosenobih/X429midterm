import os, logging, joblib, csv, numpy as np, matplotlib.pyplot as plt, pandas as pd
from typing import Dict
from keras.layers import  LSTM, Dense, Conv1D
from keras.layers import Dropout
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.optimizers import Adam
from math import sqrt

# if running locally (Ryan)
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
sess.as_default()

logging.basicConfig(filename='../logs/cnn_lstm_model_train.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

allow_pickle_flag = True

TRAIN_DATA = np.load('../data/combined_data_train.npy', allow_pickle=allow_pickle_flag)
TRAIN_LABELS = np.load("../data/scaled_yield_train.npy", allow_pickle=allow_pickle_flag)
results_dir = '../results'
YIELD_SCALER = joblib.load(results_dir + '/yield_scaler.sav')

VALIDATION_DATA = np.load('../data/combined_data_validation.npy', allow_pickle=allow_pickle_flag) 
VALIDATION_LABELS = np.load("../data/scaled_yield_validation.npy", allow_pickle=allow_pickle_flag)


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"


### CHANGE THESE BELOW FOR MESSING WITH MODEL
 
h_s = 128   # {32, 64, 96, 128, 256}
dropout = 0.25  # {0.05, 0.1, 0.2, 0.4, 0.4, 0.5}
batch_size = 512 # paper said this didn't matter much. Dont change  
epochs = 50   # Try not to go above 50 - it will stop when it starts to overfit
lr_rate = 0.001   # (0.001, 3e-4, 5e-4)
### DO NOT CHANGE BELOW

def model(Tx: int, var_ts: int, h_s: int, dropout: float) -> Model:
    """[summary]

    Args:
        Tx (int): [description]
        var_ts (int): [description]
        h_s (int): [description]
        dropout (float): [description]

    Returns:
        Tuple: [description]
    """    

    # Tx : Number of input timesteps
    # var_ts: Number of input variables
    # h_s: Hidden State Dimension

    dropout_layer = Dropout(dropout)

    first_conv_layer = Conv1D(filters=16, kernel_size=5, strides=4, padding="causal",activation="relu",input_shape=(Tx, var_ts))

    first_lstm_layer = LSTM(h_s, return_sequences=True)
    second_lstm_layer = LSTM(int(2/5 * h_s), return_sequences=False)

    output_layer = Dense(1)
    pred_model = Sequential([first_conv_layer,first_lstm_layer, dropout_layer, second_lstm_layer,output_layer])
        
    return pred_model



# Model Summary
pred_model = model(Tx = 214, var_ts = TRAIN_DATA.shape[2], h_s = h_s, dropout = dropout)
pred_model.summary()
callback_lists = [EarlyStopping(monitor = 'val_loss', patience=3)]

# Train Model
pred_model.compile(loss='mean_squared_error', optimizer = Adam(lr_rate)) 


hist = pred_model.fit (TRAIN_DATA, TRAIN_LABELS,
                  batch_size = batch_size,
                  epochs = epochs,
                  callbacks = callback_lists,
                  verbose = 1,
                  shuffle = True,
                  validation_data=(VALIDATION_DATA,VALIDATION_LABELS))

pred_model.save(f'{results_dir}/cnn_lstm_recent_model')

# Plot
loss = hist.history['loss']
val_loss = hist.history['val_loss']


def plot_loss(loss: np.ndarray, val_loss: np.ndarray):
    """[summary]

    Args:
        loss (np.ndarray): [description]
        val_loss (np.ndarray): [description]
    """    
    fig,ax = plt.subplots()
    ax.plot(loss)
    ax.plot(val_loss)
    fig.suptitle('Model Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(['Training Set', 'Validation Set'], loc='upper right')
    fig.savefig('%s/CNN_LSTM_loss_plot.png'%(results_dir))
    logging.info("Saved loss plot to disk")
    plt.close(fig)


# Save Data
loss = pd.DataFrame(loss).to_csv('%s/cnn_lstm_loss.csv'%(results_dir))    # Not in original scale 
val_loss = pd.DataFrame(val_loss).to_csv('%s/cnn_lstm_val_loss.csv'%(results_dir))  # Not in original scale
# plot_loss(loss,val_loss)



# Plot Ground Truth, Model Prediction
def actual_pred_plot (y_actual: np.ndarray, y_pred: np.ndarray, n_samples: int = 60):
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
    fig.savefig('%s/CNN_LSTM_actual_pred_plot.png'%(results_dir))
    logging.info("Saved actual vs pred plot to disk")
    plt.close(fig)

# Correlation Scatter Plot
def scatter_plot (y_actual: np.ndarray, y_pred: np.ndarray):
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
    fig_file = '%s/cnn_lstm_scatter_plot.png'%(results_dir)
    logging.info('Saving to fig file {}'.format(fig_file))
    fig.savefig(fig_file)
    logging.info("Saved scatter plot to disk")
    plt.close(fig)




 # Evaluate Model
def evaluate_model (x_data: np.ndarray, yield_data: np.ndarray, dataset: str) -> Dict:
    """[summary]

    Args:
        x_data (np.ndarray): [description]
        yield_data (np.ndarray): [description]
        dataset (str): [description]

    Returns:
        Dict: [description]
    """    
    yield_data_hat = pred_model.predict(x_data, batch_size = batch_size)
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
    actual_pred_plot(yield_data, yield_data_hat, n_samples = 69)
    scatter_plot(yield_data, yield_data_hat)
    
       
    # Save metrics
    logging.info('Getting ready to save to file')
    output_file = '%s/cnn_lstm_metrics_%s.csv' %(results_dir,dataset)
    with open(output_file, 'w', newline="") as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in metric_dict.items():
            writer.writerow([key, value])    
        
    return metric_dict


evaluate_model(VALIDATION_DATA,VALIDATION_LABELS,'Evaluation')
