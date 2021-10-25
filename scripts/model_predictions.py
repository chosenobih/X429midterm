import logging, joblib, numpy as np
from typing import Dict, Tuple
from keras.models import load_model

from ensemble_evaluation import EnsembleModel

logging.basicConfig(filename='../logs/test_model_predictions.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

allow_pickle_flag = True

TEST_DATA = np.load('../data/combined_data_test.npy', allow_pickle=allow_pickle_flag)
results_dir = '../results'
YIELD_SCALER = joblib.load(results_dir + '/yield_scaler.sav')



OG_MODEL = load_model('../results/lstm_attention_recent_model')
LSTM_MODEL = load_model('../results/lstm_recent_model')
LSTM_DEEP_MODEL = load_model('../results/deep_lstm_recent_model')

ryan_model_weights = np.array([0.2,0.4,0.4])
ryan_models = [OG_MODEL, LSTM_MODEL, LSTM_DEEP_MODEL]
ryan_ensemble = EnsembleModel(ryan_models, YIELD_SCALER, ryan_model_weights)
test_predictions = ryan_ensemble.predict(TEST_DATA, batch_size=512)
ryan_ensemble.write_predictions(test_predictions)








