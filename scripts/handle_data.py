'''
Ryan Papetti
Team X
INFO 429/529 Midterm
Fall 2021

handle_data.py contains the methods to convert the standard competition datasets into a meaninful one for training with any mode. Offers utilities to save and split data as well
'''


from typing import Tuple
import numpy as np, pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib, logging


# logging.basicConfig(filename='../logs/data_handling.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
logging.basicConfig( level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(weather_path:str,other_path:str, cluster_id_path:str, yield_path:str) -> Tuple:
    """
    Loads all relevant data for processing via inputted filenames as specified

    Args:
        weather_path (str): path to weather data
        other_path (str): path to other data
        cluster_id_path (str): path to clusterID data
        yield_path (str): path to yield data

    Returns:
        Tuple: tuple of arrays where each array is loaded from each path (in order)
    """    
    return np.load(weather_path), np.load(other_path), np.load(cluster_id_path), np.load(yield_path)



def clean_other_data(other_data: np.ndarray,cluster_id_data: np.ndarray, data_path: str = '../data/', data_stage: str = 'development') -> np.ndarray:
    """
    Takes other data and cluster_id_data. Assigns the proper genotype_id and returns all the data as a onehotencoded matrix. As of now only the first two variables are chosen. This will likely change. 

    Args:
        other_data (np.ndarray): pre-loaded
        cluster_id_data (np.ndarray): pre-loaded

    Returns:
        np.ndarray: OneHotEncoded matrix
    """    
    other_df = pd.DataFrame(other_data)
    logging.info('Put data into dataframe')
    other_df.columns = ['Maturity Group', 'Genotype ID', 'State', 'Year', 'Location']
    for col in ['Maturity Group', 'Genotype ID', 'Year', 'Location']:
        other_df[col] = other_df[col].astype(np.float32).astype(int)
    logging.info('Converted appropriate columns into ints')

    other_df['Genotype ID'] -= 1 #to match indexing for cluster_id_data
    cluster_id_mapper = lambda genotype_id: cluster_id_data[genotype_id]
    state_cleaner = lambda state: ''.join([char for char in state if char.isalpha()])
    other_df['Genotype ID'] = other_df['Genotype ID'].apply(cluster_id_mapper)
    other_df['State'] = other_df['State'].apply(state_cleaner)
    other_df = other_df[other_df['Location'] != 162]
    indices_to_pass = list(other_df.index)
    logging.info('Applied custom functions to clean dataframe')
    #now one hot encode all data 
    one_hot_encoded_data = OneHotEncoder().fit_transform(other_df).toarray().astype('float32')
    logging.info('One hot encoded all data')
    logging.info(f'SHAPE: {one_hot_encoded_data.shape}')
    assert one_hot_encoded_data.shape[1] == 229
    if data_path:
        with open(f'{data_path + data_stage} ohe_other_data.npy','wb') as writer:
            np.save(writer, one_hot_encoded_data)

    return one_hot_encoded_data, indices_to_pass




def scale_weather_data(weather_data: np.ndarray, optional_indices: list = []) -> np.ndarray:
    """
    
    Takes Weather data, reshapes it, and returns a scaled version

    Args:
        weather_data (np.ndarray): pre-loaded

    Returns:
        np.ndarray: weather_data_scaled
    """    
    
    scaler_x = MinMaxScaler(feature_range=(-1, 1))

    x_train_reshaped = weather_data.reshape((weather_data.shape[0], weather_data.shape[1] * weather_data.shape[2]))

    if optional_indices:
        x_train_reshaped = x_train_reshaped[optional_indices]

    # Scaling Coefficients calculated from the training dataset
    weather_data_scaled = scaler_x.fit_transform(x_train_reshaped)   

    desired_shape = (weather_data_scaled.shape[0], weather_data.shape[1], weather_data.shape[2])
    weather_data_scaled = weather_data_scaled.reshape(desired_shape)

    logging.info('Weather data scaled')

    return weather_data_scaled



def scale_yield_data(yield_data: np.ndarray, data_path: str = '../data/', data_stage: str = 'development', optional_indices: list = []) -> np.ndarray:
    """
    Scales the yield data, saves the scaler (and the data if data_path is provided), and returns the scaled yield data

    Args:
        yield_data (np.ndarray): pre-loaded
        data_path (str, optional): Must be provided to save data. Defaults to '../data/'.
        data_stage (str, optional): Must be provided to save data and acts as an identifier. Defaults to 'development'.

    Returns:
        np.ndarray: yield_data_scaled
    """    
    scaler_y =  MinMaxScaler(feature_range=(-1, 1))
    if optional_indices:
        yield_data = yield_data[optional_indices]
    yield_train_reshaped = yield_data.reshape((yield_data.shape[0], 1)) 
    scaler_y = scaler_y.fit(yield_train_reshaped)
    yield_data_scaled = scaler_y.transform(yield_train_reshaped)
    logging.info('Yield data scaled')
    scaler_filename = '../results/yield_scaler.sav'
    with open(scaler_filename,'wb') as writer:
        joblib.dump(scaler_y,writer) #to save for later
    if data_path:
        filename = data_path + f'scaled_yield_{data_stage}.npy'
        with open(filename,'wb') as writer:
            np.save(writer,yield_data_scaled)
    return yield_data_scaled




def combine_weather_other_data(scaled_weather_data: np.ndarray, other_data_1he: np.ndarray, data_path: str = '../data/', data_stage: str = 'development') -> np.ndarray:
    """
    Combines the weather and other data via numpy broadcasting and column stacking. Saves data if paths are provided. Returns combined data in a 3-D array

    Args:
        scaled_weather_data (np.ndarray): 3-D scaled weather data
        other_data_1he (np.ndarray): 2-D one-hot encoded other data
        data_path (str, optional): Must be provided to save data. Defaults to '../data/'.
        data_stage (str, optional): Must be provided to save data and acts as an identifier. Defaults to 'development'.

    Returns:
        np.ndarray: 3-D combined data array
    """    
    desired_arr_shape = (scaled_weather_data.shape[0], scaled_weather_data.shape[1], other_data_1he.shape[1] + scaled_weather_data.shape[2])
    new_array = []
    for ind,sub_matrix in enumerate(scaled_weather_data):
        other_data_sub = other_data_1he[ind]
        broadcasted_other = np.broadcast_to(other_data_sub,(sub_matrix.shape[0],other_data_sub.shape[0]))
        new = np.column_stack((sub_matrix,broadcasted_other))
        new = new.astype(np.float32)
        new_array.append(new)
    final_array = np.array(new_array)
    logging.info('Combined data')
    assert final_array.shape == desired_arr_shape
    if data_path:
        filename = data_path + f'combined_weather_mgcluster_214_all_{data_stage}.npy'
        with open(filename,'wb') as writer:
            np.save(writer,final_array)
    return final_array




def split_data_into_training_and_validation(combined_X: np.ndarray, scaled_yield: np.ndarray, validation_size: float = 0.25, data_path: str = '../data/') -> Tuple:
    """
    Splits data into optional training and validation sets. Returns combined_X_train, combined_X_validation, scaled_yield_train, scaled_yield_validation, but is mostly used to save the data instead. 

    Args:
        combined_X (np.ndarray): 3-D array of combined data
        scaled_yield (np.ndarray): 1-D array of yield data
        validation_size (float, optional): Splitting ratio. Defaults to 0.25.
        data_path (str, optional): Must be present to save data. Defaults to '../data/'.

    Returns:
        Tuple: combined_X_train, combined_X_validation, scaled_yield_train, scaled_yield_validation
    """    
    combined_X_train, combined_X_validation, scaled_yield_train, scaled_yield_validation = train_test_split(combined_X,scaled_yield,test_size=validation_size)
    logging.info('Split data')
    if data_path:
        combined_X_train_file_path = data_path + 'combined_data_train.npy'
        with open(combined_X_train_file_path,'wb') as writer:
            np.save(writer,combined_X_train)
            logging.info('Saved training data')

        combined_X_validation_file_path = data_path + 'combined_data_validation.npy'
        with open(combined_X_validation_file_path,'wb') as writer:
            np.save(writer,combined_X_validation)
            logging.info('Saved validation data')
        
        
        yield_train_file_path = data_path + 'scaled_yield_train.npy'
        with open(yield_train_file_path,'wb') as writer:
            np.save(writer,scaled_yield_train)
            logging.info('Saved training yield data')
        
        
        yield_validation_file_path = data_path + 'scaled_yield_validation.npy'
        with open(yield_validation_file_path,'wb') as writer:
            np.save(writer,scaled_yield_validation)
            logging.info('Saved validation yield data')

    return combined_X_train, combined_X_validation, scaled_yield_train, scaled_yield_validation





def main():
    data_path = '../data/'
    weather_path = data_path + 'inputs_weather_train.npy'
    other_path = data_path + 'inputs_others_train.npy' 
    cluster_path = data_path + 'clusterID_genotype.npy'
    yield_path = data_path + 'yield_train.npy'
    paths_in_order = [weather_path,other_path,cluster_path,yield_path] 
    weather_data, other_data, cluster_data, yield_data = load_data(*paths_in_order) 

    encoded_other_data, passable_indices = clean_other_data(other_data,cluster_data)
    scaled_weather_data = scale_weather_data(weather_data,passable_indices)
    combined_data = combine_weather_other_data(scaled_weather_data,encoded_other_data,data_path=data_path)
    scaled_yield = scale_yield_data(yield_data, data_path=data_path, optional_indices=passable_indices)
    split_data_into_training_and_validation(combined_data,scaled_yield)


if __name__ == '__main__':
    main()