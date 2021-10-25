'''
Ryan Papetti
Team X
INFO 429/529 Midterm
Fall 2021

test_handle_data.py contains the methods to convert the standard competition datasets into a meaninful one for training with any mode. Offers utilities to save and split data as well
'''


from typing import Tuple
import numpy as np, pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def load_data(weather_path:str,other_path:str, cluster_id_path:str) -> Tuple:
    """
    Loads all relevant data for processing via inputted filenames as specified

    Args:
        weather_path (str): path to weather data
        other_path (str): path to other data
        cluster_id_path (str): path to clusterID data

    Returns:
        Tuple: tuple of arrays where each array is loaded from each path (in order)
    """    
    return np.load(weather_path), np.load(other_path), np.load(cluster_id_path)



def clean_other_data(other_data: np.ndarray,cluster_id_data: np.ndarray) -> np.ndarray:
    """
    Takes other data and cluster_id_data. Assigns the proper genotype_id and returns all the data as a onehotencoded matrix. As of now only the first two variables are chosen. This will likely change. 

    Args:
        other_data (np.ndarray): pre-loaded
        cluster_id_data (np.ndarray): pre-loaded

    Returns:
        np.ndarray: OneHotEncoded matrix
    """    
    other_df = pd.DataFrame(other_data[:,[0,1]])
    other_df.columns = ['Maturity Group', 'Genotype ID']
    for col in other_df.columns:
        other_df[col] = other_df[col].astype(np.float32).astype(int)
    other_df['Genotype ID'] -= 1 #to match indexing for cluster_id_data
    cluster_id_mapper = lambda genotype_id: cluster_id_data[genotype_id]
    other_df['Genotype ID'] = other_df['Genotype ID'].apply(cluster_id_mapper)
    #now one hot encode all data 
    return OneHotEncoder().fit_transform(other_df).toarray().astype('float32')




def scale_weather_data(weather_data: np.ndarray) -> np.ndarray:
    """
    
    Takes Weather data, reshapes it, and returns a scaled version

    Args:
        weather_data (np.ndarray): pre-loaded

    Returns:
        np.ndarray: weather_data_scaled
    """    
    
    scaler_x = MinMaxScaler(feature_range=(-1, 1))

    x_train_reshaped = weather_data.reshape((weather_data.shape[0], weather_data.shape[1] * weather_data.shape[2]))


    # Scaling Coefficients calculated from the training dataset
    scaler_x = scaler_x.fit(x_train_reshaped)   


    weather_data_scaled = scaler_x.transform(x_train_reshaped).reshape(weather_data.shape)

    return weather_data_scaled


def combine_weather_other_data(scaled_weather_data: np.ndarray, other_data_1he: np.ndarray, data_path: str = '../data/', data_stage: str = 'TEST') -> np.ndarray:
    """
    Combines the weather and other data via numpy broadcasting and column stacking. Saves data if paths are provided. Returns combined data in a 3-D array

    Args:
        scaled_weather_data (np.ndarray): 3-D scaled weather data
        other_data_1he (np.ndarray): 2-D one-hot encoded other data
        data_path (str, optional): Must be provided to save data. Defaults to '../data/'.
        data_stage (str, optional): Must be provided to save data and acts as an identifier. Defaults to 'TEST'.

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
    assert final_array.shape == desired_arr_shape
    if data_path:
        filename = data_path +  f'{data_stage}/combined_{data_stage}.npy'
        with open(filename,'wb') as writer:
            np.save(writer,final_array)
    return final_array

def main():
    data_path = '../data/'
    weather_path = data_path + 'TEST/inputs_weather_test.npy'
    other_path = data_path + 'TEST/inputs_others_test.npy' 
    cluster_path = data_path + 'clusterID_genotype.npy'   
    paths_in_order = [weather_path,other_path,cluster_path] 
    weather_data, other_data, cluster_data = load_data(*paths_in_order) 

    encoded_other_data = clean_other_data(other_data,cluster_data)
    scaled_weather_data = scale_weather_data(weather_data)
    _ = combine_weather_other_data(scaled_weather_data,encoded_other_data,data_path=data_path)



if __name__ == '__main__':
    main()
