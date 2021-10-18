from typing import Tuple, final
import numpy as np, pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib


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
    """[summary]

    Args:
        weather_data (np.ndarray): [description]

    Returns:
        np.ndarray: [description]
    """    
    
    scaler_x = MinMaxScaler(feature_range=(-1, 1))

    x_train_reshaped = weather_data.reshape((weather_data.shape[0], weather_data.shape[1] * weather_data.shape[2]))


    # Scaling Coefficients calculated from the training dataset
    scaler_x = scaler_x.fit(x_train_reshaped)   


    weather_data_scaled = scaler_x.transform(x_train_reshaped).reshape(weather_data.shape)

    return weather_data_scaled



def scale_yield_data(yield_data: np.ndarray, data_path: str = '../data/', data_stage: str = 'development') -> np.ndarray:
    """[summary]

    Args:
        yield_data (np.ndarray): [description]
        data_path (str, optional): [description]. Defaults to '../data/'.
        data_stage (str, optional): [description]. Defaults to 'development'.

    Returns:
        np.ndarray: [description]
    """    
    scaler_y =  MinMaxScaler(feature_range=(-1, 1))
    yield_train_reshaped = yield_data.reshape((yield_data.shape[0], 1))   # (82692, 1)
    scaler_y = scaler_y.fit(yield_train_reshaped)
    yield_data_scaled = scaler_y.transform(yield_train_reshaped)
    scaler_filename = '../results/yield_scaler.sav'
    with open(scaler_filename,'wb') as writer:
        joblib.dump(scaler_y,writer) #to save for later
    if data_path:
        filename = data_path + f'scaled_yield_{data_stage}.npy'
        with open(filename,'wb') as writer:
            np.save(writer,yield_data_scaled)
    return yield_data_scaled




def combine_weather_other_data(scaled_weather_data: np.ndarray, other_data_1he: np.ndarray, data_path: str = '../data/', data_stage: str = 'development') -> np.ndarray:
    """[summary]

    Args:
        scaled_weather_data (np.ndarray): [description]
        other_data_1he (np.ndarray): [description]
        data_path (str, optional): [description]. Defaults to '../data/'.
        data_stage (str, optional): [description]. Defaults to 'development'.

    Returns:
        np.ndarray: [description]
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
        filename = data_path + f'combined_weather_mgcluster_214_{data_stage}.npy'
        with open(filename,'wb') as writer:
            np.save(writer,final_array)
    return final_array




def split_data_into_training_and_validation(combined_X: np.ndarray, scaled_yield: np.ndarray, validation_size: float = 0.25, data_path: str = '../data/') -> Tuple:
    """[summary]

    Args:
        combined_X (np.ndarray): [description]
        scaled_yield (np.ndarray): [description]
        validation_size (float, optional): [description]. Defaults to 0.25.
        data_path (str, optional): [description]. Defaults to '../data/'.

    Returns:
        Tuple: [description]
    """    
    combined_X_train, combined_X_validation, scaled_yield_train, scaled_yield_validation = train_test_split(combined_X,scaled_yield,test_size=validation_size)
    if data_path:
        combined_X_train_file_path = data_path + 'combined_data_train.npy'
        with open(combined_X_train_file_path,'wb') as writer:
            np.save(writer,combined_X_train)

        combined_X_validation_file_path = data_path + 'combined_data_validation.npy'
        with open(combined_X_validation_file_path,'wb') as writer:
            np.save(writer,combined_X_validation)
        
        
        yield_train_file_path = data_path + 'scaled_yield_train.npy'
        with open(yield_train_file_path,'wb') as writer:
            np.save(writer,scaled_yield_train)
        
        
        yield_validation_file_path = data_path + 'scaled_yield_validation.npy'
        with open(yield_validation_file_path,'wb') as writer:
            np.save(writer,scaled_yield_validation)

    return combined_X_train, combined_X_validation, scaled_yield_train, scaled_yield_validation





def main():
    data_path = '../data/'
    weather_path = data_path + 'inputs_weather_train.npy'
    other_path = data_path + 'inputs_others_train.npy' 
    cluster_path = data_path + 'clusterID_genotype.npy'
    yield_path = data_path + 'yield_train.npy'
    paths_in_order = [weather_path,other_path,cluster_path,yield_path] 
    weather_data, other_data, cluster_data, yield_data = load_data(*paths_in_order) 

    encoded_other_data = clean_other_data(other_data,cluster_data)
    scaled_weather_data = scale_weather_data(weather_data)
    combined_data = combine_weather_other_data(scaled_weather_data,encoded_other_data,data_path=data_path)
    scaled_yield = scale_yield_data(yield_data, data_path=data_path)
    split_data_into_training_and_validation(combined_data,scaled_yield)


if __name__ == '__main__':
    main()