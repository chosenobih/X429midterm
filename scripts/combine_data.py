from typing import final
import numpy as np, pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import joblib


def load_data(weather_path,other_path, cluster_id_path, yield_path):
    return np.load(weather_path), np.load(other_path), np.load(cluster_id_path), np.load(yield_path)



def clean_other_data(other_data,cluster_id_data):
    other_df = pd.DataFrame(other_data[[0,1]])
    other_df.columns = ['Maturity Group', 'Genotype ID']
    for col in other_df.columns:
        other_df[col] = other_df[col].astype(float).astype(int)
    other_df['Genotype ID'] -= 1 #to match indexing for cluster_id_data
    cluster_id_mapper = lambda genotype_id: cluster_id_data[genotype_id]
    other_df['Genotype ID'] = other_df['Genotype ID'].apply(cluster_id_mapper)
    #now one hot encode all data 
    return OneHotEncoder().fit_transform(other_df)




def scale_weather_data(weather_data):
    
    scaler_x = MinMaxScaler(feature_range=(-1, 1))

    x_train_reshaped = weather_data.reshape((weather_data.shape[0], weather_data.shape[1] * weather_data.shape[2]))


    # Scaling Coefficients calculated from the training dataset
    scaler_x = scaler_x.fit(x_train_reshaped)   


    weather_data_scaled = scaler_x.transform(x_train_reshaped).reshape(weather_data.shape)

    return weather_data_scaled



def scale_yield_data(yield_data):
    scaler_y =  MinMaxScaler(feature_range=(-1, 1))
    yield_train_reshaped = yield_data.reshape((yield_data.shape[0], 1))   # (82692, 1)
    scaler_y = scaler_y.fit(yield_train_reshaped)
    yield_data_scaled = scaler_y.transform(yield_train_reshaped)
    scaler_filename = '../results/yield_scaler.sav'
    joblib.dump(scaler_y,scaler_filename) #to save for later
    
    return yield_data_scaled




def combine_weather_other_data(scaled_weather_data, other_data_1he):
    desired_arr_shape = (scaled_weather_data.shape[0], scaled_weather_data.shape[1], other_data_1he.shape[1] + scaled_weather_data.shape[2])
    new_array = []
    for ind,sub_matrix in enumerate(scaled_weather_data):
        other_data_sub = other_data_1he[ind]
        broadcasted_other = np.broadcast_to(other_data_sub,(sub_matrix.shape[0],other_data_sub.shape[1]))
        new = np.column_stack((sub_matrix,broadcasted_other))
        new_array.append(new)
    final_array = np.array(new_array)
    assert final_array.shape == desired_arr_shape
    return final_array



def main():
    pass

if __name__ == '__main__':
    main()