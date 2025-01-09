import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline

def interpolate_1Dsignals(df, target_rate):
 
    time_column = df.columns[0]
    signal_columns = df.columns[1:]

    original_time = df[time_column].values

    num_points = int((original_time[-1] - original_time[0]) * target_rate / 1000) + 1
    target_time = np.linspace(original_time[0], original_time[-1], num_points)

    interpolated_data = {'timestamp': target_time}

    for column in signal_columns:
        signal = df[column].values
        interpolator = interp1d(original_time, signal, kind='cubic', fill_value="extrapolate")
        interpolated_data[column] = interpolator(target_time)

    interpolated_df = pd.DataFrame(interpolated_data)

    return interpolated_df

def interpolate_video(df, video, target_rate):

    time_column = df.columns[0]