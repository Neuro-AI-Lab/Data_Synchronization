import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d

def load_csv(directory, file_name):
    file_path = os.path.join(directory, file_name)
    return pd.read_csv(file_path)

def find_overlap(*dataframes):
    start = max(df['timestamp'].iloc[0] for df in dataframes)
    end = min(df['timestamp'].iloc[-1] for df in dataframes)
    return start, end

def interpolate_signals(df, target_timestamps):
    interpolated_data = {'timestamp': target_timestamps}
    for column in df.columns:
        if column != 'timestamp':
            f = interp1d(df['timestamp'], df[column], kind='cubic', bounds_error=False, fill_value="extrapolate")
            interpolated_data[column] = f(target_timestamps)
    return pd.DataFrame(interpolated_data)

def select_segment_and_resample(video_fps, segment_duration, video_timestamps, signal_rates, *signal_dfs):
    num_video_points = int(video_fps * segment_duration)

    segment_start = video_timestamps[0]
    segment_end = segment_start + segment_duration
    segment_timestamps = video_timestamps[(video_timestamps >= segment_start) & (video_timestamps < segment_end)]

    if len(segment_timestamps) < num_video_points:
        segment_timestamps = np.linspace(segment_start, segment_end, num_video_points)

    resampled_dfs = []
    for df, rate in zip(signal_dfs, signal_rates):
        segment_data = df[(df['timestamp'] >= segment_start) & (df['timestamp'] < segment_end)]
        target_timestamps = np.linspace(segment_start, segment_end, int(rate * segment_duration))
        resampled_dfs.append(interpolate_signals(segment_data, target_timestamps))

    return segment_timestamps, resampled_dfs

if __name__ == "__main__":
    
    input_directory = "C:\Users\이동혁\Desktop\vidcap\signals"
    output_directory = "C:\Users\이동혁\Desktop\vidcap\output"

    os.makedirs(output_directory, exist_ok=True)

    eeg_df = load_csv(input_directory, "eeg.csv")
    ecg_df = load_csv(input_directory, "ecg.csv")
    gsr_df = load_csv(input_directory, "gsr.csv")
    video_df = load_csv(input_directory, "video.csv") 

    video_timestamps = video_df['timestamp'].values

    start, end = find_overlap(eeg_df, ecg_df, gsr_df, video_df)

    eeg_df = eeg_df[(eeg_df['timestamp'] >= start) & (eeg_df['timestamp'] <= end)]
    ecg_df = ecg_df[(ecg_df['timestamp'] >= start) & (ecg_df['timestamp'] <= end)]
    gsr_df = gsr_df[(gsr_df['timestamp'] >= start) & (gsr_df['timestamp'] <= end)]
    video_timestamps = video_timestamps[(video_timestamps >= start) & (video_timestamps <= end)]
    
    segment_duration = 3 # 사용할 데이터 포인트를 초 단위로 clip
    video_fps = 30
    signal_rates = [2048, 512, 256] # EEG, ECG, GSR sampling rate

    segment_timestamps, resampled_dfs = select_segment_and_resample(video_fps, segment_duration, video_timestamps, signal_rates, eeg_df, ecg_df, gsr_df)

    resampled_dfs[0].to_csv(os.path.join(output_directory, "synchronized_eeg.csv"), index=False)
    resampled_dfs[1].to_csv(os.path.join(output_directory, "synchronized_ecg.csv"), index=False)
    resampled_dfs[2].to_csv(os.path.join(output_directory, "synchronized_gsr.csv"), index=False)

    print("Synchronization Complete")
