import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline

def interpolate_1Dsignals(df, target_rate):
 
    """
    유닉스 타임스탬프와 생체신호호 데이터를 샘플링 레이트 기반으로 보간.
    """

    return interpolated_df

def interpolate_video(timestamps, frame_data, sampling_rate):
    """
    유닉스 타임스탬프와 프레임 데이터를 샘플링 레이트 기반으로 보간 (uint8).
    """
    # 입력 데이터를 numpy 배열로 변환
    timestamps = np.array(timestamps)
    frame_data = np.array(frame_data, dtype=np.uint8)  # uint8 타입으로 변환
    frame_indices = np.arange(len(frame_data))

    # 보간할 총 샘플 수 계산
    total_samples = int((timestamps[-1] - timestamps[0]) * sampling_rate)

    # 보간된 타임스탬프 생성
    interpolated_timestamps = np.linspace(timestamps[0], timestamps[-1], total_samples)

    interpolated_frames = []
    
    for channel in range(frame_data.shape[-1]):  # RGB 채널 반복
        channel_data = frame_data[:, :, :, channel]
        interpolation_function = interp1d(timestamps, channel_data, axis=0, kind='linear', fill_value="extrapolate")
        interpolated_channel = interpolation_function(interpolated_timestamps)
        
        interpolated_channel = np.clip(interpolated_channel, 0, 255).astype(np.uint8)
        interpolated_frames.append(interpolated_channel)

    interpolated_frames = np.stack(interpolated_frames, axis=-1)

    return interpolated_timestamps.tolist(), interpolated_frames