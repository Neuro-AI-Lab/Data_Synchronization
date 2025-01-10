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

    # 보간할 총 샘플 수 계산
    total_samples = int((timestamps[-1] - timestamps[0]) * sampling_rate)

    # 보간된 타임스탬프 생성
    interpolated_timestamps = np.linspace(timestamps[0], timestamps[-1], total_samples)

    # 보간된 프레임 데이터를 저장할 배열 생성
    height, width, channels = frame_data.shape[1:]  # 영상 해상도 및 채널 정보 추출
    interpolated_frames = np.empty((total_samples, height, width, channels), dtype=np.uint8)

    # 각 RGB 채널별로 보간 수행
    for channel in range(channels):
        print(f"Processing channel {channel + 1}/{channels}...")  # 상태 출력
        channel_data = frame_data[..., channel]  # 해당 채널 데이터 추출 (N, H, W)
        interpolated_channel = np.empty((total_samples, height, width), dtype=np.uint8)

        # 각 픽셀 위치에 대해 보간 수행 (높이 x 너비만큼 반복)
        for h in range(height):
            for w in range(width):
                pixel_values = channel_data[:, h, w]  # 픽셀 타임라인 데이터 추출
                interpolation_function = interp1d(timestamps, pixel_values, kind='linear', fill_value="extrapolate")
                interpolated_channel[:, h, w] = np.clip(interpolation_function(interpolated_timestamps), 0, 255)

        interpolated_frames[..., channel] = interpolated_channel

    return interpolated_timestamps.tolist(), interpolated_frames