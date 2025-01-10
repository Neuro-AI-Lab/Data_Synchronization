import os
import pandas as pd
import numpy as np
# import json
# import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import json
import cv2

class DataLoader:
    def __init__(self, data_path):
        """
        데이터가 저장된 경로를 초기화합니다.

        :param data_path: 데이터가 저장된 디렉토리 또는 파일 경로.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The specified path does not exist: {data_path}")
        self.data_path = data_path

    def load(self, file_name, file_type=None, delimiter='\n'):
        """
        지정된 경로에서 파일을 로드합니다.

        :param file_name: 로드할 파일의 이름.
        :param file_type: 파일 형식(csv, json, excel, txt). None인 경우 확장자로 결정합니다.
        :param delimiter: 텍스트 파일(txt) 로드 시 사용할 구분자. 기본값은 '\n'.
        :return: 로드된 데이터 (pandas DataFrame, dict 또는 list).
        """
        file_path = os.path.join(self.data_path, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # 파일 형식을 확장자로 결정
        if file_type is None:
            if file_name.endswith('.csv'):
                file_type = 'csv'
            elif file_name.endswith('.avi'):
                file_type = 'avi'

            # elif file_name.endswith('.json'):
            #     file_type = 'json'
            # elif file_name.endswith(('.xls', '.xlsx')):
            #     file_type = 'excel'
            # elif file_name.endswith('.txt'):
            #     file_type = 'txt'
            
            else:
                raise ValueError("Unsupported file type. Please specify the file_type.")

        # 파일 형식에 따라 데이터 로드
        if file_type == 'csv':
            return pd.read_csv(file_path)
        elif file_type == 'avi':
            return file_path  # 비디오 파일 경로 반환
        
        # elif file_type == 'json':
        #     with open(file_path, 'r') as file:
        #         return json.load(file)
        # elif file_type == 'excel':
        #     return pd.read_excel(file_path)
        # elif file_type == 'txt':
        #     with open(file_path, 'r') as file:
        #         return file.read().split(delimiter)
        
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

class VideoLoader:
    def __init__(self, video_path, timestamp_path=None):
        """
        비디오 데이터를 로드하고 프레임 및 타임스탬프를 추출합니다.

        :param video_path: 비디오 파일 경로.
        :param timestamp_path: 타임스탬프 CSV 파일 경로 (선택적).
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.video_path = video_path
        self.timestamp_path = timestamp_path
        self.timestamps = self._load_timestamps() if timestamp_path else None

    def _load_timestamps(self):
        """
        타임스탬프 데이터를 로드합니다.

        :return: 타임스탬프를 포함한 Pandas DataFrame.
        """
        if not os.path.exists(self.timestamp_path):
            raise FileNotFoundError(f"Timestamp file not found: {self.timestamp_path}")

        # 타임스탬프 CSV 파일 로드
        timestamps = pd.read_csv(self.timestamp_path)
        
        # 타임스탬프 열이 있는지 확인
        if 'timestamp' not in timestamps.columns:
            raise ValueError("The timestamp CSV must contain a 'timestamp' column.")
        
        return timestamps['timestamp'].tolist()

    def load_frames(self):
        """
        비디오 프레임을 로드하고 타임스탬프와 매핑합니다.

        :return: 프레임 데이터와 타임스탬프 데이터 (리스트로 반환).
        """
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")

        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        # 프레임 수와 타임스탬프 수 확인
        if self.timestamps is not None and len(frames) != len(self.timestamps):
            raise ValueError(
                f"Frame count ({len(frames)}) and timestamp count ({len(self.timestamps)}) do not match."
            )

        return frames, self.timestamps

class Data:
    def __init__(self, modality_type, data_file_name, timestamp_file_name=None):
        """
        :param modality_type: 데이터 모달리티 [EEG, ECG, GSR, PPG, VIDEO].
        :param data_file_name: 데이터 파일 이름.
        :param timestamp_file_name: VIDEO 타임스탬프 파일 이름.
        """
        self.modality_type = modality_type
        self.data_file_name = data_file_name
        self.timestamp_file_name = timestamp_file_name
        self.loader = DataLoader('./recordings/Input')
        
        if modality_type == 'VIDEO':
            video_path = self.loader.load(self.data_file_name, 'avi')
            timestamp_path = (
                os.path.join(self.loader.data_path, self.timestamp_file_name)
                if self.timestamp_file_name
                else None
            )
            self.video_loader = VideoLoader(video_path, timestamp_path)
            self.frames, self.timestamp = self.video_loader.load_frames()
            self.data = self.frames
        else:
            self.load_data = self.loader.load(self.data_file_name)
            self.timestamp = self.get_timestamp()
            self.data = self.get_data()

    def get_timestamp(self):
        if self.modality_type in ['ECG', 'GSR', 'PPG']:
            temp = self.load_data['sep=\t'][2:].apply(lambda x: float(x.split('\t')[0])/1000.0)
            temp.index = pd.RangeIndex(start=0, stop=len(temp), step = 1)
            return temp
        # add new modality here

    def get_data(self):
        if self.modality_type == 'ECG':
            return self.load_data['sep=\t'].apply(lambda x: x.split('\t')[3:])
        elif self.modality_type == 'GSR':
            return self.load_data['sep=\t'].apply(lambda x: x.split('\t')[2:4])
        elif self.modality_type == 'PPG':
            return self.load_data['sep=\t'].apply(lambda x: x.split('\t')[4])
        # add new modality here

def unix_to_standard(unix_time, timezone_offset=9):
    """
    유닉스 시간을 표준 시간(YYYY-MM-DD HH:MM:SS.sss)으로 변환하는 함수.

    :param unix_time: 유닉스 시간 (초 단위, 실수).
    :return: 표준 시간 문자열 (YYYY-MM-DD HH:MM:SS.sss 형식).
    """
    try:
        # 유닉스 시간을 timezone-aware UTC datetime 객체로 변환
        utc_time = datetime.fromtimestamp(unix_time, tz=timezone.utc)
        # 시간대 오프셋을 추가
        local_time = utc_time + timedelta(hours=timezone_offset)
        # 밀리초 계산
        milliseconds = str(unix_time).split('.')[1]
        # milliseconds = int((unix_time - int(unix_time)) * 1000)
        return local_time.strftime('%Y-%m-%d %H:%M:%S') + f".{milliseconds}"
    except Exception as e:
        return f"오류 발생: {e}"

def standard_to_unix(standard_time, timezone_offset=0):
    """
    표준 시간(YYYY-MM-DD HH:MM:SS.sss)을 유닉스 시간으로 변환하는 함수.

    :param standard_time: 표준 시간 문자열 (YYYY-MM-DD HH:MM:SS.sss 형식).
    :param timezone_offset: 표준 시간대 오프셋 (시간 단위, 기본값: 9).
    :return: 유닉스 시간 (초 단위, float 형식).
    """
    try:
        # 표준 시간을 datetime 객체로 변환
        time_part, milliseconds_part = standard_time.split('.')
        local_time = datetime.strptime(time_part, '%Y-%m-%d %H:%M:%S')
        # 시간대 오프셋을 제거하여 UTC로 변환
        utc_time = local_time - timedelta(hours=timezone_offset)
        # UTC 시간을 유닉스 시간으로 변환
        return float(str(utc_time.timestamp()).split('.')[0] +'.'+ milliseconds_part)
    except Exception as e:
        return f"오류 발생: {e}"

def get_data_point_index(target_signal, interpolate_range):
    return target_signal.timestamp[(target_signal.timestamp >= standard_to_unix(interpolate_range[0])) & (target_signal.timestamp <= standard_to_unix(interpolate_range[1]))].index

# def interpolate_shimmer_timestamps(timestamps, values, new_timestamps):
#     """
#     쉬머 타임스탬프를 기준으로 값을 선형 보간(interpolation)하는 함수.

#     :param timestamps: 기존 타임스탬프 (밀리초 단위, 리스트 또는 numpy 배열).
#     :param values: 기존 타임스탬프에 대응하는 값 (리스트 또는 numpy 배열).
#     :param new_timestamps: 새로 보간할 타임스탬프 (밀리초 단위, 리스트 또는 numpy 배열).
#     :return: 보간된 값의 리스트.
#     """
#     try:
#         # 밀리초 단위를 초 단위로 변환
#         timestamps = np.array(timestamps) / 1000.0
#         new_timestamps = np.array(new_timestamps) / 1000.0

#         # 선형 보간 수행
#         interpolated_values = np.interp(new_timestamps, timestamps, values)
#         return interpolated_values
#     except Exception as e:
#         return f"오류 발생: {e}"


if __name__ == "__main__":
    data_path = "./recordings/Input"
    loader = DataLoader(data_path)

    EEG_file_name = ''
    ECG_file_name = 'LJY241119_E_Session1_id820D_Calibrated_SD.csv'
    GSR_file_name = 'LJY241119_GP_Session1_id95AE_Calibrated_SD.csv'
    PPG_file_name = 'LJY241119_GP_Session1_id95AE_Calibrated_SD.csv'

    video_file_name = 'output_video.avi'
    video_timestamp_name = 'video.csv'

    ECG = Data('ECG', ECG_file_name)
    GSR = Data('GSR', GSR_file_name)
    PPG = Data('PPG', PPG_file_name)
    VIDEO = Data('VIDEO', video_file_name, video_timestamp_name)

    # print(ECG.timestamp, ECG.data)
    # print(GSR.timestamp, GSR.data)
    # print(PPG.timestamp, PPG.data)

    print('ECG timestamp range : '+unix_to_standard(ECG.timestamp[2])+' ~ '+unix_to_standard(ECG.timestamp[len(ECG.timestamp)-2]))
    print('GSR timestamp range : '+unix_to_standard(GSR.timestamp[2])+' ~ '+unix_to_standard(GSR.timestamp[len(GSR.timestamp)-2]))
    print('PPG timestamp range : '+unix_to_standard(PPG.timestamp[2])+' ~ '+unix_to_standard(PPG.timestamp[len(PPG.timestamp)-2]))

    print(datetime.now())

    interpolate_data_point_idx = get_data_point_index(target_signal=ECG, interpolate_range=['2024-11-19 16:29:00.01213', '2024-11-19 16:30:00.114135'])

    print(interpolate_data_point_idx)

    print(len(VIDEO.data))
