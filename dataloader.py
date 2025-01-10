import os
import pandas as pd
import numpy as np
# import json
# import matplotlib.pyplot as plt
import datetime

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
        # elif file_type == 'json':
        #     with open(file_path, 'r') as file:
        #         return json.load(file)
        # elif file_type == 'excel':
        #     return pd.read_excel(file_path)
        # elif file_type == 'txt':
        #     with open(file_path, 'r') as file:
        #         return file.read().split(delimiter)
        else:
            raise ValueError("Unsupported file type: {file_type}")
        
class Data:
    def __init__(self, modality_type, data_file_name):
        """
        :param modality_type: 데이터 모달리티 [EEG, ECG, GSR, PPG, vidio].
        :param data_fime_name: 데이터 파일 이름
        """
        self.modality_type = modality_type
        self.data_file_name = data_file_name
        self.loader = DataLoader('./recordings/Input')
        self.load_data = self.loader.load(self.data_file_name)
        self.timestamp = self.get_timestamp()
        self.data = self.get_data()

    def get_timestamp(self):
        if self.modality_type in ['ECG','GSR', 'PPG']:
            return self.load_data['sep=\t'].apply(lambda x: x.split('\t')[0])
        
    def get_data(self):
        if self.modality_type == 'ECG':
            return self.load_data['sep=\t'].apply(lambda x: x.split('\t')[3:])
        elif self.modality_type == 'GSR':
            return self.load_data['sep=\t'].apply(lambda x: x.split('\t')[2:4])
        elif self.modality_type == 'PPG':
            return self.load_data['sep=\t'].apply(lambda x: x.split('\t')[4])

def ms_unix_to_standard(unix_time, timezone_offset=9):
    """
    유닉스 시간을 표준 시간(YYYY-MM-DD HH:MM:SS.ms)으로 변환하는 함수.

    :param unix_time: 유닉스 시간 (ms 단위,  실수).
    :param timezone_offset: 표준 시간대 오프셋 (시간 단위, 기본값: 9 대한민국 시간 (UTC+9)).
    :return: 표준 시간 문자열 (YYYY-MM-DD HH:MM:SS.ms 형식).
    """
    try:
        # 유닉스 시간을 datetime 객체로 변환
        standard_time = datetime.datetime.utcfromtimestamp(float(unix_time)/1000.0) + datetime.timedelta(hours=timezone_offset)
        return standard_time.strftime('%Y-%m-%d %H:%M:%S')+'.'+str(float(unix_time)*100%100000).split('.')[0]
    except Exception as e:
        return f"오류 발생: {e}"
    
def standard_to_unix(standard_time, timezone_offset=9):
    """
    표준 시간(YYYY-MM-DD HH:MM:SS)을 유닉스 시간으로 변환하는 함수.

    :param standard_time: 표준 시간 문자열 (YYYY-MM-DD HH:MM:SS 형식).
    :param timezone_offset: 표준 시간대 오프셋 (시간 단위, 기본값: 9).
    :return: 유닉스 시간 (초 단위, float 형식).
    """
    try:
        # 표준 시간을 datetime 객체로 변환
        dt = datetime.datetime.strptime(standard_time, '%Y-%m-%d %H:%M:%S')
        # 시간대 오프셋을 제거한 UTC 시간으로 변환
        dt_utc = dt - datetime.timedelta(hours=timezone_offset)
        # UTC 시간을 유닉스 시간으로 변환
        return dt_utc.timestamp()
    except Exception as e:
        return f"오류 발생: {e}"
    
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
    # 데이터 로더 초기화
    data_path = "./recordings/Input"  # 데이터가 저장된 경로
    loader = DataLoader(data_path)

    EEG_file_name = ''
    ECG_file_name = 'LJY241119_E_Session1_id820D_Calibrated_SD.csv'
    GSR_file_name = 'LJY241119_GP_Session1_id95AE_Calibrated_SD.csv'
    PPG_file_name = 'LJY241119_GP_Session1_id95AE_Calibrated_SD.csv'
    vidio_file_name = ''

    ECG = Data('ECG', ECG_file_name)
    GSR = Data('GSR', GSR_file_name)
    PPG = Data('PPG', PPG_file_name)

    # print(ECG.timestamp, ECG.data)
    # print(GSR.timestamp, GSR.data)
    # print(PPG.timestamp, PPG.data)

    print('ECG timestamp range : '+ms_unix_to_standard(ECG.timestamp[2])+' ~ '+ms_unix_to_standard(ECG.timestamp[len(ECG.timestamp)-2]))
    print('GSR timestamp range : '+ms_unix_to_standard(GSR.timestamp[2])+' ~ '+ms_unix_to_standard(GSR.timestamp[len(GSR.timestamp)-2]))
    print('PPG timestamp range : '+ms_unix_to_standard(PPG.timestamp[2])+' ~ '+ms_unix_to_standard(PPG.timestamp[len(PPG.timestamp)-2]))

    target = ECG
    interpolate_lange = []


