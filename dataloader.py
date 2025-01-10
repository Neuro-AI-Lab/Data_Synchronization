import os
import pandas as pd
import json

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

    print(ECG.timestamp, ECG.data)
    print(GSR.timestamp, GSR.data)
    print(PPG.timestamp, PPG.data)