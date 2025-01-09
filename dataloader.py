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
            elif file_name.endswith('.json'):
                file_type = 'json'
            elif file_name.endswith(('.xls', '.xlsx')):
                file_type = 'excel'
            elif file_name.endswith('.txt'):
                file_type = 'txt'
            else:
                raise ValueError("Unsupported file type. Please specify the file_type.")

        # 파일 형식에 따라 데이터 로드
        if file_type == 'csv':
            return pd.read_csv(file_path)
        elif file_type == 'json':
            with open(file_path, 'r') as file:
                return json.load(file)
        elif file_type == 'excel':
            return pd.read_excel(file_path)
        elif file_type == 'txt':
            with open(file_path, 'r') as file:
                return file.read().split(delimiter)
        else:
            raise ValueError("Unsupported file type: {file_type}")