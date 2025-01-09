import os
import pandas as pd
import numpy as np

# csv 읽는 함수수
def load_csv(directory, file_name):
    file_path = os.path.join(directory, file_name)
    return pd.read_csv(file_path)

# 데이터 포인트 확인 및 보완 함수
def check_and_adjust_signals(df, start_time, end_time, sampling_rate):
    filtered_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

    required_points = int(sampling_rate * (end_time - start_time)) # 이상적 필요 데이터 포인트 수
    actual_points = len(filtered_df) # 실제 측정된 데이터 포인트 수수

    print(f"Required Points: {required_points}, Actual Points: {actual_points}")

    if actual_points < required_points: # 데이터 포인트가 부족하면 종료시간 이후에서 가져옴
        missing_points = required_points - actual_points
        additional_data = df[df['timestamp'] > end_time].head(missing_points)

        if len(additional_data) < missing_points: # 뒤에서 부족한 만큼 채울 수 없는 경우
            print("Warning: Not enough additional data to fill missing points.")
        
        combined_df = pd.concat([filtered_df, additional_data]).sort_values('timestamp')
        print(f"Added {len(additional_data)} points from additional data.")

    elif actual_points > required_points: # 데이터 포인트가 넘칠 경우 뒤에서부터 삭제
        excess_points = actual_points - required_points
        combined_df = filtered_df.iloc[:-excess_points]
        print(f"Removed {excess_points} excess points.")

    else: # 완벽하게 데이터 포인트가 구성되었다면 그대로 반환환
        combined_df = filtered_df
        print("Data points are correct.")

    return combined_df

