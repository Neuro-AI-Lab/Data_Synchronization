import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch


def lowpass_filter(data, highcut=5.0, fs=128, order=5):
    """
    Applies a Butterworth lowpass filter with the given highcut frequency.
    """
    nyquist = 0.5 * fs  # 나이퀴스트 주파수 계산
    high = highcut / nyquist
    b, a = butter(order, high, btype='low')  # 저역 필터 계수 계산
    filtered_data = filtfilt(b, a, data)  # 필터 적용
    return filtered_data


# 데이터 읽기
file_path = r'C:\Users\hyuk\PycharmProjects\C100_DKB\Shimmer\sample\GP2.csv'
df = pd.read_csv(file_path, skiprows=[0, 2], sep='\t', low_memory=False)


df = df[500:1000].copy()


conductance_columns = [col for col in df.columns if 'id95AE_GSR_Skin_Conductance_CAL' in col]
for col in conductance_columns:
    df[col] = lowpass_filter(df[col], highcut=5.0, fs=128, order=5)



def on_key(event):
    if event.key == 'e':
        plt.close('all')


# 각 열에 대해 그래프 생성
figures = []
for column in df.columns:
    fig, ax = plt.subplots()
    ax.plot(df[column])

    if column == 'id95AE_GSR_Skin_Conductance_CAL':
        ax.set_title('Skin Conductance')
        ax.set_xlabel("Sampling point")
        ax.set_ylabel("Value (uS)")
    elif column == 'id95AE_GSR_Skin_Resistance_CAL':
        ax.set_title('Skin Resistance')
        ax.set_xlabel("Sampling point")
        ax.set_ylabel("Value (kOhms)")
    elif column == 'id95AE_PPG_A13_CAL':
        ax.set_title('PPG')
        ax.set_xlabel("Sampling point")
        ax.set_ylabel("Value (mV)")
    else:
        ax.set_title(column)
        ax.set_xlabel("Sampling point")
        ax.set_ylabel("Value")

    fig.canvas.mpl_connect('key_press_event', on_key)
    figures.append(fig)

# 그래프를 순서대로 표시
for fig in reversed(figures):
    fig.show()
    plt.pause(0)

print("Press 'e' in each window to close them in order.")
