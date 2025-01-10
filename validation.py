import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr

def synchronize_nearest_frames(original_frames, original_timestamps, interpolated_frames, interpolated_timestamps):
    """
    30 FPS 원본 프레임과 500 Hz 보간된 프레임을 동기화.
    """
    synchronized_frames = []
    for t in original_timestamps:

        # 보간된 타임스탬프 중에서 원본 타임스탬프와 가장 가까운 값 선택
        closest_idx = np.argmin(np.abs(np.array(interpolated_timestamps) - t))
        print('original timestamp :', t)
        print('interpolated timestamps :', interpolated_timestamps[closest_idx])
        synchronized_frames.append(interpolated_frames[closest_idx])

    return synchronized_frames

def calculate_psnr(original_frames, synchronized_frames):
    """
    PSNR 계산.
    """
    psnr_values = []
    for original, synchronized in zip(original_frames, synchronized_frames):
        if isinstance(original, np.ndarray) and isinstance(synchronized, np.ndarray):
            value = psnr(original, synchronized, data_range=255)
            psnr_values.append(value)
        else:
            print("Invalid frame format. Skipping...")

    return np.mean(psnr_values)

