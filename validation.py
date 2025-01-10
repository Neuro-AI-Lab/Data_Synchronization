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

def save_frames_to_video(frames, output_path, fps=30):
    if not frames:
        print("프레임 리스트가 비어 있습니다.")
        return

    # 프레임 크기 확인
    height, width, channels = frames[0].shape
    size = (width, height)

    # VideoWriter 객체 생성
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # AVI 코덱
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    # 프레임 추가
    for frame in frames:
        out.write(frame)

    # 자원 해제
    out.release()
    print(f"동영상 저장 완료: {output_path}")
