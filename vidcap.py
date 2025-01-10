import cv2
import time
import pandas as pd
import os
from datetime import datetime


def get_directory_by_name(base_dir, custom_name):
    """Creates a directory under base_dir with the given custom name."""
    custom_path = os.path.join(base_dir, custom_name)
    if not os.path.exists(custom_path):
        os.makedirs(custom_path)
    return custom_path


def record_video(directory, fps=30):
    """Records a video and saves timestamps in memory. Writes timestamps to CSV after recording stops."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_FPS, fps)

    video_path = os.path.join(directory, 'output_video.avi')
    csv_path = os.path.join(directory, 'video.csv')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

    start_time = time.time()
    timestamps = []  # 타임스탬프를 메모리에 저장

    print("Recording... Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        current_time = time.time()
        timestamps.append(current_time)
        out.write(frame)

        cv2.imshow('Recording', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording stopped by user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 타임스탬프를 CSV로 저장
    timestamp_df = pd.DataFrame({
        'timestamp': timestamps,
        'Elapsed Time (s)': [t - start_time for t in timestamps]
    })
    timestamp_df.to_csv(csv_path, index=False)

    print(f"Video saved to {video_path}")
    print(f"Timestamps saved to {csv_path}")


if __name__ == "__main__":
    base_dir = 'recordings'

    custom_name = str("Input")
    if not custom_name:
        print("No folder name entered. Exiting.")
        exit()

    next_dir = get_directory_by_name(base_dir, custom_name)

    fps = int(30)
    if fps:
        record_video(next_dir, fps=fps)
    else:
        print("No FPS value entered. Exiting.")
