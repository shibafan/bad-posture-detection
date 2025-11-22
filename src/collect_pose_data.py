import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, enable_segmentation=False)
draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)
print(cap.isOpened())

def extract_landmark_data(image):
    '''
    Takes a BGR image from cv2 and flattens it into a list of normalized pose landmark coordinates
    '''

    # Convert to RGB for mediapipe
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if not res.pose_landmarks:
        return None

    lm = res.pose_landmarks.landmark

    # Shoulder landmarks
    L_SH = 11
    R_SH = 12

    # Mid-shoulder anchor (normalized origin)
    mid_sh_x = (lm[L_SH].x + lm[R_SH].x) / 2
    mid_sh_y = (lm[L_SH].y + lm[R_SH].y) / 2
    mid_sh_z = (lm[L_SH].z + lm[R_SH].z) / 2

    row = []
    for point in lm:
        x = point.x - mid_sh_x
        y = point.y - mid_sh_y
        z = point.z - mid_sh_z
        v = point.visibility
        row.extend([x, y, z, v])

    return row, res


def create_landmark_dataframe():
    cols = []
    for i in range(33):  # 33 pose landmarks
        cols.extend([
            f"lm{i}_x", 
            f"lm{i}_y", 
            f"lm{i}_z", 
            f"lm{i}_v"
        ])
    
    df = pd.DataFrame(columns=cols)
    return df

def add_row_to_dataframe(df, row):
    """Append a new row to the dataframe"""
    df.loc[len(df)] = row
    return df

frame_count = 0
df = create_landmark_dataframe()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    frame_count += 1
    if frame_count % 10 != 0:
        row = extract_landmark_data(frame)
        if row is not None:
            df = add_row_to_dataframe(df, row)
            print("Row added. Total rows:", len(df))
    
    if res.pose_landmarks:
        draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Pose viewer", frame)

    # if esc pressed, exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

df.to_csv("bad3.csv", index=False)
print("Saved bad3.csv")
