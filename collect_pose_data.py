import cv2
import mediapipe as mp
import pandas as pd

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

    # normalize on hip center
    L_HIP = 23
    R_HIP = 24
    hip_x = (lm[L_HIP].x + lm[R_HIP].x) / 2
    hip_y = (lm[L_HIP].y + lm[R_HIP].y) / 2
    hip_z = (lm[L_HIP].z + lm[R_HIP].z) / 2

    row = []
    for i in range(len(lm)):
        x = lm[i].x - hip_x
        y = lm[i].y - hip_y
        z = lm[i].z - hip_z
        v = lm[i].visibility

        row.extend([x, y, z, v])

    return row


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
