import cv2
import mediapipe as mp
import numpy as np
from joblib import load

model = load("posture_rf_model.joblib")
scaler = load("posture_scaler.joblib")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, enable_segmentation=False)
draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)  # change to 0 if needed
print("Camera opened:", cap.isOpened())

def extract_landmarks_and_result(image):
    """
    runs Mediapipe ONCE and returns flattened landmark row and result object
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if not res.pose_landmarks:
        return None, res

    lm = res.pose_landmarks.landmark

    # normalize using hips
    L_HIP, R_HIP = 23, 24
    hip_x = (lm[L_HIP].x + lm[R_HIP].x) / 2
    hip_y = (lm[L_HIP].y + lm[R_HIP].y) / 2
    hip_z = (lm[L_HIP].z + lm[R_HIP].z) / 2

    row = []
    for point in lm:
        row.extend([
            point.x - hip_x,
            point.y - hip_y,
            point.z - hip_z,
            point.visibility
        ])

    return np.array(row, dtype=float), res


while True:
    ok, frame = cap.read()
    if not ok:
        break

    row, res = extract_landmarks_and_result(frame)

    if row is not None:
        X = scaler.transform([row])

        # prediction
        pred_label = model.predict(X)[0]
        pred_prob = model.predict_proba(X)[0].max()

        if pred_label == "good":
            label = f"GOOD POSTURE ({pred_prob:.2f})"
            color = (0, 255, 0)
        else:
            label = f"BAD POSTURE ({pred_prob:.2f})"
            color = (0, 0, 255)

        cv2.putText(frame, label, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # draw skeleton
    if res.pose_landmarks:
        draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Realtime Posture Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()