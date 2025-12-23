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
    Runs Mediapipe ONCE and returns flattened landmark row and result object
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if not res.pose_landmarks:
        return None, res

    lm = res.pose_landmarks.landmark

    # normalize to mid-shoulder
    L_SH, R_SH = 11, 12
    mid_sh_x = (lm[L_SH].x + lm[R_SH].x) / 2
    mid_sh_y = (lm[L_SH].y + lm[R_SH].y) / 2
    mid_sh_z = (lm[L_SH].z + lm[R_SH].z) / 2

    row = []
    for point in lm:
        row.extend([
            point.x - mid_sh_x,
            point.y - mid_sh_y,
            point.z - mid_sh_z,
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
        pred_proba = model.predict_proba(X)[0]
        
        # debug info to see what model is thinking
        good_prob = pred_proba[1]  # assuming 1 = good
        bad_prob = pred_proba[0]   # assuming 0 = bad

        if pred_label == 1:  # good posture
            label = f"GOOD POSTURE ({good_prob:.2f})"
            color = (0, 255, 0)
        else:  # bad posture
            label = f"BAD POSTURE ({bad_prob:.2f})"
            color = (0, 0, 255)

        cv2.putText(frame, label, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # show probability distribution for debugging
        debug_text = f"Good: {good_prob:.2f} | Bad: {bad_prob:.2f}"
        cv2.putText(frame, debug_text, (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # draw skeleton
    if res.pose_landmarks:
        draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Realtime Posture Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()