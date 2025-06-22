import cv2
import mediapipe as mp
import csv
import os
label = "hello"

os.makedirs("dataset", exist_ok=True)
csv_path = os.path.join("dataset", f"dataset_{label}.csv")
write_header = not os.path.exists(csv_path)
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)
mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

mp_drawing = mp.solutions.drawing_utils

POSE_LANDMARKS_TO_SHOW = [
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
    mp.solutions.pose.PoseLandmark.LEFT_WRIST,
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
    mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
]

FACE_LANDMARKS_TO_SHOW = {
    "nose_tip": 1,
    "left_eye_inner": 133,
    "left_eye_outer": 33,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
    "left_ear": 234,
    "right_ear": 454,
}

def create_header():
    header = []
    for i in range(2):
        for j in range(21):
            header += [f'hand{i}_{j}_x', f'hand{i}_{j}_y', f'hand{i}_{j}_z']
    for lm_id in POSE_LANDMARKS_TO_SHOW:
        header += [f'pose_{lm_id.name}_x', f'pose_{lm_id.name}_y', f'pose_{lm_id.name}_z']
    for name in FACE_LANDMARKS_TO_SHOW.keys():
        header += [f'face_{name}_x', f'face_{name}_y', f'face_{name}_z']
    header.append('label')
    return header

cap = cv2.VideoCapture(0)
with open(csv_path, 'a', newline='') as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(create_header())

    print("[INFO] Press 's' to save, 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hands_results = mp_hands.process(rgb)
        pose_results = mp_pose.process(rgb)
        face_results = mp_face_mesh.process(rgb)

        h, w, _ = frame.shape
        row = []

        if hands_results.multi_hand_landmarks:
            for hand_idx in range(2):
                if hand_idx < len(hands_results.multi_hand_landmarks):
                    hand_landmarks = hands_results.multi_hand_landmarks[hand_idx]
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])
                else:
                    row.extend([-1, -1, -1]*21)
        else:
            row.extend([-1, -1, -1]*21*2)

        if pose_results.pose_landmarks:
            for lm_id in POSE_LANDMARKS_TO_SHOW:
                lm = pose_results.pose_landmarks.landmark[lm_id]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 7, (0, 0, 255), cv2.FILLED)
                row.extend([lm.x, lm.y, lm.z])
        else:
            row.extend([-1, -1, -1]*len(POSE_LANDMARKS_TO_SHOW))

        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            for name, idx in FACE_LANDMARKS_TO_SHOW.items():
                lm = face_landmarks.landmark[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                cv2.putText(frame, name, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                row.extend([lm.x, lm.y, lm.z])
        else:
            row.extend([-1, -1, -1]*len(FACE_LANDMARKS_TO_SHOW))

        cv2.putText(frame, "Press 's' to save, 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Landmark Viewer", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            writer.writerow(row + [label])
            print(f"[SAVED] Label: {label}")
        elif key == ord('q'):
            print("[EXIT]")
            break

cap.release()
cv2.destroyAllWindows()
