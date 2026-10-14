import cv2
import mediapipe as mp

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        print("Camera could not be opened.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand
    result = hands.process(rgb_frame)

    fingers = 0

    if result.multi_hand_landmarks:

        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks = hand_landmarks.landmark

        # Draw hand
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # Thumb
        if landmarks[4].x < landmarks[3].x:
            fingers += 1

        # Index
        if landmarks[8].y < landmarks[6].y:
            fingers += 1

        # Middle
        if landmarks[12].y < landmarks[10].y:
            fingers += 1

        # Ring
        if landmarks[16].y < landmarks[14].y:
            fingers += 1

        # Little
        if landmarks[20].y < landmarks[18].y:
            fingers += 1

    # Display title
    cv2.putText(
        frame,
        "FINGER COUNTER",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    # Display finger count
    cv2.putText(
        frame,
        "FINGERS: " + str(fingers),
        (30, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        4
    )

    # Show camera
    cv2.imshow("AI Finger Counter", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()