import cv2
import numpy as np

# Chessboard settings
chessboard_size = (7, 7)
square_size = 1.0

# Prepare object points based on the chessboard dimensions
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Camera setup
cap = cv2.VideoCapture(0)  # Use 0 for the default camera
if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

# Try to find the chessboard in the first few frames to calibrate
found = False
while not found:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        # Corners found
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        found = True

# If calibration is successful, set up the transformation
if found:
    img_size = gray.shape[::-1]
    h, w = img_size
    src_points = np.float32([corners[0], corners[chessboard_size[0]-1], corners[-1], corners[-chessboard_size[0]]])
    dst_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    H, _ = cv2.findHomography(src_points, dst_points)
    print("Homography matrix:")
    print(H)
else:
    print("Chessboard not found, check your camera and chessboard setup.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Apply the transformation in real-time
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        warped_image = cv2.warpPerspective(frame, H, (w, h))
        cv2.imshow('Warped Image', warped_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
