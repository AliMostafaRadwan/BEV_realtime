import cv2
import numpy as np

# Chessboard settings
chessboard_size = (7, 7)  # Chessboard corners dimensions
square_size = 1.0  # Chessboard square size

# Prepare object points based on the chessboard dimensions
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Input source selection
use_camera = False  # Set to False to use an image file instead of camera

if use_camera:
    # Camera setup
    cap = cv2.VideoCapture(1)  # Use 0 for the default camera
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        exit()
else:
    # Image file setup
    image_path = "image.jpg"  # Replace with the path to your image file
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image file: {image_path}")
        exit()

# Try to find the chessboard in the first few frames to calibrate
found = False
while not found:
    if use_camera:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        # Corners found
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        found = True
    elif not use_camera:
        print("Chessboard not found in the image.")
        exit()

# If calibration is successful, set up the transformation
if found:
    h_img, w_img = frame.shape[:2]

    # Source points (corners of the chessboard in the image)
    src_points = np.float32([corners[0], corners[chessboard_size[0]-1], corners[-1], corners[-chessboard_size[0]]]).reshape(-1, 2)

    # Destination points (desired positions of the corners)
    # Increase the output image size for a wider BEV field
    output_size_width = int(w_img * 2.5)  # Adjust this factor to increase field of view width
    output_size_height = int(h_img * 2.5)  # Adjust this factor to increase field of view height
    
    dst_points = np.float32([
        [w_img * 0.2, h_img * 0.2],  # top-left, moved further out
        [w_img * 1.3, h_img * 0.2],  # top-right, moved further out
        [w_img * 1.3, h_img * 1.3],  # bottom-right, moved further out
        [w_img * 0.2, h_img * 1.3]   # bottom-left, moved further out
    ])
    


    # Compute the homography matrix
    H, _ = cv2.findHomography(src_points, dst_points)

    # Use the homography to transform the entire image, specifying the new output size
    warped_image = cv2.warpPerspective(frame, H, (output_size_width, output_size_height))

    # Optionally resize for display purposes
    warped_image_resized = cv2.resize(warped_image, (400, 400))

    try:
        cv2.imshow('Warped Bird\'s Eye View', warped_image_resized)
        if not use_camera:
            cv2.waitKey(0)  # Wait indefinitely if using an image
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                warped_image = cv2.warpPerspective(frame, H, (output_size_width, output_size_height))
                cv2.imshow('Warped Bird\'s Eye View', cv2.resize(warped_image, (output_size_width // 2, output_size_height // 2)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        if use_camera:
            cap.release()
        cv2.destroyAllWindows()
