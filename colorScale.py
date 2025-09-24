import cv2
import numpy as np
import os
import glob

# === CONFIGURATION ===
chessboard_size = (9, 6)  # (columns, rows) — inner corners!
square_size = 23.5  # millimeters
calibration_images_folder = 'calibrationFrames'
save_file = 'calibration_data.npz'

# === PREPARE OBJECT POINTS ===
# Example: (0,0,0), (1,0,0), (2,0,0) ... (8,5,0) scaled by square_size
objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0],
                       0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image plane

# === LOAD IMAGES ===
images = glob.glob(os.path.join(calibration_images_folder, '*.jpg'))

if len(images) == 0:
    print("No images found in folder:", calibration_images_folder)
    exit()

for fname in images:
    img = cv2.imread(fname)
    print(img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:

        h, w = gray.shape[:2] # Get image dimensions
        
        # Reshape the corners array for easier access
        corners_reshaped = corners.reshape(-1, 2)
        
        # Assert that all x-coordinates are within the width
        assert np.all(corners_reshaped[:, 0] >= 0) and np.all(corners_reshaped[:, 0] < w), "Corner x-coordinates are out of bounds!"
        
        # Assert that all y-coordinates are within the height
        assert np.all(corners_reshaped[:, 1] >= 0) and np.all(corners_reshaped[:, 1] < h), "Corner y-coordinates are out of bounds!"

        objpoints.append(objp.reshape(54,1,3))
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)  # Flatten from (54, 1, 2) to (54, 2)
    
        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(100)
    else:
        print(f"Corners not found in {fname}")

cv2.destroyAllWindows()

# === PERFORM CALIBRATION ===
img_shape = gray.shape[::-1]  # (width, height)
print("Number of valid images:", len(objpoints))
print("Object points shape:", np.array(objpoints).shape)
print("Image points shape:", np.array(imgpoints).shape)

if len(objpoints) == 0 or len(imgpoints) == 0:
    print("❌ No valid corner detections. Exiting...")
    exit()

print(f'objpoints_list{np.array(objpoints).shape}')
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_shape, None, None
)

print("\n=== Calibration Results ===")
print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs)

# === SAVE CALIBRATION RESULTS ===
np.savez(save_file,
         camera_matrix=camera_matrix,
         dist_coeffs=dist_coeffs,
         rvecs=rvecs,
         tvecs=tvecs)
print(f"\nCalibration data saved to {save_file}")