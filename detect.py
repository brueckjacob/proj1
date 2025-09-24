import cv2
import numpy as np

# === Load Calibration Data ===
with np.load('calibration_data.npz') as data:
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

# === Set up AprilTag Detector ===
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# === Start Webcam ===
cap = cv2.VideoCapture(0)
tag_size = 0.172  # meters

# Variables to store clicked points for measurement
clicked_points = []

# Mouse callback to select points
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 2:
            clicked_points.append((x, y))
            print(f"Point {len(clicked_points)}: {x}, {y}")

cv2.namedWindow('AprilTag Detection')
cv2.setMouseCallback('AprilTag Detection', click_event)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    undistorted = cv2.undistort(gray, camera_matrix, dist_coeffs)

    corners, ids, _ = detector.detectMarkers(undistorted)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, tag_size, camera_matrix, dist_coeffs)
        
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, tag_size / 2)
            x, y, z = tvec[0]
            text = f"X: {x:.2f}m Y: {y:.2f}m Z: {z:.2f}m"
            y_offset = 30 * i + 30
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Use first tag as reference plane
        plane_rvec, plane_tvec = rvecs[0], tvecs[0]
        R_plane, _ = cv2.Rodrigues(plane_rvec)
        t_plane = plane_tvec.reshape(3)

        # If two points are clicked, calculate distance
        if len(clicked_points) == 2:
            pts_world = []
            for pt in clicked_points:
                u, v = pt
                # Convert to normalized camera coordinates
                x_norm = (u - camera_matrix[0,2]) / camera_matrix[0,0]
                y_norm = (v - camera_matrix[1,2]) / camera_matrix[1,1]
                ray = np.array([x_norm, y_norm, 1.0])

                # Transform ray to world coordinates
                ray_world = R_plane @ ray
                s = -t_plane[2] / ray_world[2]  # intersect with plane z=0
                pt_world = t_plane + s * ray_world
                pts_world.append(pt_world)

            # Compute distance
            dist = np.linalg.norm(pts_world[0] - pts_world[1])
            cv2.putText(frame, f"Measured distance: {dist:.3f} m", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Draw a line between clicked points
            cv2.line(frame, clicked_points[0], clicked_points[1], (0, 255, 255), 2)

    else:
        cv2.putText(frame, "No AprilTags detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('AprilTag Detection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):  # reset clicked points
        clicked_points = []

cap.release()
cv2.destroyAllWindows()
