import numpy as np
import cv2
import cv2.aruco as aruco

# Load camera calibration
camera_matrix = np.load("camera_matrix.npy")
dist_coeffs = np.load("dist_coeffs.npy")

# ArUco setup
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# Marker IDs
corner_ids = [0, 1, 3, 2]  # Order: top-left, top-right, bottom-left, bottom-right
bot_marker_id = 4

# Real-world positions (in cm)
real_points = np.array([
    [0, 0],      # ID 0
    [30, 0],     # ID 1
    [0, 20],     # ID 3
    [30, 20],    # ID 2
], dtype=np.float32)

# Use RealSense or any USB camera
cap = cv2.VideoCapture(2)  # Change to 0,1,2 if 3 doesn't work

if not cap.isOpened():
    print("Camera not accessible.")
    exit()

print("Running... Press Q to quit.")
last_H = None  # Store last known good homography

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame received.")
            break

        frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

        # Detect ArUco markers
        corners, ids, _ = aruco.detectMarkers(frame_undistorted, aruco_dict, parameters=parameters)

        image_points = [None] * 4

        if ids is not None:
            ids = ids.flatten()

            # Match each marker to the correct real-world point
            for i, marker_id in enumerate(corner_ids):
                if marker_id in ids:
                    idx = np.where(ids == marker_id)[0][0]
                    c = corners[idx][0]
                    center = np.mean(c, axis=0)
                    image_points[i] = center

            # If all 4 corner markers found
            if all(p is not None for p in image_points):
                image_points = np.array(image_points, dtype=np.float32)
                H = cv2.getPerspectiveTransform(image_points, real_points)
                last_H = H  # Update cached homography

        # Draw grid if homography exists
        if last_H is not None:
            # Draw grid points
            for x in range(0, 31, 5):
                for y in range(0, 21, 5):
                    pt = np.array([[[x, y]]], dtype=np.float32)
                    pt_img = cv2.perspectiveTransform(pt, np.linalg.inv(last_H))
                    cv2.circle(frame_undistorted, tuple(np.int32(pt_img[0][0])), 2, (0, 0, 0), -1)

            # Vertical lines
            for x in range(0, 31, 5):
                pt1 = np.array([[[x, 0]]], dtype=np.float32)
                pt2 = np.array([[[x, 20]]], dtype=np.float32)
                p1 = cv2.perspectiveTransform(pt1, np.linalg.inv(last_H))[0][0]
                p2 = cv2.perspectiveTransform(pt2, np.linalg.inv(last_H))[0][0]
                cv2.line(frame_undistorted, tuple(np.int32(p1)), tuple(np.int32(p2)), (0, 0, 0), 1)

            # Horizontal lines
            for y in range(0, 21, 5):
                pt1 = np.array([[[0, y]]], dtype=np.float32)
                pt2 = np.array([[[30, y]]], dtype=np.float32)
                p1 = cv2.perspectiveTransform(pt1, np.linalg.inv(last_H))[0][0]
                p2 = cv2.perspectiveTransform(pt2, np.linalg.inv(last_H))[0][0]
                cv2.line(frame_undistorted, tuple(np.int32(p1)), tuple(np.int32(p2)), (0, 0, 0), 1)

            # Bot marker (ID 4)
            if ids is not None and bot_marker_id in ids:
                idx = np.where(ids == bot_marker_id)[0][0]
                c = corners[idx][0]
                bot_center = np.mean(c, axis=0)
                bot_pos = cv2.perspectiveTransform(np.array([[bot_center]], dtype=np.float32), last_H)
                x, y = bot_pos[0][0]
                cv2.circle(frame_undistorted, tuple(np.int32(bot_center)), 5, (0, 255, 0), -1)
                cv2.putText(frame_undistorted, f"({x:.1f}, {y:.1f})",
                            (int(bot_center[0]) + 10, int(bot_center[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw marker boxes
        if ids is not None:
            aruco.drawDetectedMarkers(frame_undistorted, corners, ids)

        # Show output
        cv2.imshow("ArUco Coordinate System", frame_undistorted)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
