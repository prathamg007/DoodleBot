import cv2
import numpy as np
import time

# Checkerboard dimensions (number of **inner** corners)
CHECKERBOARD = (9, 6)

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (e.g., (0,0,0), (1,0,0), ..., (8,5,0))
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Storage for points
objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image plane

# Open RealSense or USB camera
cap = cv2.VideoCapture(3)  # Change to 0, 1, 2... as needed

if not cap.isOpened():
    print("❌ Camera not accessible.")
    exit()

print("📷 Move the checkerboard into view... Press 'q' to stop early.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ No frame received.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret_corners:
            objpoints.append(objp)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_refined)

            cv2.drawChessboardCorners(frame, CHECKERBOARD, corners_refined, ret_corners)
            print(f"✅ Captured {len(objpoints)} good frame(s)")
            time.sleep(1.5)  # Wait before capturing next

        cv2.imshow("Calibration (press q to stop)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or len(objpoints) >= 20:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

# --- Camera Calibration ---
print("\n📐 Calibrating...")

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("✅ Calibration complete.")
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)

# Save results
np.save("camera_matrix.npy", camera_matrix)
np.save("dist_coeffs.npy", dist_coeffs)
print("💾 Saved 'camera_matrix.npy' and 'dist_coeffs.npy'")
