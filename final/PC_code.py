import numpy as np
import cv2
import cv2.aruco as aruco
import socket
import json
import time
import threading

# --- Configuration Parameters (PC) ---
# Camera Calibration Files (Adjust paths if necessary)
CAMERA_MATRIX_PATH_PC ="C:/Users/Jugal pahuja/OneDrive/Desktop/camera_matrix_updated.npy"
DIST_COEFFS_PATH_PC = "C:/Users/Jugal pahuja/OneDrive/Desktop/dist_coeffs_updated.npy"

# ArUco Setup
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
ARUCO_PARAMETERS = aruco.DetectorParameters()

# Marker IDs
PLANE_CORNER_IDS = [6, 5, 9, 8]   # Order: top-left, top-right, bottom-left, bottom-right
BOT_MARKER_ID = 15                 # Marker on top of your robot

# Real-world positions of plane markers (in cm)
REAL_WORLD_PLANE_POINTS_CM = np.array([
    [0, 0],     # ID 0 (Top-left physical corner)
    [70, 0],    # ID 1 (Top-right physical corner)
    [0, 55],    # ID 3 (Bottom-left physical corner)
    [70, 55],   # ID 2 (Bottom-right physical corner)
], dtype=np.float32)

PLANE_WIDTH_CM = REAL_WORLD_PLANE_POINTS_CM[1][0] - REAL_WORLD_PLANE_POINTS_CM[0][0] # 70cm
PLANE_HEIGHT_CM = REAL_WORLD_PLANE_POINTS_CM[3][1] - REAL_WORLD_PLANE_POINTS_CM[0][1] # 55cm

# Camera Index
CAMERA_INDEX_PC = 1

# UDP Communication
ROBOT_UDP_IP = "10.152.206.195"
ROBOT_UDP_PORT_SEND_COMMANDS = 5006

# --- FIXED Corner-Based Steering Parameters ---
BASE_SPEED = 11  # Increased base speed
TURN_SPEED = 9  # Separate speed for turning
FAST_TURN_SPEED = 9  # Speed for fast turns

# FIXED thresholds - these were backwards in original code
NORMAL_TURN_THRESHOLD = 5.0    # CM difference for normal turn (larger threshold)
FAST_TURN_THRESHOLD = 15.0     # CM difference for fast turn (even larger threshold)
WAYPOINT_REACHED_THRESHOLD_CM = 3.0 # Distance to consider waypoint reached (increased)
WAYPOINT_OVERSHOOT_THRESHOLD_CM = 5.0 # If robot goes this much *past* waypoint, advance it.

# Command timing
COMMAND_INTERVAL = 0.15  # Send commands every 150ms (reduced frequency)

# --- Global Variables for State Management ---
camera_matrix_pc = None
dist_coeffs_pc = None
last_homography_matrix = None
last_inv_homography_matrix = None

# Robot's marker corner positions (in real-world CM)
robot_center_cm = None
robot_left_front_cm = None
robot_right_front_cm = None

# User-drawn path
drawing_pixel_points = []
user_drawing_cm = []

drawing_in_progress = False
path_tracing_active = False
current_waypoint_index = 0
current_pen_state = 0 # 0 for up, 1 for down

last_command_time = time.time()

# --- UI Button Definitions ---
# Each button is defined by (text, x_start, y_start, width, height, action_key)
# These positions are relative to the top-right corner of the display_frame
BUTTONS = {
    "QUIT": {"text": "Quit (Q)", "x_offset": -100, "y_offset": 10, "width": 90, "height": 30, "action": "quit"},
    "SAVE": {"text": "Save (S)", "x_offset": -100, "y_offset": 50, "width": 90, "height": 30, "action": "save"},
    "CLEAR": {"text": "Clear (C)", "x_offset": -100, "y_offset": 90, "width": 90, "height": 30, "action": "clear"}
}
# Global variable to signal application exit
app_should_quit = False

# --- UDP Socket Setup ---
robot_sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(f"PC UDP client ready to send commands to RPi at {ROBOT_UDP_IP}:{ROBOT_UDP_PORT_SEND_COMMANDS}")

# --- Helper Functions ---

def load_camera_calibration_pc():
    global camera_matrix_pc, dist_coeffs_pc
    try:
        camera_matrix_pc = np.load(CAMERA_MATRIX_PATH_PC)
        dist_coeffs_pc = np.load(DIST_COEFFS_PATH_PC)
        print("PC: Camera calibration loaded successfully.")
    except FileNotFoundError:
        print(f"PC Error: Camera calibration files not found.")
        exit()

def send_command_to_rpi(command_type, speed, pen_state):
    """
    Sends movement commands to the RPi.
    command_type: "forward", "left", "right", "fast_left", "fast_right", "stop"
    """
    global current_pen_state
    
    command_data = {
        "command": command_type,
        "speed": int(speed),
        "pen_state": int(pen_state)
    }
    json_message = json.dumps(command_data)

    try:
        robot_sock_send.sendto(json_message.encode('utf-8'), (ROBOT_UDP_IP, ROBOT_UDP_PORT_SEND_COMMANDS))
        current_pen_state = pen_state
        # print(f"PC: Sent command: {json_message}") # Too verbose, uncomment for debugging
    except Exception as e:
        print(f"PC Error: Failed to send UDP command to RPi: {e}")

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points in CM - FIXED"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def is_robot_within_boundaries(robot_center_cm, margin_cm=3.0):
    """
    Check if robot is within the defined coordinate system boundaries.
    
    Args:
        robot_center_cm: [x, y] position of robot center in CM
        margin_cm: Safety margin in CM from the boundaries
    
    Returns:
        bool: True if robot is within boundaries, False otherwise
    """
    if robot_center_cm is None:
        return False
    
    x, y = robot_center_cm[0], robot_center_cm[1]
    
    # Define boundaries with margin
    min_x = 0 + margin_cm
    max_x = PLANE_WIDTH_CM - margin_cm
    min_y = 0 + margin_cm
    max_y = PLANE_HEIGHT_CM - margin_cm
    
    # Check if robot is within boundaries
    within_bounds = (min_x <= x <= max_x) and (min_y <= y <= max_y)
    
    if not within_bounds:
        print(f"WARNING: Robot at ({x:.2f}, {y:.2f}) is outside boundaries!")
        print(f"Allowed range: X({min_x:.1f} to {max_x:.1f}), Y({min_y:.1f} to {max_y:.1f})")
    
    return within_bounds

def calculate_robot_heading_angle(left_front_cm, right_front_cm):
    """
    Calculate the robot's heading angle in degrees (0-360).
    0 degrees = facing positive X direction (right)
    90 degrees = facing positive Y direction (down) (due to image coordinate system)
    """
    robot_front_midpoint_cm = (np.array(left_front_cm) + np.array(right_front_cm)) / 2
    heading_vector = robot_front_midpoint_cm - np.array(robot_center_cm)
    
    angle_rad = np.arctan2(heading_vector[1], heading_vector[0])
    angle_deg = np.degrees(angle_rad)
    
    # Normalize to 0-360 degrees
    if angle_deg < 0:
        angle_deg += 360
    
    return angle_deg


def calculate_angle_to_goal(robot_center_cm, goal_cm):
    """
    Calculate the angle from robot center to goal in degrees (0-360).
    """
    goal_vector = np.array(goal_cm) - np.array(robot_center_cm)
    angle_rad = np.arctan2(goal_vector[1], goal_vector[0])
    angle_deg = np.degrees(angle_rad)
    
    # Normalize to 0-360 degrees
    if angle_deg < 0:
        angle_deg += 360
    
    return angle_deg

def angle_difference(angle1, angle2):
    """
    Calculate the shortest angular difference between two angles.
    Returns positive if need to turn right, negative if need to turn left.
    """
    diff = angle2 - angle1
    
    # Normalize to [-180, 180]
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    
    return diff

def moveTowardsGoal(right_front_cm, left_front_cm, center_cm, goal_cm, base_speed):
    """
    IMPROVED steering logic using both distance-based and angle-based navigation,
    with an added check for overshooting a waypoint.
    Returns "reached", "moving", "out_of_bounds", or "passed_waypoint".
    """
    # BOUNDARY CHECK - Stop robot if outside coordinate system
    if not is_robot_within_boundaries(center_cm):
        send_command_to_rpi("stop", 0, 0) # Pen up on emergency stop
        print("EMERGENCY STOP: Robot is outside coordinate system boundaries!")
        return "out_of_bounds"
    
    # Calculate distance to goal
    center_distance = calculate_distance(center_cm, goal_cm)
    print(f"Distance to goal: {center_distance:.2f} cm")
    
    # Check if waypoint is reached
    if center_distance < WAYPOINT_REACHED_THRESHOLD_CM:
        send_command_to_rpi("stop", 0, 1) # Keep pen down for next segment if drawing
        return "reached"
    
    # NEW LOGIC: Check if the robot has significantly overshot the waypoint based on its heading
    # This prevents circling when the robot passes a waypoint without registering "reached".
    robot_heading = calculate_robot_heading_angle(left_front_cm, right_front_cm)
    angle_robot_to_goal = calculate_angle_to_goal(center_cm, goal_cm)
    
    # If the robot is already facing away from the goal, and is a certain distance past it
    # this suggests it has overshot.
    # Calculate the angle of the vector from the goal to the robot's current position.
    angle_goal_to_robot = calculate_angle_to_goal(goal_cm, center_cm)
    
    # If the robot's heading is roughly opposite to the vector from the goal to the robot,
    # it means the robot is heading away from the goal.
    # And if the distance is greater than the overshoot threshold, it has likely overshot.
    angle_diff_overshoot = abs(angle_difference(robot_heading, angle_goal_to_robot))
    
    if center_distance > WAYPOINT_REACHED_THRESHOLD_CM and angle_diff_overshoot < 45 and \
       center_distance > WAYPOINT_OVERSHOOT_THRESHOLD_CM:
        # The robot is moving away from the goal (angle_diff_overshoot is small)
        # and it's further than the overshoot threshold.
        print(f"WARNING: Robot appears to have overshot waypoint! Current distance {center_distance:.2f} cm.")
        send_command_to_rpi("stop", 0, 1) # Keep pen down for next segment if drawing
        return "passed_waypoint"

    # IMPROVED NAVIGATION: Use both corner distances AND angles
    
    # Method 1: Corner distance comparison (original logic - FIXED)
    right_distance = calculate_distance(right_front_cm, goal_cm)
    left_distance = calculate_distance(left_front_cm, goal_cm)
    distance_diff = right_distance - left_distance  # Positive = right is farther, need to turn left
    
    # Method 2: Angle-based navigation
    angle_error = angle_difference(robot_heading, angle_robot_to_goal)
    
    # print(f"Robot heading: {robot_heading:.1f}°, Goal angle: {angle_robot_to_goal:.1f}°, Angle Error: {angle_error:.1f}°")
    # print(f"Distance diff: {distance_diff:.2f} (R:{right_distance:.2f} - L:{left_distance:.2f})")
    
    # Combine both methods for more robust navigation
    # Large angle errors take priority for initial alignment
    if abs(angle_error) > 40:  # Robot is significantly off-course angle-wise
        if angle_error > 0:  # Need to turn right
            send_command_to_rpi("fast_right", FAST_TURN_SPEED, 1)
            # print(f"FAST RIGHT (large angle error: {angle_error:.1f}°)")
        else:  # Need to turn left
            send_command_to_rpi("fast_left", FAST_TURN_SPEED, 1)
            # print(f"FAST LEFT (large angle error: {angle_error:.1f}°)")
    
    elif abs(angle_error) > 15:  # Moderate angle error
        if angle_error > 0:
            send_command_to_rpi("right", TURN_SPEED*0.8, 1)
            # print(f"RIGHT (moderate angle error: {angle_error:.1f}°)")
        else:
            send_command_to_rpi("left", TURN_SPEED*0.8, 1)
            # print(f"LEFT (moderate angle error: {angle_error:.1f}°)")
    
    # For small angle errors, use distance-based fine-tuning and forward motion
    elif abs(distance_diff) > FAST_TURN_THRESHOLD:
        if distance_diff > 0:  # Right corner is farther from goal, turn left
            send_command_to_rpi("fast_left", FAST_TURN_SPEED*0.9, 1)
            # print(f"FAST LEFT (large distance diff: {distance_diff:.2f})")
        else: # Left corner is farther from goal, turn right
            send_command_to_rpi("fast_right", FAST_TURN_SPEED*0.9, 1)
            # print(f"FAST RIGHT (large distance diff: {distance_diff:.2f})")
    
    elif abs(distance_diff) > NORMAL_TURN_THRESHOLD:
        if distance_diff > 0: # Right corner is farther from goal, turn left
            send_command_to_rpi("left", TURN_SPEED*0.7, 1)
            # print(f"LEFT (normal distance diff: {distance_diff:.2f})")
        else: # Left corner is farther from goal, turn right
            send_command_to_rpi("right", TURN_SPEED*0.7, 1)
            # print(f"RIGHT (normal distance diff: {distance_diff:.2f})")
    
    else:
        # Robot is well-aligned, move forward
        send_command_to_rpi("forward", base_speed, 1)
        # print("FORWARD (well aligned)")
    
    return "moving"

def get_robot_corner_positions_cm(bot_marker_corners, homography_inv):
    """
    Calculate the real-world CM positions of the robot's center and front corners
    from the ArUco marker corners in pixel coordinates.
    
    Assumes ArUco corners are ordered: top-left (0), top-right (1), bottom-right (2), bottom-left (3)
    Robot's front is defined as the edge between corners 0 and 1.
    """
    # Get center of the robot marker
    center_px = np.mean(bot_marker_corners, axis=0)
    
    # Get front corners (top-left and top-right of the marker)
    left_front_px = bot_marker_corners[0]   # top-left corner
    right_front_px = bot_marker_corners[1]  # top-right corner
    
    # Convert pixel positions to real-world CM using inverse homography
    center_cm_transform = cv2.perspectiveTransform(
        np.array([[center_px]], dtype=np.float32), homography_inv)[0][0]
    
    left_front_cm_transform = cv2.perspectiveTransform(
        np.array([[left_front_px]], dtype=np.float32), homography_inv)[0][0]
    
    right_front_cm_transform = cv2.perspectiveTransform(
        np.array([[right_front_px]], dtype=np.float32), homography_inv)[0][0]
    
    return (center_cm_transform, left_front_cm_transform, right_front_cm_transform)

def mouse_callback(event, x, y, flags, param):
    """Callback function for mouse events on the OpenCV window for drawing and UI buttons."""
    global drawing_in_progress, drawing_pixel_points, path_tracing_active, app_should_quit, user_drawing_cm, current_waypoint_index

    # Get frame dimensions to calculate button positions
    frame_width = param[0]
    frame_height = param[1]

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check for button clicks first
        button_clicked = False
        for btn_name, btn_info in BUTTONS.items():
            btn_x_start = frame_width + btn_info["x_offset"]
            btn_y_start = btn_info["y_offset"]
            btn_x_end = btn_x_start + btn_info["width"]
            btn_y_end = btn_y_start + btn_info["height"]

            if btn_x_start <= x <= btn_x_end and btn_y_start <= y <= btn_y_end:
                button_clicked = True
                if btn_info["action"] == "quit":
                    app_should_quit = True
                    print("PC: 'Quit' button pressed. Application will terminate.")
                elif btn_info["action"] == "clear":
                    drawing_pixel_points = []
                    user_drawing_cm = []
                    path_tracing_active = False
                    current_waypoint_index = 0
                    send_command_to_rpi("stop", 0, 0) # Stop robot and lift pen
                    print("PC: 'Clear' button pressed. Drawing cleared and robot stopped.")
                elif btn_info["action"] == "save":
                    if last_inv_homography_matrix is not None and len(drawing_pixel_points) > 0:
                        user_drawing_cm = []
                        # Subsample the drawing points to reduce waypoints for smoother movement
                        # Limit to ~100 waypoints for path complexity
                        step = max(1, len(drawing_pixel_points) // 100)
                        for i in range(0, len(drawing_pixel_points), step):
                            px, py = drawing_pixel_points[i]
                            pt_px = np.array([[[px, py]]], dtype=np.float32)
                            pt_cm = cv2.perspectiveTransform(pt_px, last_inv_homography_matrix)[0][0]
                            user_drawing_cm.append([pt_cm[0], pt_cm[1]])
                        
                        # Ensure the very last point is always included
                        if (len(drawing_pixel_points) - 1) % step != 0:
                             px, py = drawing_pixel_points[-1]
                             pt_px = np.array([[[px, py]]], dtype=np.float32)
                             pt_cm = cv2.perspectiveTransform(pt_px, last_inv_homography_matrix)[0][0]
                             user_drawing_cm.append([pt_cm[0], pt_cm[1]])

                        path_tracing_active = False # Just save, don't start tracing immediately
                        current_waypoint_index = 0
                        print(f"PC: 'Save Path' button pressed. Converted drawing to {len(user_drawing_cm)} waypoints. Ready to trace.")
                    else:
                        print("PC: Cannot save drawing. Need plane markers and a drawn path.")
                break # Exit loop after finding a clicked button

        if not button_clicked:
            # If no button was clicked, then it's a drawing click
            drawing_in_progress = True # Keep for status display consistency
            # Removed: drawing_pixel_points = []
            drawing_pixel_points.append((x, y)) # Add the single discrete point
            path_tracing_active = False # Stop any active tracing
            send_command_to_rpi("stop", 0, 0) # Lift pen when starting to draw
            print(f"PC: Point added at ({x}, {y}). Total points: {len(drawing_pixel_points)}. Click 'Clear' to reset.")

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing_in_progress:
            # DO NOT add points on mouse move for discrete input.
            pass # No action here, effectively disables continuous drawing

    elif event == cv2.EVENT_LBUTTONUP:
        drawing_in_progress = False # End the "drawing session" for status display
        # print(f"PC: Drawing finished. {len(drawing_pixel_points)} points drawn.") # This message is better after a button press
        # print("Press Enter to convert to waypoints and start tracing, or 's' to trace existing waypoints.")


# --- Main Program ---
def main():
    global last_homography_matrix, last_inv_homography_matrix
    global robot_center_cm, robot_left_front_cm, robot_right_front_cm
    global drawing_pixel_points, user_drawing_cm, drawing_in_progress
    global path_tracing_active, current_waypoint_index, current_pen_state
    global last_command_time, app_should_quit

    load_camera_calibration_pc()

    # Initialize camera capture
    cap = cv2.VideoCapture(CAMERA_INDEX_PC)
    
    if not cap.isOpened():
        print(f"PC Error: Could not open camera at index {CAMERA_INDEX_PC}.")
        return

    cv2.namedWindow("Doodlebot PC Control - FIXED Corner Steering")
    # Pass frame dimensions to mouse callback for button positioning
    # We'll get actual frame dimensions after reading the first frame
    ret, frame = cap.read()
    if not ret:
        print("PC: Failed to grab initial frame for dimensions.")
        return
    frame_height, frame_width, _ = frame.shape
    cv2.setMouseCallback("Doodlebot PC Control - FIXED Corner Steering", lambda event, x, y, flags, param: mouse_callback(event, x, y, flags, (frame_width, frame_height)))

    print("\n--- Doodlebot PC Control - FIXED Corner Steering Method ---")
    print("Click discrete points on the camera feed window to draw a path.")
    print("Use the UI buttons on the right for 'Quit', 'Save Path', and 'Clear'.")
    print("Press 'Enter' to convert the drawn points to a path and start tracing.")
    print("Press 'S' to start tracing a previously saved path.")

    try:
        while not app_should_quit: # Loop until quit signal is received
            ret, frame = cap.read()
            if not ret:
                print("PC: Failed to grab frame.")
                time.sleep(0.1)
                continue

            display_frame = frame.copy()
            display_frame = cv2.undistort(display_frame, camera_matrix_pc, dist_coeffs_pc)

            corners, ids, rejected = aruco.detectMarkers(display_frame, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

            # --- Homography Calculation ---
            current_H = None
            if ids is not None:
                ids = ids.flatten()

                detected_plane_corners_by_id = {}
                for i, marker_id_val in enumerate(ids):
                    if marker_id_val in PLANE_CORNER_IDS:
                        center_x = np.mean(corners[i][0][:, 0])
                        center_y = np.mean(corners[i][0][:, 1])
                        detected_plane_corners_by_id[marker_id_val] = (center_x, center_y)

                if all(id_val in detected_plane_corners_by_id for id_val in PLANE_CORNER_IDS):
                    ordered_image_points_plane = []
                    for id_val in PLANE_CORNER_IDS:
                        ordered_image_points_plane.append(detected_plane_corners_by_id[id_val])
                    
                    image_points_plane_np = np.array(ordered_image_points_plane, dtype=np.float32)
                    
                    # Ensure REAL_WORLD_PLANE_POINTS_CM matches the order of image_points_plane_np
                    # based on PLANE_CORNER_IDS mapping
                    # REAL_WORLD_PLANE_POINTS_CM is already ordered according to PLANE_CORNER_IDS
                    H, _ = cv2.findHomography(REAL_WORLD_PLANE_POINTS_CM, image_points_plane_np)
                    current_H = H
                    last_homography_matrix = H
                    last_inv_homography_matrix = np.linalg.inv(H)
                else:
                    current_H = last_homography_matrix # Use last known homography if plane markers not fully detected

            # --- Draw Grid ---
            if last_homography_matrix is not None:
                # Draw grid points and lines
                for x_cm in range(0, int(PLANE_WIDTH_CM) + 1, 5):
                    for y_cm in range(0, int(PLANE_HEIGHT_CM) + 1, 5):
                        pt_cm = np.array([[[x_cm, y_cm]]], dtype=np.float32)
                        pt_img = cv2.perspectiveTransform(pt_cm, last_homography_matrix)[0][0]
                        cv2.circle(display_frame, tuple(np.int32(pt_img)), 2, (0, 0, 0), -1)

                # Vertical and horizontal lines
                for x_cm in range(0, int(PLANE_WIDTH_CM) + 1, 5):
                    pt1_cm = np.array([[[x_cm, 0]]], dtype=np.float32)
                    pt2_cm = np.array([[[x_cm, PLANE_HEIGHT_CM]]], dtype=np.float32)
                    p1_img = cv2.perspectiveTransform(pt1_cm, last_homography_matrix)[0][0]
                    p2_img = cv2.perspectiveTransform(pt2_cm, last_homography_matrix)[0][0]
                    cv2.line(display_frame, tuple(np.int32(p1_img)), tuple(np.int32(p2_img)), (0, 0, 0), 1)

                for y_cm in range(0, int(PLANE_HEIGHT_CM) + 1, 5):
                    pt1_cm = np.array([[[0, y_cm]]], dtype=np.float32)
                    pt2_cm = np.array([[[PLANE_WIDTH_CM, y_cm]]], dtype=np.float32)
                    p1_img = cv2.perspectiveTransform(pt1_cm, last_homography_matrix)[0][0]
                    p2_img = cv2.perspectiveTransform(pt2_cm, last_homography_matrix)[0][0]
                    cv2.line(display_frame, tuple(np.int32(p1_img)), tuple(np.int32(p2_img)), (0, 0, 0), 1)
            
            # Draw detected markers
            if ids is not None:
                aruco.drawDetectedMarkers(display_frame, corners, ids)
                
                # --- Robot Corner Position Detection ---
                if BOT_MARKER_ID in ids and last_inv_homography_matrix is not None:
                    idx = np.where(ids == BOT_MARKER_ID)[0][0]
                    bot_marker_corners = corners[idx][0]
                    
                    # Get robot's corner positions in real-world CM
                    robot_center_cm, robot_left_front_cm, robot_right_front_cm = \
                        get_robot_corner_positions_cm(bot_marker_corners, last_inv_homography_matrix)

                    # Draw robot visualization on PC screen
                    # Pixel coordinates for drawing are directly from `bot_marker_corners`
                    center_px = np.mean(bot_marker_corners, axis=0)
                    left_front_px = bot_marker_corners[0]
                    right_front_px = bot_marker_corners[1]
                    
                    # Draw robot center (green circle)
                    cv2.circle(display_frame, tuple(np.int32(center_px)), 5, (0, 255, 0), -1)
                    
                    # Draw front corners (red circles)
                    cv2.circle(display_frame, tuple(np.int32(left_front_px)), 4, (0, 0, 255), -1)
                    cv2.circle(display_frame, tuple(np.int32(right_front_px)), 4, (0, 0, 255), -1)
                    
                    # Draw front edge line (blue line)
                    cv2.line(display_frame, tuple(np.int32(left_front_px)), 
                             tuple(np.int32(right_front_px)), (255, 0, 0), 2)
                    
                    # Draw robot heading direction (green arrow)
                    if robot_center_cm is not None and robot_left_front_cm is not None and robot_right_front_cm is not None:
                        heading = calculate_robot_heading_angle(robot_left_front_cm, robot_right_front_cm)
                        arrow_length = 30
                        # Calculate end point of arrow based on heading angle in image coordinates
                        # (0 deg is right, 90 deg is down)
                        arrow_end_x = int(center_px[0] + arrow_length * np.cos(np.radians(heading)))
                        arrow_end_y = int(center_px[1] + arrow_length * np.sin(np.radians(heading)))
                        cv2.arrowedLine(display_frame, tuple(np.int32(center_px)), 
                                        (arrow_end_x, arrow_end_y), (0, 255, 0), 3, tipLength=0.3)

            # --- Visualizing User Drawing ---
            if len(drawing_pixel_points) > 1:
                for i in range(1, len(drawing_pixel_points)):
                    cv2.line(display_frame, drawing_pixel_points[i-1], drawing_pixel_points[i], (0, 255, 255), 2)
            
            # Draw individual points for clarity
            for p_px in drawing_pixel_points:
                cv2.circle(display_frame, p_px, 3, (255, 0, 0), -1)

            # Draw current target waypoint if tracing
            if (path_tracing_active and current_waypoint_index < len(user_drawing_cm) and 
                last_homography_matrix is not None):
                target_cm = user_drawing_cm[current_waypoint_index]
                target_px_array = cv2.perspectiveTransform(
                    np.array([[[target_cm[0], target_cm[1]]]], dtype=np.float32), 
                    last_homography_matrix)[0][0]
                target_px = tuple(np.int32(target_px_array))
                cv2.circle(display_frame, target_px, 8, (0, 255, 0), 3)   # Green circle for current target
                
                # Draw the path that has already been traced (optional)
                for i in range(current_waypoint_index):
                    if i > 0:
                        p1_cm = user_drawing_cm[i-1]
                        p2_cm = user_drawing_cm[i]
                        p1_px = cv2.perspectiveTransform(np.array([[[p1_cm[0], p1_cm[1]]]], dtype=np.float32), last_homography_matrix)[0][0]
                        p2_px = cv2.perspectiveTransform(np.array([[[p2_cm[0], p2_cm[1]]]], dtype=np.float32), last_homography_matrix)[0][0]
                        cv2.line(display_frame, tuple(np.int32(p1_px)), tuple(np.int32(p2_px)), (255, 0, 255), 2) # Magenta for traced path

            # --- FIXED Corner-Based Steering Control Logic ---
            if (path_tracing_active and robot_center_cm is not None and 
                robot_left_front_cm is not None and robot_right_front_cm is not None and
                current_waypoint_index < len(user_drawing_cm)):
                
                target_cm = user_drawing_cm[current_waypoint_index]
                
                # Send commands at controlled rate
                if time.time() - last_command_time > COMMAND_INTERVAL:
                    result = moveTowardsGoal(
                        robot_right_front_cm, robot_left_front_cm, 
                        robot_center_cm, target_cm, BASE_SPEED
                    )
                    last_command_time = time.time()
                    
                    if result == "reached" or result == "passed_waypoint":
                        print(f"PC: Waypoint {current_waypoint_index + 1}/{len(user_drawing_cm)} {'reached' if result == 'reached' else 'overshot'}.")
                        current_waypoint_index += 1
                        # Small pause between waypoints
                        time.sleep(0.1) # Give robot a moment to stop/re-orient
                    elif result == "out_of_bounds":
                        print("PC: Path tracing stopped - Robot went out of bounds!")
                        path_tracing_active = False
                        current_waypoint_index = 0

            elif path_tracing_active and current_waypoint_index >= len(user_drawing_cm):
                print("PC: Path tracing complete.")
                send_command_to_rpi("stop", 0, 0)   # Stop motors, pen up
                path_tracing_active = False
                current_waypoint_index = 0
            
            else:
                # Ensure robot is stopped when not tracing, but also check boundaries
                if robot_center_cm is not None and not is_robot_within_boundaries(robot_center_cm):
                    send_command_to_rpi("stop", 0, 0)
                elif current_pen_state != 0: # If pen is down and not tracing, lift it
                    send_command_to_rpi("stop", 0, 0)
                # If not tracing and pen is already up and in bounds, do nothing (idle)

            # --- Display Status Information ---
            y_offset = 30
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            text_color = (255, 255, 255) # White text for visibility
            
            # Status
            if last_homography_matrix is None:
                status = "Waiting for PLANE markers (0,1,2,3)"
            elif robot_center_cm is None:
                status = f"Waiting for ROBOT marker (ID {BOT_MARKER_ID})"
            elif path_tracing_active:
                status = f"Tracing Waypoint {current_waypoint_index + 1}/{len(user_drawing_cm)}"
            elif drawing_in_progress:
                status = f"Drawing ({len(drawing_pixel_points)} pts)"
            elif len(drawing_pixel_points) > 0 and len(user_drawing_cm) == 0:
                status = "Drawing READY (Press Enter)"
            else:
                status = "Idle - FIXED Corner Steering Mode"
            
            cv2.putText(display_frame, f"Status: {status}", (10, y_offset), font, font_scale, text_color, thickness)
            y_offset += 30

            # Robot position info
            if robot_center_cm is not None:
                cv2.putText(display_frame, f"Robot Center: ({robot_center_cm[0]:.1f}, {robot_center_cm[1]:.1f}) cm", 
                               (10, y_offset), font, font_scale, text_color, thickness)
                y_offset += 30
                
                # Show robot heading
                if robot_left_front_cm is not None and robot_right_front_cm is not None:
                    heading = calculate_robot_heading_angle(robot_left_front_cm, robot_right_front_cm)
                    cv2.putText(display_frame, f"Robot Heading: {heading:.1f}°", 
                                   (10, y_offset), font, font_scale, text_color, thickness)
                    y_offset += 30
            
            # Path info
            cv2.putText(display_frame, f"Drawn Points: {len(drawing_pixel_points)}", (10, y_offset), font, font_scale, text_color, thickness)
            y_offset += 30
            cv2.putText(display_frame, f"Waypoints: {len(user_drawing_cm)}", (10, y_offset), font, font_scale, text_color, thickness)

            # --- Draw UI Buttons ---
            for btn_name, btn_info in BUTTONS.items():
                btn_x_start = frame_width + btn_info["x_offset"]
                btn_y_start = btn_info["y_offset"]
                btn_x_end = btn_x_start + btn_info["width"]
                btn_y_end = btn_y_start + btn_info["height"]

                # Draw button rectangle
                cv2.rectangle(display_frame, (btn_x_start, btn_y_start), (btn_x_end, btn_y_end), (200, 200, 200), -1) # Gray background
                cv2.rectangle(display_frame, (btn_x_start, btn_y_start), (btn_x_end, btn_y_end), (50, 50, 50), 2)  # Dark border

                # Put button text
                text_size = cv2.getTextSize(btn_info["text"], font, font_scale, thickness)[0]
                text_x = btn_x_start + (btn_info["width"] - text_size[0]) // 2
                text_y = btn_y_start + (btn_info["height"] + text_size[1]) // 2
                cv2.putText(display_frame, btn_info["text"], (text_x, text_y), font, font_scale, (0, 0, 0), thickness) # Black text

            cv2.imshow("Doodlebot PC Control - FIXED Corner Steering", display_frame)

            # --- Key Handling (Only 'Enter' remains for keyboard) ---
            key = cv2.waitKey(1) & 0xFF
            if key == 13:   # Enter key
                if last_inv_homography_matrix is not None and len(drawing_pixel_points) > 0:
                    user_drawing_cm = []
                    # Subsample the drawing points to reduce waypoints for smoother movement
                    # Limit to ~100 waypoints for path complexity
                    step = max(1, len(drawing_pixel_points) // 100)
                    for i in range(0, len(drawing_pixel_points), step):
                        px, py = drawing_pixel_points[i]
                        pt_px = np.array([[[px, py]]], dtype=np.float32)
                        pt_cm = cv2.perspectiveTransform(pt_px, last_inv_homography_matrix)[0][0]
                        user_drawing_cm.append([pt_cm[0], pt_cm[1]])
                    
                    # Ensure the very last point is always included
                    if (len(drawing_pixel_points) - 1) % step != 0:
                         px, py = drawing_pixel_points[-1]
                         pt_px = np.array([[[px, py]]], dtype=np.float32)
                         pt_cm = cv2.perspectiveTransform(pt_px, last_inv_homography_matrix)[0][0]
                         user_drawing_cm.append([pt_cm[0], pt_cm[1]])

                    path_tracing_active = True
                    current_waypoint_index = 0
                    print(f"PC: 'Enter' pressed. Converted drawing to {len(user_drawing_cm)} waypoints. Starting trace.")
                else:
                    print("PC: Cannot convert drawing. Need plane markers and a drawn path.")
            elif key == ord('s'): # Keeping 's' for keyboard start of saved path as it's a common workflow
                if len(user_drawing_cm) > 0:
                    path_tracing_active = True
                    current_waypoint_index = 0
                    send_command_to_rpi("stop", 0, 1) # Ensure pen is down if tracing a saved path
                    print(f"PC: 'S' pressed. Starting path tracing with {len(user_drawing_cm)} waypoints.")
                else:
                    print("PC: No saved path to trace. Draw a path first and press Enter.")

    finally:
        send_command_to_rpi("stop", 0, 0) # Ensure robot stops and pen is lifted on exit
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        if robot_sock_send:
            robot_sock_send.close()
        print("PC: Application terminated gracefully.")

if __name__ == "__main__":
    main()
