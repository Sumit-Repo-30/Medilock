# BY SUMIT KUMAR SINGH

import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from PIL import Image, ImageTk

# ---------------- Mediapipe---------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# global video source handle
cap = None

# Global frames for card layout
splash_frame = None
exercise_frame = None
source_frame = None

# Define a small buffer for form checking against the calibrated range
RANGE_OF_MOTION_BUFFER = 5  # Degrees

# ----------------- UI ROOT INITIALIZATION ----------------------------
# IMPORTANT FIX: Initialize root here so it is available globally before any UI interaction functions are called.
root = tk.Tk()
root.title("MEDILOCK AI Based HealthCare App")
root.geometry("800x600")


# Styling setup moved down to the UI section where necessary, but root must exist now.


def calculate_angle(a, b, c):
    """Calculates the angle between three 2D points (A, B, C) with B as the vertex."""
    # Ensure points are numpy arrays for calculation
    a, b, c = np.array(a), np.array(b), np.array(c)

    # Calculate vector BA and BC
    # For angle at B (elbow, knee, or hip)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])

    angle = np.abs(radians * 180.0 / np.pi)

    # Normalize angle to be between 0 and 180
    if angle > 180.0:
        angle = 360 - angle
    return angle


# ------------------ Card Layout Navigation ------------------
def show_frame(frame_to_show):
    """Hides all frames and shows the selected frame."""
    global splash_frame, exercise_frame, source_frame
    for frame in [splash_frame, exercise_frame, source_frame]:
        if frame:
            frame.pack_forget()
    frame_to_show.pack(fill="both", expand=True)


def show_splash():
    show_frame(splash_frame)


def show_exercise_selection():
    show_frame(exercise_frame)


def show_source_selection(exercise_name):
    global current_exercise_name
    current_exercise_name = exercise_name
    show_frame(source_frame)


# ------------------ Popup Function ------------------
def select_reps_and_weight(exercise_name, start_callback):
    """Presents a popup for the user to select target reps and weight."""
    # 'root' is now guaranteed to exist globally.
    rep_window = tk.Toplevel(root)
    rep_window.title(f"Select Target Reps & Weight for {exercise_name}")
    rep_window.geometry("500x420")
    bg_color = "#e8f5e9"
    rep_window.configure(bg=bg_color)
    rep_window.grab_set()
    rep_window.resizable(False, False)

    # Determine if weight is relevant (Push Up is bodyweight only)
    is_bodyweight_default = (exercise_name == "Push Up" or exercise_name == "Squat")

    tk.Label(rep_window, text=f"How many reps of {exercise_name} do you want to do?",
             font=("Helvetica", 14, "bold"), bg=bg_color).pack(pady=10)

    chosen_reps = tk.IntVar(value=10)
    # Default weight to 5kg, or 0 if bodyweight
    chosen_weight = tk.IntVar(value=5 if not is_bodyweight_default else 0)

    # ---------------- Reps/Weight Status ----------------
    def get_weight_display():
        w = chosen_weight.get()
        if exercise_name == "Push Up" or (w == 0 and is_bodyweight_default):
            return "Bodyweight"
        return f"{w} kg"

    current_rep_label = tk.Label(rep_window,
                                 text=f"Current Reps: {chosen_reps.get()} | Weight: {get_weight_display()}",
                                 font=("Helvetica", 12), bg=bg_color, fg="#333")
    current_rep_label.pack(pady=(0, 10))

    def update_current_label():
        r = chosen_reps.get()
        current_rep_label.config(
            text=f"Current Reps: {r if r else 'None'} | Weight: {get_weight_display()}"
        )

    # ---------------- Reps ----------------
    btn_frame = tk.Frame(rep_window, bg=bg_color)
    btn_frame.pack(pady=5)

    preset_reps = [5, 10, 15, 20, 25, 30]

    def set_reps(val):
        chosen_reps.set(val)
        update_current_label()

    for rep in preset_reps:
        ttk.Button(btn_frame, text=str(rep), width=4,
                   command=lambda v=rep: set_reps(v)).pack(side="left", padx=5, ipadx=6, ipady=6)

    # Custom reps option
    def custom_reps_clicked():
        rep_window.withdraw()
        custom_val = simpledialog.askinteger("Custom Reps", "Enter custom reps (1-100):",
                                             parent=rep_window, minvalue=1, maxvalue=100)
        rep_window.deiconify()
        if custom_val:
            chosen_reps.set(custom_val)
            update_current_label()

    tk.Button(rep_window, text="Custom Reps", font=("Helvetica", 12),
              bg="#4CAF50", fg="white", command=custom_reps_clicked).pack(pady=8)

    # ---------------- Weight (Only show for non-bodyweight exercises or squat) ----------------
    if exercise_name != "Push Up":
        tk.Label(rep_window, text=f"Select Weight (kg):",
                 font=("Helvetica", 14, "bold"), bg=bg_color).pack(pady=6)

        weight_frame = tk.Frame(rep_window, bg=bg_color)
        weight_frame.pack(pady=5)

        preset_weights = [0, 5, 10, 20, 30]

        def set_weight(val):
            chosen_weight.set(val)
            update_current_label()

        for w in preset_weights:
            label = "BW" if w == 0 and is_bodyweight_default else str(w)
            ttk.Button(weight_frame, text=label, width=4,
                       command=lambda v=w: set_weight(v)).pack(side="left", padx=5, ipadx=6, ipady=6)

        # Custom weight option
        def custom_weight_clicked():
            rep_window.withdraw()
            custom_val = simpledialog.askinteger("Custom Weight", "Enter weight (0-100 kg):",
                                                 parent=rep_window, minvalue=0, maxvalue=100)
            rep_window.deiconify()
            if custom_val is not None:
                chosen_weight.set(custom_val)
                update_current_label()

        tk.Button(rep_window, text="Custom Weight", font=("Helvetica", 12),
                  bg="#2196F3", fg="white", command=custom_weight_clicked).pack(pady=8)

    # ---------------- Start Button ----------------
    def on_start():
        if chosen_reps.get() < 1:
            messagebox.showwarning("Invalid", "Please select valid reps.")
            return
        rep_window.destroy()
        # Pass the weight/reps to the exercise start function
        start_callback(chosen_reps.get(), chosen_weight.get())

    tk.Button(rep_window, text="Start Exercise", font=("Helvetica", 14, "bold"),
              bg="#f44336", fg="white", command=on_start).pack(pady=10, ipadx=10, ipady=5)

    tk.Button(rep_window, text="Cancel", font=("Helvetica", 12),
              bg="#9E9E9E", fg="white", command=rep_window.destroy).pack(pady=6)


def show_summary_popup(exercise_name, reps_target, total_reps, correct_reps, incorrect_reps, weight_target,
                       form_errors):
    """Displays the final summary of the workout session."""
    # Placeholder values for Time and Calories as they are not calculated in the current code
    # Time: Estimate based on 4 seconds per rep * total reps
    # Calories: Placeholder estimate (e.g., 0.5 kcal per bodyweight rep, 1.0 kcal per weighted rep)

    time_spent_sec = total_reps * 4
    minutes = time_spent_sec // 60
    seconds = time_spent_sec % 60
    time_display = f"{minutes} min {seconds} sec"

    if exercise_name == "Push Up":
        calories_burnt = round(total_reps * 0.5, 1)
        weight_display = "Bodyweight"
        muscles = "Chest, Shoulders (Anterior Deltoid), Triceps, Core"
    elif exercise_name == "Bicep Curl":
        # Estimate: Base 0.8 kcal/rep + (Weight in kg / 10) for contribution
        calories_burnt = round(total_reps * (0.8 + (weight_target / 10)), 1)
        weight_display = f"{weight_target} kg"
        muscles = "Biceps, Forearms"
    elif exercise_name == "Squat":
        # Estimate: Base 0.8 kcal/rep + (Weight in kg / 5) for contribution
        calories_burnt = round(total_reps * (0.8 + (weight_target / 5)), 1)
        weight_display = f"{weight_target} kg"
        muscles = "Quads, Glutes, Hamstrings, Core"
    elif exercise_name == "Shoulder Press":
        # Estimate: Base 0.9 kcal/rep + (Weight in kg / 7) for contribution
        calories_burnt = round(total_reps * (0.9 + (weight_target / 7)), 1)
        weight_display = f"{weight_target} kg"
        muscles = "Shoulders (Deltoids), Triceps, Trapezius"
    else:
        calories_burnt = 0.0
        weight_display = f"{weight_target} kg"
        muscles = "N/A"

    summary_window = tk.Toplevel(root)
    summary_window.title(f"🏆 {exercise_name} Session Summary 🏆")

    summary_window.geometry("550x500")

    bg_color = "#E8F5E9"  # Light Green Background
    fg_color = "#000000"
    header_color = "#4CAF50"
    summary_window.configure(bg=bg_color)
    summary_window.grab_set()
    summary_window.resizable(False, False)

    tk.Label(summary_window, text=f"Session Complete: {exercise_name}",
             font=("Helvetica", 18, "bold"), bg=header_color, fg="white").pack(fill='x', pady=(0, 15), ipady=10)

    # Use a Frame for better alignment
    data_frame = tk.Frame(summary_window, bg=bg_color)
    data_frame.pack(padx=20, pady=0, fill='x')

    # Helper function for label pairs
    def add_summary_line(parent, label_text, value_text, color="#333", font_size=12, bold=False):
        row = tk.Frame(parent, bg=bg_color)
        row.pack(fill='x', pady=5)
        font_style = ("Helvetica", font_size, "bold" if bold else "")
        tk.Label(row, text=label_text, font=font_style, bg=bg_color, fg=fg_color, anchor='w').pack(side='left')
        tk.Label(row, text=value_text, font=font_style, bg=bg_color, fg=color, anchor='e').pack(side='right')

    # Reps Summary
    add_summary_line(data_frame, "Target Reps:", str(reps_target), font_size=12)
    add_summary_line(data_frame, "Total Reps Completed:", str(total_reps), color="#4CAF50", font_size=14, bold=True)
    add_summary_line(data_frame, "Correct Reps:", str(correct_reps), color="#00C853", font_size=12)
    add_summary_line(data_frame, "Incorrect Reps:", str(incorrect_reps), color="#F44336", font_size=12)

    tk.Frame(data_frame, height=2, bg="#CCCCCC").pack(fill='x', pady=10)  # Separator

    # Other Metrics
    add_summary_line(data_frame, "Weight Used:", weight_display, font_size=12)
    add_summary_line(data_frame, "Time Spent:", time_display, font_size=12)
    add_summary_line(data_frame, "Calories Burnt (Est.):", f"{calories_burnt} kcal", font_size=12)

    # Muscle Impacted
    tk.Label(summary_window, text="Muscles Impacted:", font=("Helvetica", 12, "bold"), bg=bg_color, fg=fg_color).pack(
        pady=(10, 0), anchor='w', padx=20)
    tk.Label(summary_window, text=muscles, font=("Helvetica", 11), bg=bg_color, fg="#333").pack(
        pady=(0, 10), anchor='w', padx=20)

    # Form Errors
    error_display = ", ".join(form_errors) if form_errors else "None (Excellent!)"
    tk.Label(summary_window, text="Reported Form Errors:", font=("Helvetica", 12, "bold"), bg=bg_color,
             fg=fg_color).pack(
        pady=(0, 0), anchor='w', padx=20)
    tk.Label(summary_window, text=error_display, font=("Helvetica", 11), bg=bg_color,
             fg="#D50000" if form_errors else "#00C853").pack(pady=(0, 15), anchor='w', padx=20)


# ---------------- Calibration (webcam interactive) - BICEP CURL ---------------- 160-40
def calibrate_bicep_angle(use_webcam=True, video_cap=None):
    """
    Calibrates the elbow angle for the bicep curl exercise.
    Returns (down_threshold, up_threshold) for elbow angle.
    """
    if not use_webcam and video_cap is not None:
        print("Starting automatic BICEP CURL calibration from video file...")
        # FIX: Added required 'exercise_type' argument
        max_angle, min_angle = calibrate_from_video(video_cap, 'bicep_curl')
        if max_angle is not None and min_angle is not None:
            # Apply a buffer to the calculated min/max
            down_threshold = max_angle - 15
            up_threshold = min_angle + 15
            return down_threshold, up_threshold
        else:
            print("Auto-Calibration failed. Using default BICEP CURL calibration (160, 40).")
            return 160, 40

    # Interactive Calibration (for Webcam) - Bicep Curl
    if not use_webcam or video_cap is None:
        return 160, 40  # Fallback default if all else fails

    cap_local = video_cap
    calibrated_min = None
    calibrated_max = None

    if not cap_local.isOpened():
        messagebox.showerror("Error", "Webcam not opened for BICEP CURL calibration.")
        return 160, 40

    # STEP 1 – Arm extended (max angle)
    messagebox.showinfo("BICEP CURL Calibration Step 1",
                        "Stand straight and extend your left arm fully.\nPress 's' when ready.")
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        angle = None
        while cap_local.isOpened():
            ret, frame = cap_local.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW].y]
                wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST].x, lm[mp_pose.PoseLandmark.LEFT_WRIST].y]

                angle = calculate_angle(shoulder, elbow, wrist)
                cv2.putText(image, f"Extend Arm - Angle: {int(angle)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("BICEP CURL Calibration", image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):
                if angle is not None:
                    calibrated_max = angle
                    break
                else:
                    messagebox.showwarning("No detection",
                                           "No pose detected yet. Please make sure you are visible to the camera.")
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return 160, 40  # User quit calibration

    # STEP 2 – Full curl (min angle)
    messagebox.showinfo("BICEP CURL Calibration Step 2",
                        "Now curl your left arm fully.\nPress 's' when ready.")

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        angle = None
        while cap_local.isOpened():
            ret, frame = cap_local.read()
            if not ret:
                break

            # Mirror the image for better UX during webcam calibration
            if use_webcam:
                frame = cv2.flip(frame, 1)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW].y]
                wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST].x, lm[mp_pose.PoseLandmark.LEFT_WRIST].y]

                angle = calculate_angle(shoulder, elbow, wrist)
                cv2.putText(image, f"Curl Up - Angle: {int(angle)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("BICEP CURL Calibration", image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):
                if angle is not None:
                    calibrated_min = angle
                    break
                else:
                    messagebox.showwarning("No detection",
                                           "No pose detected yet. Please make sure you are visible to the camera.")
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return 160, 40  # User quit calibration

    cv2.destroyAllWindows()

    if calibrated_min is not None and calibrated_max is not None:
        down_threshold = calibrated_max - 15
        up_threshold = calibrated_min + 15
        print(f"BICEP CURL Calibrated → Down: {down_threshold}, Up: {up_threshold}")
        return down_threshold, up_threshold
    else:
        return 160, 40


# ---------------- Calibration (webcam interactive) - PUSH UP ----------------160-90
def calibrate_push_up_angle(use_webcam=True, video_cap=None):
    """
    Calibrates the elbow angle for the push-up exercise.
    Returns (down_threshold, up_threshold) for elbow angle.
    """
    if not use_webcam and video_cap is not None:
        print("Starting automatic PUSH UP calibration from video file...")
        # FIX: Added required 'exercise_type' argument
        max_angle, min_angle = calibrate_from_video(video_cap, 'push_up')
        if max_angle is not None and min_angle is not None:
            # Apply a buffer to the calculated min/max
            down_threshold = max_angle - 15  # Arm straightest, max angle
            up_threshold = min_angle + 15  # Arm bent, min angle
            print(f"Auto-Calibrated -> Down: {down_threshold:.2f}, Up: {up_threshold:.2f}")
            return down_threshold, up_threshold
        else:
            print("Auto-Calibration failed. Using default PUSH UP calibration (160, 90).")
            return 160, 90  # Default: 160 for top, 90 for bottom

    # Interactive Calibration (for Webcam) - Push Up
    if not use_webcam or video_cap is None:
        return 160, 90

    cap_local = video_cap
    calibrated_min = None  # Angle at the bottom (most bent)
    calibrated_max = None  # Angle at the top (most straight)

    if not cap_local.isOpened():
        messagebox.showerror("Error", "Webcam not opened for PUSH UP calibration.")
        return 160, 90

    # STEP 1 – Arms extended (max angle - body up)
    messagebox.showinfo("PUSH UP Calibration Step 1",
                        "Get into the push-up position (body up, arms straight).\nPress 's' when ready.")
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        angle = None
        while cap_local.isOpened():
            ret, frame = cap_local.read()
            if not ret:
                break

            # Mirror the image for better UX during webcam calibration
            if use_webcam:
                frame = cv2.flip(frame, 1)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # Use RIGHT arm for primary detection (more visible often)
                shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                elbow = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
                wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y]

                angle = calculate_angle(shoulder, elbow, wrist)
                cv2.putText(image, f"Arms Straight (Top) - Angle: {int(angle)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("PUSH UP Calibration", image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):
                if angle is not None:
                    calibrated_max = angle
                    break
                else:
                    messagebox.showwarning("No detection",
                                           "No pose detected yet. Make sure you are visible and in position.")
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return 160, 90

    # STEP 2 – Full contraction (min angle - body down)
    messagebox.showinfo("PUSH UP Calibration Step 2",
                        "Now lower your body fully (elbow bent).\nPress 's' when ready.")

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        angle = None
        while cap_local.isOpened():
            ret, frame = cap_local.read()
            if not ret:
                break

            # Mirror the image for better UX during webcam calibration
            if use_webcam:
                frame = cv2.flip(frame, 1)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # Use RIGHT arm
                shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                elbow = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
                wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y]

                angle = calculate_angle(shoulder, elbow, wrist)
                cv2.putText(image, f"Body Down (Bottom) - Angle: {int(angle)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("PUSH UP Calibration", image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):
                if angle is not None:
                    calibrated_min = angle
                    break
                else:
                    messagebox.showwarning("No detection",
                                           "No pose detected yet. Make sure you are visible and in position.")
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return 160, 90

    cv2.destroyAllWindows()

    if calibrated_min is not None and calibrated_max is not None:
        # Down threshold is for arm straight (max angle)
        down_threshold = calibrated_max - 15
        # Up threshold is for arm bent (min angle)
        up_threshold = calibrated_min + 15
        print(f"PUSH UP Calibrated → Down: {down_threshold}, Up: {up_threshold}")
        return down_threshold, up_threshold
    else:
        return 160, 90


# ---------------- Calibration (webcam interactive) - SQUAT ----------------170-80
def calibrate_squat_angle(use_webcam=True, video_cap=None):
    """
    Calibrates the knee angle for the squat exercise.
    Returns (up_threshold, down_threshold) for knee angle (up_threshold is for standing, down_threshold is for depth).
    """
    if not use_webcam and video_cap is not None:
        print("Starting automatic SQUAT calibration from video file...")
        # FIX: Added required 'exercise_type' argument
        max_angle, min_angle = calibrate_from_video(video_cap, 'squat')
        if max_angle is not None and min_angle is not None:
            # max_angle is at the top (standing/large angle)
            # min_angle is at the bottom (squatted/small angle)
            up_threshold = max_angle - 15  # Threshold for the top position
            down_threshold = min_angle + 15  # Threshold for squat depth
            print(f"Auto-Calibrated -> Up (Max Angle): {up_threshold:.2f}, Down (Min Angle): {down_threshold:.2f}")
            return up_threshold, down_threshold
        else:
            print("Auto-Calibration failed. Using default SQUAT calibration (170, 80).")
            return 170, 80  # Default: 170 for top, 80 for depth (knee angle)

    # Interactive Calibration (for Webcam) - Squat
    if not use_webcam or video_cap is None:
        return 170, 80

    cap_local = video_cap
    calibrated_min = None  # Angle at the bottom (most bent)
    calibrated_max = None  # Angle at the top (most straight)

    if not cap_local.isOpened():
        messagebox.showerror("Error", "Webcam not opened for SQUAT calibration.")
        return 170, 80

    # We track the angle: Hip -> Knee -> Ankle

    # STEP 1 – Standing fully upright (max angle)
    messagebox.showinfo("SQUAT Calibration Step 1",
                        "Stand fully upright, legs straight (Max Angle).\nPress 's' when ready.")
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        angle = None
        while cap_local.isOpened():
            ret, frame = cap_local.read()
            if not ret:
                break

            # Mirror the image for better UX during webcam calibration
            if use_webcam:
                frame = cv2.flip(frame, 1)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # Use RIGHT leg for tracking: Hip, Knee, Ankle
                hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
                knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y]
                ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

                angle = calculate_angle(hip, knee, ankle)
                cv2.putText(image, f"Stand Up (Max Angle) - Knee: {int(angle)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("SQUAT Calibration", image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):
                if angle is not None:
                    calibrated_max = angle
                    break
                else:
                    messagebox.showwarning("No detection",
                                           "No pose detected yet. Make sure you are visible to the camera.")
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return 170, 80

    # STEP 2 – Squat down to target depth (min angle - most bent)
    messagebox.showinfo("SQUAT Calibration Step 2",
                        "Squat down to your target depth (Hips low).\nPress 's' when ready.")

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        angle = None
        while cap_local.isOpened():
            ret, frame = cap_local.read()
            if not ret:
                break

            # Mirror the image for better UX during webcam calibration
            if use_webcam:
                frame = cv2.flip(frame, 1)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # Use RIGHT leg for tracking: Hip, Knee, Ankle
                hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
                knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y]
                ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

                angle = calculate_angle(hip, knee, ankle)
                cv2.putText(image, f"Squat Down (Min Angle) - Knee: {int(angle)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("SQUAT Calibration", image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):
                if angle is not None:
                    calibrated_min = angle
                    break
                else:
                    messagebox.showwarning("No detection",
                                           "No pose detected yet. Make sure you are visible to the camera.")
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return 170, 80

    cv2.destroyAllWindows()

    if calibrated_min is not None and calibrated_max is not None:
        # Up threshold is for stand-up (max angle)
        up_threshold = calibrated_max - 15
        # Down threshold is for squat depth (min angle)
        down_threshold = calibrated_min + 15
        print(f"SQUAT Calibrated → Up (Max Angle): {up_threshold}, Down (Min Angle): {down_threshold}")
        return up_threshold, down_threshold
    else:
        return 170, 80


# ---------------- Calibration (webcam interactive) - SHOULDER PRESS ----------------160-80
def calibrate_shoulder_press_angle(use_webcam=True, video_cap=None):
    """
    Calibrates the elbow angle for the shoulder press exercise.
    Returns (down_threshold, up_threshold) for elbow angle.
    """
    if not use_webcam and video_cap is not None:
        print("Starting automatic SHOULDER PRESS calibration from video file...")
        # FIX: Added required 'exercise_type' argument
        max_angle, min_angle = calibrate_from_video(video_cap, 'shoulder_press')
        if max_angle is not None and min_angle is not None:
            # Apply a buffer to the calculated min/max
            # Max angle is at the top (arms extended)
            # Min angle is at the bottom (arms bent)
            down_threshold = max_angle - 15
            up_threshold = min_angle + 15
            print(f"Auto-Calibrated -> Down (Max Angle): {down_threshold:.2f}, Up (Min Angle): {up_threshold:.2f}")
            return down_threshold, up_threshold
        else:
            print("Auto-Calibration failed. Using default SHOULDER PRESS calibration (160, 80).")
            return 160, 80  # Default: 160 for top, 80 for bottom

    # Interactive Calibration (for Webcam) - Shoulder Press
    if not use_webcam or video_cap is None:
        return 160, 80

    cap_local = video_cap
    calibrated_min = None  # Angle at the bottom (most bent)
    calibrated_max = None  # Angle at the top (most straight)

    if not cap_local.isOpened():
        messagebox.showerror("Error", "Webcam not opened for SHOULDER PRESS calibration.")
        return 160, 80

    # We track the angle: Shoulder -> Elbow -> Wrist

    # STEP 1 – Arms extended overhead (max angle)
    messagebox.showinfo("SHOULDER PRESS Calibration Step 1",
                        "Stand straight and press your right arm fully overhead (lockout).\nPress 's' when ready.")
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        angle = None
        while cap_local.isOpened():
            ret, frame = cap_local.read()
            if not ret:
                break

            # Mirror the image for better UX during webcam calibration
            if use_webcam:
                frame = cv2.flip(frame, 1)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # Use RIGHT arm for tracking
                shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                elbow = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
                wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y]

                angle = calculate_angle(shoulder, elbow, wrist)
                cv2.putText(image, f"Arms Extended (Top) - Angle: {int(angle)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("SHOULDER PRESS Calibration", image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):
                if angle is not None:
                    calibrated_max = angle
                    break
                else:
                    messagebox.showwarning("No detection",
                                           "No pose detected yet. Make sure you are visible and in position.")
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return 160, 80

    # STEP 2 – Lower the weight/arms (min angle - arms bent)
    messagebox.showinfo("SHOULDER PRESS Calibration Step 2",
                        "Now lower your right arm to the bottom position (elbow bent, hand near ear).\nPress 's' when ready.")

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        angle = None
        while cap_local.isOpened():
            ret, frame = cap_local.read()
            if not ret:
                break

            # Mirror the image for better UX during webcam calibration
            if use_webcam:
                frame = cv2.flip(frame, 1)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # Use RIGHT arm for tracking
                shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                elbow = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
                wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y]

                angle = calculate_angle(shoulder, elbow, wrist)
                cv2.putText(image, f"Arms Lowered (Bottom) - Angle: {int(angle)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("SHOULDER PRESS Calibration", image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):
                if angle is not None:
                    calibrated_min = angle
                    break
                else:
                    messagebox.showwarning("No detection",
                                           "No pose detected yet. Make sure you are visible and in position.")
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return 160, 80

    cv2.destroyAllWindows()

    if calibrated_min is not None and calibrated_max is not None:
        # Down threshold is for arm straight (max angle)
        down_threshold = calibrated_max - 15
        # Up threshold is for arm bent (min angle)
        up_threshold = calibrated_min + 15
        print(f"SHOULDER PRESS Calibrated → Down: {down_threshold}, Up: {up_threshold}")
        return down_threshold, up_threshold
    else:
        return 160, 80


# ---------------- Test helpers ------------------------
def load_test_video(path):
    c = cv2.VideoCapture(path)
    if not c.isOpened():
        messagebox.showerror("Error", f"Unable to load test video: {path}")
        return None
    return c


def calibrate_from_video(cap_obj, exercise_type, frames=60):
    """Improved AUTO calibration for Bicep Curl, Push-up, Shoulder Press, and Squat."""
    angles = []
    original_pos = cap_obj.get(cv2.CAP_PROP_POS_FRAMES)

    # Exercise → angle joint mapping
    landmark_config = {
        'bicep_curl': (
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST
        ),
        'push_up': (
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_WRIST
        ),
        'shoulder_press': (
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_WRIST
        ),
        'squat': (
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.RIGHT_ANKLE
        ),
    }

    if exercise_type not in landmark_config:
        print(f"[ERROR] Unsupported exercise for calibration: {exercise_type}")
        return None, None

    A, B, C = landmark_config[exercise_type]

    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
        for _ in range(frames):
            ret, frame = cap_obj.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                pA = (lm[A].x, lm[A].y)
                pB = (lm[B].x, lm[B].y)
                pC = (lm[C].x, lm[C].y)

                angle = calculate_angle(pA, pB, pC)
                angles.append(angle)

    # Reset video pointer
    cap_obj.set(cv2.CAP_PROP_POS_FRAMES, original_pos)

    if len(angles) < 10:
        print("[WARNING] Not enough angle samples, using default safe ranges.")
        return 160, 40      # fallback general values

    # Sort angles for filtering
    angles = sorted(angles)

    # Use top and bottom 10% average to avoid noise/outliers
    k = max(1, len(angles) // 10)
    down_angle = sum(angles[-k:]) / k     # highest range (full extension)
    up_angle = sum(angles[:k]) / k        # lowest range (full contraction)

    # Apply small tolerance to allow natural movement
    down_angle -= 8
    up_angle += 8

    return round(down_angle, 2), round(up_angle, 2)



# ---------------- UI actions ------------------------
def start_video_mode():
    global cap, current_exercise_name
    file_path = filedialog.askopenfilename(
        filetypes=[("Video Files", "*.mp4 *.mov *.avi")]
    )
    if file_path:
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            if current_exercise_name == "Bicep Curl":
                select_reps_and_weight("Bicep Curl", start_bicep_curl_with_values)
            elif current_exercise_name == "Push Up":
                select_reps_and_weight("Push Up", start_push_up_with_values)
            elif current_exercise_name == "Shoulder Press":
                select_reps_and_weight("Shoulder Press", start_shoulder_press_with_values)
            elif current_exercise_name == "Squat":
                select_reps_and_weight("Squat", start_squat_with_values)
            # Removed Deadlift/Hip Hinge and Lunge
        else:
            messagebox.showerror("Error", "Could not open the selected video file.")
            cap = None


def start_webcam_mode():
    global cap, current_exercise_name
    if cap is not None:
        cap.release()
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        if current_exercise_name == "Bicep Curl":
            select_reps_and_weight("Bicep Curl", start_bicep_curl_with_values)
        elif current_exercise_name == "Push Up":
            select_reps_and_weight("Push Up", start_push_up_with_values)
        elif current_exercise_name == "Shoulder Press":
            select_reps_and_weight("Shoulder Press", start_shoulder_press_with_values)
        elif current_exercise_name == "Squat":
            select_reps_and_weight("Squat", start_squat_with_values)
        # Removed Deadlift/Hip Hinge and Lunge
    else:
        messagebox.showerror("Error", "Could not access the webcam.")
        cap = None


# ----------------- BICEP CURL Implementation --------------------✅
def start_bicep_curl_with_values(reps_target, weight_target):
    global cap
    if cap is None or not cap.isOpened():
        messagebox.showerror("Error", "No video source. Choose Upload Video or Start Webcam.")
        return

    # Detect if webcam or uploaded video
    is_webcam = False
    try:
        if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) <= 1:
            is_webcam = True
    except:
        is_webcam = True

    print("Starting Bicep Curl calibration...")
    down_threshold, up_threshold = calibrate_bicep_angle(is_webcam, cap)

    # Reset video if file
    if not is_webcam:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ---------------------- Variables ----------------------
    counter = 0
    correct_reps = 0
    incorrect_reps = 0
    form_errors = set()
    stage = "down"

    last_feedback = ""
    feedback_cooldown = 0

    swing_threshold = 0.06
    RANGE_OF_MOTION_BUFFER = 10

    # NEW: Form error smoothing
    current_rep_has_error = False
    bad_form_frames = 0
    BAD_FORM_THRESHOLD = 10  # must hold error this long to count

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
            image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            overlay_color = (255, 255, 255)
            overlay_text = ""
            form_errors_this_frame = set()

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW].y]
                wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST].x, lm[mp_pose.PoseLandmark.LEFT_WRIST].y]

                angle = calculate_angle(shoulder, elbow, wrist)

                elbow_shift = abs(elbow[0] - shoulder[0])

                # ---------------------- Form Rules ----------------------
                if elbow_shift > swing_threshold:
                    form_errors_this_frame.add("arm swing")
                    overlay_text = "ARM SWING: Keep elbow fixed!"
                    overlay_color = (0, 0, 255)

                elif stage == "down" and angle < down_threshold - RANGE_OF_MOTION_BUFFER:
                    form_errors_this_frame.add("incomplete extension")
                    overlay_text = "Extend fully!"
                    overlay_color = (0, 165, 255)

                elif stage == "up" and angle > up_threshold + RANGE_OF_MOTION_BUFFER:
                    form_errors_this_frame.add("incomplete curl")
                    overlay_text = "Curl fully!"
                    overlay_color = (0, 165, 255)

                else:
                    overlay_text = "Good form! Full range"
                    overlay_color = (0, 255, 0)

                # ----------- NEW: Form Detection with Stability -----------
                if form_errors_this_frame:
                    bad_form_frames += 1
                else:
                    bad_form_frames = 0

                if bad_form_frames > BAD_FORM_THRESHOLD:
                    current_rep_has_error = True
                    form_errors.update(form_errors_this_frame)

                # ---------- Rep Logic ----------
                TOLERANCE = 12
                adjusted_down_threshold = down_threshold - TOLERANCE
                adjusted_up_threshold = up_threshold + TOLERANCE

                # Phase change DOWN -> UP
                if angle < adjusted_up_threshold and stage == "down":
                    stage = "up"

                # Rep complete when arm returns down AFTER peak
                elif angle > adjusted_down_threshold and stage == "up":
                    counter += 1

                    if current_rep_has_error:
                        incorrect_reps += 1
                    else:
                        correct_reps += 1

                    # Reset for next rep
                    current_rep_has_error = False
                    bad_form_frames = 0
                    stage = "down"

                # Draw Pose
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Stabilized display text
                if overlay_text != last_feedback or feedback_cooldown <= 0:
                    last_feedback = overlay_text
                    feedback_cooldown = 6
                else:
                    feedback_cooldown -= 1

                # ---------------- GUI Display ----------------
                cv2.putText(image, f"Reps: {counter}/{reps_target}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(image, f"Weight: {weight_target}kg", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.putText(image, last_feedback, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, overlay_color, 2)
                # cv2.putText(image, f"Angle: {int(angle)}°", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

                if counter >= reps_target:
                    cv2.putText(image, "TARGET COMPLETED!", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
                    cv2.imshow("Bicep Curl Tracker", image)
                    cv2.waitKey(1000)
                    break

            else:
                cv2.putText(image, "No person detected.", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imshow("Bicep Curl Tracker", image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # ---------------------- Report ----------------------
    print("\n===== BICEP CURL REPORT =====")
    print(f"Total Reps: {counter}")
    print(f"Correct Reps: {correct_reps}")
    print(f"Incorrect Reps: {incorrect_reps}")
    print(f"Form Issues: {', '.join(form_errors) if form_errors else 'None'}")
    print("================================\n")

    show_summary_popup("Bicep Curl", reps_target, counter, correct_reps, incorrect_reps, weight_target, form_errors)


# ----------------- PUSH UP Implementation --------------------✅
def start_push_up_with_values(reps_target, weight_target):
    global cap
    if cap is None or not cap.isOpened():
        messagebox.showerror("Error", "No video source. Choose Upload Video or Start Webcam.")
        return

    is_webcam = False
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 1:
            is_webcam = True
    except:
        is_webcam = True

    print("Starting Push Up calibration...")
    # Calibration returns elbow angle thresholds (down = straight arm, up = bent arm)
    down_threshold, up_threshold = calibrate_push_up_angle(is_webcam, cap)

    if not is_webcam:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Rep Counting Variables
    counter = 0
    correct_reps = 0
    incorrect_reps = 0
    form_errors = set()
    stage = "down"  # "down" stage means body is up (arms straight), ready to go down
    last_feedback = ""
    feedback_cooldown = 0
    # Hip angle threshold for straight back (close to 180 degrees)
    HIP_ANGLE_THRESHOLD = 160  # Adjusted for slight forgiveness

    # NEW: Flag to track if a hip/critical form error occurred during the current rep cycle
    current_rep_form_is_good = True

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if not is_webcam:
                    break
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            overlay_text = ""
            overlay_color = (255, 255, 255)
            color = overlay_color

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # Right Arm: Shoulder, Elbow, Wrist
                r_shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                r_elbow = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
                r_wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y]

                # Back/Hip: Shoulder, Hip, Knee
                hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
                knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y]

                elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                hip_angle = calculate_angle(r_shoulder, hip, knee)

                # --- Form Check Logic ---
                form_errors_current = set()

                # Priority 1: Hip Sag/Raise Check (Straight back)
                if hip_angle < HIP_ANGLE_THRESHOLD:
                    overlay_text = "HIP SAG: Keep your back straight!"
                    form_errors_current.add("hip sag")
                    overlay_color = (0, 0, 255)  # Red
                elif hip_angle > 180 + (180 - HIP_ANGLE_THRESHOLD):  # A raise in the hip (like a pike)
                    overlay_text = "HIP RAISE: Keep your back straight!"
                    form_errors_current.add("hip raise")
                    overlay_color = (0, 0, 255)  # Red

                # Priority 2: Range of Motion Check (Elbow angle)
                elif stage == "down" and elbow_angle < down_threshold - RANGE_OF_MOTION_BUFFER:
                    # User is at the top (should be arm straight) but arm isn't straight enough
                    overlay_text = "INCOMPLETE EXTENSION: Straighten arms fully at the top!"
                    form_errors_current.add("incomplete extension")
                    overlay_color = (0, 165, 255)  # Orange

                elif stage == "up" and elbow_angle > up_threshold + RANGE_OF_MOTION_BUFFER:
                    # User is at the bottom (should be arm bent) but arm isn't bent enough
                    overlay_text = "INCOMPLETE DEPTH: Go lower to complete the rep!"
                    form_errors_current.add("incomplete depth")
                    overlay_color = (0, 165, 255)  # Orange

                else:
                    # Good form, or within acceptable range
                    overlay_text = "Good form! Full range."
                    overlay_color = (0, 255, 0)  # Green

                color = overlay_color
                form_errors.update(form_errors_current)

                # --- Rep Counting Logic (State Machine) ---
                # Down stage: Arm is straight (elbow angle > down_threshold), body is up
                if elbow_angle > down_threshold:
                    stage = "down"

                # Up stage: Arm is bent (elbow angle < up_threshold), body is low
                # This counts the transition from a straight arm (down) to a bent arm (up)
                if elbow_angle < up_threshold and stage == "down":
                    stage = "up"
                    counter += 1

                    # Check form at the point of counting the rep (bottom of the movement)
                    if 'hip sag' in form_errors_current or 'hip raise' in form_errors_current or 'incomplete depth' in form_errors_current:
                        incorrect_reps += 1
                    else:
                        correct_reps += 1

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

                # Feedback Cooldown
                if overlay_text != last_feedback or feedback_cooldown == 0:
                    feedback_cooldown = 8
                    last_feedback = overlay_text
                elif feedback_cooldown > 0:
                    feedback_cooldown -= 1

                # Display Stats and Feedback
                cv2.putText(image, f"Reps: {counter}/{reps_target}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(image, f"Weight: {weight_target} kg (Bodyweight)", (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, last_feedback, (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, color, 2)

                if counter >= reps_target:
                    cv2.putText(image, "Target Completed!", (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                    cv2.imshow('Push Up Tracker', image)
                    cv2.waitKey(1200)
                    break
            else:
                cv2.putText(image, "No person detected. Move into camera view.", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Push Up Tracker', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    try:
        cap.release()
    except:
        pass
    cv2.destroyAllWindows()

    print("\n===== PUSH UP REPORT =====")
    print(f"Reps: {counter}")
    print(f"Correct reps: {correct_reps}")
    print(f"Incorrect reps: {incorrect_reps}")
    print(f"Form errors: {', '.join(form_errors) if form_errors else 'None'}")
    print("============================\n")

    # Call the new summary popup
    show_summary_popup("Push Up", reps_target, counter, correct_reps, incorrect_reps, 0, form_errors)


# ----------------- SQUAT Implementation --------------------✅
def start_squat_with_values(reps_target, weight_target):
    global cap
    if cap is None or not cap.isOpened():
        messagebox.showerror("Error", "No video source. Choose Upload Video or Start Webcam.")
        return

    is_webcam = False
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 1:
            is_webcam = True
    except:
        is_webcam = True

    print(f"Starting Squat calibration (Weight: {weight_target} kg)...")
    # Calibration returns (up_threshold, down_threshold) for knee angle
    up_threshold, down_threshold = calibrate_squat_angle(is_webcam, cap)

    if not is_webcam:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Rep Counting Variables
    counter = 0
    correct_reps = 0
    incorrect_reps = 0
    form_errors = set()
    stage = "up"  # "up" stage means user is standing/at the top, ready to squat down
    last_feedback = ""
    feedback_cooldown = 0

    # Threshold for excessive forward torso lean (Shoulder-Hip-Knee angle)
    FORWARD_LEAN_THRESHOLD = 95

    # Increase buffer to allow slightly more tolerance for motion extremes
    SQUAT_MOTION_BUFFER = 7

    # NEW: Flag to track if a critical form error occurred during the current rep cycle
    current_rep_is_valid = True

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if not is_webcam:
                    break
                break

            # --- USABILITY FIX: Mirror the webcam image ---
            if is_webcam:
                frame = cv2.flip(frame, 1)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            overlay_text = ""
            overlay_color = (255, 255, 255)
            color = overlay_color

            form_errors_in_frame = set()  # Track errors only for the current frame for display/feedback

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # Knee Angle (Hip, Knee, Ankle)
                hip_p = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
                knee_p = [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y]
                ankle_p = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

                # Torso/Hip Angle (Shoulder, Hip, Knee)
                shoulder_p = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]

                knee_angle = calculate_angle(hip_p, knee_p, ankle_p)
                hip_angle = calculate_angle(shoulder_p, hip_p, knee_p)

                # --- Form Check Logic ---

                # Priority 1: Torso Lean Check (Warning)
                if hip_angle < FORWARD_LEAN_THRESHOLD:
                    overlay_text = f"LEAN WARNING: Keep chest higher! "
                    form_errors_in_frame.add("forward lean detected")
                    overlay_color = (0, 100, 255)  # Blue/Orange warning

                # Priority 2: Incomplete Depth Check (CRITICAL)
                # This logic is complex and better handled by the state transition (knee_angle < down_threshold)
                # but providing visual feedback helps.
                elif stage == "up" and knee_angle > up_threshold - 5 and knee_angle > down_threshold + 5:
                    # If user has started moving but isn't deep enough, give feedback
                    overlay_text = f"Squat deeper! "
                    overlay_color = (255, 255, 0)  # Yellow warning

                else:
                    # Default feedback
                    overlay_text = "Good form! Full range."
                    overlay_color = (0, 255, 0)  # Green

                # Check if we are still in the 'up' phase and haven't hit depth yet (visual feedback)
                if stage == "up" and knee_angle > down_threshold:
                    overlay_color = (255, 255, 0)  # Yellow: Mid-rep motion

                color = overlay_color

                # --- Rep Counting Logic (State Machine) ---

                # State Check: Down phase (Squatted/Knees bent - Depth achieved)
                if knee_angle < down_threshold:
                    # Depth achieved, start ascent phase
                    if stage == "up":
                        stage = "down"
                        current_rep_is_valid = True  # Reset validity on hitting depth

                # Rep Count Trigger: Up phase (Standing/Knees straight)
                # A rep is counted when we transition back up past the standing threshold.
                if knee_angle > up_threshold and stage == "down":

                    # --- JUDGEMENT ---

                    # Check for incomplete extension at the *top* if the angle is too low (e.g. they only hit 165 but threshold is 170)
                    is_incomplete_lockout_at_top = knee_angle < up_threshold - RANGE_OF_MOTION_BUFFER

                    if not current_rep_is_valid or is_incomplete_lockout_at_top or (
                            "forward lean detected" in form_errors_in_frame):
                        incorrect_reps += 1
                        # Aggregate ALL errors (critical and warning) for the final report
                        form_errors.update(form_errors_in_frame)
                        if is_incomplete_lockout_at_top:
                            form_errors.add("Incomplete Lockout")
                        if not current_rep_is_valid:
                            form_errors.add("Rep failed due to early error")  # Should not happen with corrected logic
                    else:
                        correct_reps += 1

                    # Reset state for the next rep
                    stage = "up"
                    counter += 1
                    current_rep_is_valid = True
                    form_errors_in_frame.clear()

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

                # Feedback Cooldown
                if overlay_text != last_feedback or feedback_cooldown == 0:
                    feedback_cooldown = 8
                    last_feedback = overlay_text
                elif feedback_cooldown > 0:
                    feedback_cooldown -= 1

                # Display Stats and Feedback
                cv2.putText(image, f"Reps: {counter}/{reps_target}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(image, f"Weight: {weight_target} kg", (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, last_feedback, (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, color, 2)


                if counter >= reps_target:
                    cv2.putText(image, "Target Completed!", (10, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                    cv2.imshow('Squat Tracker', image)
                    cv2.waitKey(1200)
                    break
            else:
                cv2.putText(image, "No person detected. Move into camera view.", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Squat Tracker', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    try:
        cap.release()
    except:
        pass
    cv2.destroyAllWindows()

    print("\n===== SQUAT REPORT =====")
    print(f"Reps: {counter}")
    print(f"Correct reps: {correct_reps}")
    print(f"Incorrect reps: {incorrect_reps}")
    print(f"Form errors: {', '.join(form_errors) if form_errors else 'None'}")
    print("============================\n")

    # Call the new summary popup
    show_summary_popup("Squat", reps_target, counter, correct_reps, incorrect_reps, weight_target, form_errors)


# ----------------- SHOULDER PRESS Implementation --------------------❌✅
def start_shoulder_press_with_values(reps_target, weight_target):
    global cap
    if cap is None or not cap.isOpened():
        messagebox.showerror("Error", "No video source. Choose Upload Video or Start Webcam.")
        return

    is_webcam = False
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 1:
            is_webcam = True
    except:
        is_webcam = True

    print("Starting Shoulder Press calibration...")
    # Calibration returns elbow angle thresholds (down = straight arm, up = bent arm)
    down_threshold, up_threshold = calibrate_shoulder_press_angle(is_webcam, cap)

    if not is_webcam:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Rep Counting Variables
    counter = 0
    correct_reps = 0
    incorrect_reps = 0
    form_errors = set()
    stage = "down"  # "down" stage means arms are lowered, ready to press "up"
    last_feedback = ""
    feedback_cooldown = 0
    LEAN_THRESHOLD = 0.06  # Horizontal displacement threshold for hip relative to shoulder

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if not is_webcam:
                    break
                break

            # --- USABILITY FIX: Mirror the webcam image ---
            if is_webcam:
                frame = cv2.flip(frame, 1)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            overlay_text = ""
            overlay_color = (255, 255, 255)
            color = overlay_color

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # Right Arm: Shoulder, Elbow, Wrist
                r_shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                r_elbow = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
                r_wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y]
                r_hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]

                elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                # Check for leaning (horizontal displacement of hip relative to shoulder)
                lean_dx = r_hip[0] - r_shoulder[0]

                # --- Form Check Logic ---
                form_errors_current = set()

                # Priority 1: Leaning Check (Safety)
                if abs(lean_dx) > LEAN_THRESHOLD:
                    overlay_text = "LEANING: Keep your core tight and torso straight!"
                    form_errors_current.add("leaning")
                    overlay_color = (0, 0, 255)  # Red

                # Priority 2: Range of Motion Check (Elbow angle)
                elif stage == "down" and elbow_angle < down_threshold - 10:
                    # User is at the top (should be straight arm) but angle is too small
                    overlay_text = "INCOMPLETE LOCKOUT: Press fully overhead!"
                    form_errors_current.add("incomplete lockout")
                    overlay_color = (0, 165, 255)  # Orange

                elif stage == "up" and elbow_angle > up_threshold + 10:
                    # User is at the bottom (should be bent arm) but angle is too large
                    overlay_text = "INCOMPLETE DEPTH: Lower the bar closer to your shoulders!"
                    form_errors_current.add("incomplete depth")
                    overlay_color = (0, 165, 255)  # Orange

                else:
                    # Good form, or within acceptable range
                    overlay_text = "Good form! Full range."
                    overlay_color = (0, 255, 0)  # Green

                color = overlay_color
                form_errors.update(form_errors_current)

                # --- Rep Counting Logic (State Machine) ---
                # Down stage: Arm is lowered/bent (elbow angle < up_threshold), ready to press up
                if elbow_angle < up_threshold:
                    stage = "down"

                # Up stage: Arm is pressed/straight (elbow angle > down_threshold), rep completed
                # This counts the transition from bent arm (down) to straight arm (up)
                if elbow_angle > down_threshold and stage == "down":
                    stage = "up"
                    counter += 1

                    # Check form at the point of counting the rep (top of the movement)
                    if form_errors_current:
                        incorrect_reps += 1
                    else:
                        correct_reps += 1
                    # Clear errors for the next rep
                    form_errors_current.clear()

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

                # Feedback Cooldown
                if overlay_text != last_feedback or feedback_cooldown == 0:
                    feedback_cooldown = 8
                    last_feedback = overlay_text
                elif feedback_cooldown > 0:
                    feedback_cooldown -= 1

                # Display Stats and Feedback
                cv2.putText(image, f"Reps: {counter}/{reps_target}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(image, f"Weight: {weight_target} kg", (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, last_feedback, (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, color, 2)


                if counter >= reps_target:
                    cv2.putText(image, "Target Completed!", (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                    cv2.imshow('Shoulder Press Tracker', image)
                    cv2.waitKey(1200)
                    break
            else:
                cv2.putText(image, "No person detected. Move into camera view.", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Shoulder Press Tracker', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    try:
        cap.release()
    except:
        pass
    cv2.destroyAllWindows()

    print("\n===== SHOULDER PRESS REPORT =====")
    print(f"Reps: {counter}")
    print(f"Correct reps: {correct_reps}")
    print(f"Incorrect reps: {incorrect_reps}")
    print(f"Form errors: {', '.join(form_errors) if form_errors else 'None'}")
    print("============================\n")

    # Call the new summary popup
    show_summary_popup("Shoulder Press", reps_target, counter, correct_reps, incorrect_reps, weight_target, form_errors)


# ----------------- UI ----------------------------
PRIMARY_COLOR = "#4CAF50"
SECONDARY_COLOR = "#2196F3"
DANGER_COLOR = "#f44336"
BG_COLOR = "#E8F5E9"
TEXT_MUTED = "#4A4A4A"

# Setup is done globally at the top of the file, but styling definitions are applied here.

# root.configure(bg=BG_COLOR) # Done globally above

style = ttk.Style(root)
try:
    style.theme_use("clam")
except:
    pass
style.configure("TButton", font=("Helvetica", 12, "bold"), padding=10)
style.configure("Exercise.TButton", background=PRIMARY_COLOR, foreground="white")
style.map("Exercise.TButton", background=[("active", "#45a049")])
style.configure("Info.TButton", background=SECONDARY_COLOR, foreground="white")
style.map("Info.TButton", background=[("active", "#1976D2")])
style.configure("Danger.TButton", background=DANGER_COLOR, foreground="white")
style.map("Danger.TButton", background=[("active", "#d32f2f")])

# ---------------- FRAME DEFINITIONS ----------------

# 1. Splash Frame (Initial)
splash_frame = tk.Frame(root, bg=BG_COLOR)
tk.Label(splash_frame, text="💪 MEDILOCK AI ", font=("Helvetica", 28, "bold"), bg=BG_COLOR, fg=PRIMARY_COLOR).pack(
    pady=(60, 10))
tk.Label(splash_frame, text="Track Reps • Check Form • Track Data", font=("Helvetica", 14), bg=BG_COLOR,
         fg=TEXT_MUTED).pack(pady=(0, 50))

ttk.Button(splash_frame, text="Start Exercise", style="Exercise.TButton",
           command=show_exercise_selection).pack(pady=20, ipadx=40, ipady=15)

tk.Label(splash_frame, text="“Train Smart, Eat Right, Stay Fit!”", font=("Helvetica", 12, "italic"), bg=BG_COLOR,
         fg=TEXT_MUTED).pack(side="bottom", pady=20)

# 2. Exercise Selection Frame
exercise_frame = tk.Frame(root, bg=BG_COLOR)
tk.Label(exercise_frame, text="Select Exercise", font=("Helvetica", 24, "bold"), bg=BG_COLOR, fg=PRIMARY_COLOR).pack(
    pady=(40, 20))

# Bicep Curl Button
ttk.Button(exercise_frame, text="Bicep Curl 💪", style="Exercise.TButton",
           command=lambda: show_source_selection("Bicep Curl")).pack(pady=10, ipadx=20, ipady=10)

# Push Up Button
ttk.Button(exercise_frame, text="Push Up 🙌", style="Exercise.TButton",
           command=lambda: show_source_selection("Push Up")).pack(pady=10, ipadx=20, ipady=10)

# Shoulder Press Button
ttk.Button(exercise_frame, text="Shoulder Press 🏋️", style="Exercise.TButton",
           command=lambda: show_source_selection("Shoulder Press")).pack(pady=10, ipadx=20, ipady=10)

# Squat Button
ttk.Button(exercise_frame, text="Squat 🦵", style="Exercise.TButton",
           command=lambda: show_source_selection("Squat")).pack(pady=10, ipadx=20, ipady=10)

# Back Button
ttk.Button(exercise_frame, text="← Back to Home", style="Info.TButton",
           command=show_splash).pack(side="bottom", pady=20)

# 3. Source Selection Frame
source_frame = tk.Frame(root, bg=BG_COLOR)
tk.Label(source_frame, text="Select Video Source", font=("Helvetica", 24, "bold"), bg=BG_COLOR,
         fg=SECONDARY_COLOR).pack(pady=(40, 30))

# Upload Video Button
ttk.Button(source_frame, text="Upload Video File", style="Exercise.TButton",
           command=start_video_mode).pack(pady=15, ipadx=30, ipady=15)

# Start Webcam Button.....................................................................................................................................................................................................................................................................................

ttk.Button(source_frame, text="Start Webcam", style="Exercise.TButton",
           command=start_webcam_mode).pack(pady=15, ipadx=30, ipady=15)

# Back Button
ttk.Button(source_frame, text="← Back to Exercises", style="Info.TButton",
           command=show_exercise_selection).pack(side="bottom", pady=20)

# Global variable to hold the currently selected exercise
current_exercise_name = None

# Initialize the UI state
show_splash()

root.mainloop()