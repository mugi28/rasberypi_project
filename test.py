import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import time
import threading
import numpy as np
import mediapipe as mp # MP 임포트 추가 (이미지 생성에 필요)

# ★ Picamera2 library (Required for Bookworm)
try:
    from picamera2 import Picamera2
except ImportError:
    print("❌ Picamera2 library not found. Installation required.")
    raise

# ★ User-defined module imports
try:
    import config
    from cpu_utils import MediaPipeHandler
    from tpu_utils import load_labels, make_interpreter
    from audio_utils import AudioAnalyzer
    # servo.py의 내용은 DC 모터 PWM 코드로 대체되었지만, 함수 이름은 유지됩니다.
    from servo import setup_hardware, activate_feedback, cleanup_hardware 
except ImportError as e:
    print(f"❌ Required file missing: {e}")
    raise

# ==========================================
# [CLASS] Camera Thread
# ==========================================
class CameraThread:
    def __init__(self):
        self.frame = None
        self.running = True
        self.picam2 = None
        self.is_opened = False

        try:
            self.picam2 = Picamera2()
            # Analysis resolution 640x480 (maintain aspect ratio)
            cfg = self.picam2.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            self.picam2.configure(cfg)
            self.picam2.start()
            self.is_opened = True
            print("[System] Camera started successfully")
        except Exception as e:
            print(f"[System] Camera error: {e}")

        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            if self.is_opened:
                try:
                    self.frame = self.picam2.capture_array()
                except:
                    time.sleep(0.01)
            else:
                time.sleep(0.1)

    def get_frame(self):
        return self.frame

    def stop(self):
        self.running = False
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()

# ==========================================
# [APP] Main Application
# ==========================================
class SmartFocusApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Focus AI (HUD Version)")
        self.width = 800
        self.height = 480
        self.root.geometry(f"{self.width}x{self.height}")

        # Variable settings
        self.is_monitoring = False
        self.use_mic = True
        self.use_cam = True
        self.show_camera = True  # Show camera on screen
        
        # Focus level setting (High/Mid/Low)
        self.focus_level = "Mid"  # Default: Mid
        self.max_warnings = 2  # Mid = 2 warnings
        self.current_warnings = 0
        self.warning_history = []  # Warning stack/history
        
        self.slacking_start_time = None 
        self.violation_threshold = 1.5 
        # WARNING: self.violation_threshold는 이제 사용되지 않습니다.

        # Timer settings
        self.timer_enabled = True  # Timer on/off toggle
        self.current_streak_start = time.time()
        self.timer_paused_time = 0  # Accumulated paused time
        self.timer_paused_at = None  # When timer was paused
        self.is_timer_paused = False  # Current pause state
        self.last_punish_time = 0
        self.t_eyes_start = None
        self.t_yawn_start = None # 하품 타이머 추가

        # AI initialization
        print("[Init] Loading AI engine... (takes about 5 seconds)")
        setup_hardware()
        self.audio_handler = AudioAnalyzer()
        self.mp_handler = MediaPipeHandler()
        
        self.labels = load_labels("labels.txt")
        self.inter = make_interpreter("model_edgetpu.tflite")
        self.in_det = self.inter.get_input_details()[0]
        self.out_det = self.inter.get_output_details()[0]
        self.tpu_w = self.in_det["shape"][2]
        self.tpu_h = self.in_det["shape"][1]
        print("[Init] AI engine ready!")

        self.cam_thread = CameraThread()
        self.cam_image_id = None

        self.create_home_screen()
        self.create_active_screen()
        self.frame_home.pack(fill="both", expand=True)

        self.frame_count = 0 
        self.analysis_interval = 3 

        self.update_loop()

    # --- UI ---
    def create_home_screen(self):
        self.frame_home = tk.Frame(self.root, bg="#2c3e50")
        
        btn_set = tk.Button(self.frame_home, text="⚙️ Settings", command=self.open_settings, font=("Arial", 12))
        btn_set.place(x=self.width - 120, y=20, width=100, height=40)

        btn_start = tk.Button(self.frame_home, text="▶ Start Focus", command=self.start_monitoring,
                              font=("Arial", 24, "bold"), bg="#27ae60", fg="white")
        btn_start.place(relx=0.5, rely=0.5, anchor="center", width=300, height=100)
        
        status = "System OK" if self.cam_thread.is_opened else "Cam Error"
        tk.Label(self.frame_home, text=status, font=("Arial", 14), bg="#2c3e50", fg="white").place(relx=0.5, rely=0.8, anchor="center")

    def create_active_screen(self):
        self.frame_active = tk.Frame(self.root)
        self.canvas = tk.Canvas(self.frame_active, width=self.width, height=self.height, bg="black")
        self.canvas.pack(fill="both", expand=True)

        btn_back = tk.Button(self.canvas, text="⏹ Stop", command=self.stop_monitoring, bg="#e74c3c", fg="white", font=("bold"))
        self.canvas.create_window(60, 30, window=btn_back, width=100, height=40)
        
        btn_set = tk.Button(self.canvas, text="⚙️", command=self.open_settings, bg="gray", fg="white")
        self.canvas.create_window(self.width - 40, 30, window=btn_set, width=40, height=40)

        self.btn_mic = tk.Button(self.canvas, text="Mic ON", command=self.toggle_mic, bg="#27ae60", fg="white", font=("Arial", 10))
        self.canvas.create_window(self.width//2 - 100, 30, window=self.btn_mic, width=80, height=40)
        
        self.btn_cam = tk.Button(self.canvas, text="Cam ON", command=self.toggle_cam, bg="#27ae60", fg="white", font=("Arial", 10))
        self.canvas.create_window(self.width//2 + 100, 30, window=self.btn_cam, width=80, height=40)
        
        self.btn_timer = tk.Button(self.canvas, text="Timer ON", command=self.toggle_timer, bg="#3498db", fg="white", font=("Arial", 10))
        self.canvas.create_window(self.width//2, 80, window=self.btn_timer, width=90, height=35)

        # Camera display area (small box in top-right corner)
        self.cam_box_x = self.width - 340  # Camera box position X
        self.cam_box_y = 80  # Camera box position Y
        self.cam_box_w = 320  # Camera box width
        self.cam_box_h = 240  # Camera box height
        
        # Draw camera frame border
        self.cam_border_id = self.canvas.create_rectangle(
            self.cam_box_x - 2, self.cam_box_y - 2,
            self.cam_box_x + self.cam_box_w + 2, self.cam_box_y + self.cam_box_h + 2,
            outline="white", width=2
        )
        
        # Text UI
        self.id_status = self.canvas.create_text(self.width//2, 440, text="Ready", font=("Arial", 20, "bold"), fill="white")
        self.id_timer = self.canvas.create_text(self.width - 100, 50, text="00:00", font=("Arial", 30, "bold"), fill="white")
        self.id_warning = self.canvas.create_text(100, 440, text=f"Warning: 0/{self.max_warnings} ({self.focus_level})", font=("Arial", 16, "bold"), fill="yellow")
        
        # Additional info display area (left side)
        self.info_text_y = 100
        self.id_info_ear = self.canvas.create_text(20, self.info_text_y, text="EAR: --", font=("Arial", 12), fill="white", anchor=tk.W)
        self.id_info_mar = self.canvas.create_text(20, self.info_text_y + 25, text="MAR: --", font=("Arial", 12), fill="white", anchor=tk.W)
        self.id_info_action = self.canvas.create_text(20, self.info_text_y + 50, text="Action: --", font=("Arial", 12), fill="white", anchor=tk.W)
        self.id_info_person = self.canvas.create_text(20, self.info_text_y + 75, text="Person: --", font=("Arial", 12), fill="white", anchor=tk.W)
        
        # Warning stack display (left side, below info)
        self.warning_stack_y = self.info_text_y + 110
        self.id_warning_stack = self.canvas.create_text(20, self.warning_stack_y, text="Warnings: None", font=("Arial", 11), fill="yellow", anchor=tk.W)

    def open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.geometry("350x350")
        win.configure(bg="#ecf0f1")
        
        # Focus level setting
        tk.Label(win, text="Focus Level", font=("Arial", 14, "bold"), bg="#ecf0f1").pack(pady=(15, 10))
        
        v_level = tk.StringVar(value=self.focus_level)
        
        def update_warnings():
            level = v_level.get()
            if level == "High":
                self.max_warnings = 1
                self.focus_level = "High"
            elif level == "Mid":
                self.max_warnings = 2
                self.focus_level = "Mid"
            else:  # Low
                self.max_warnings = 3
                self.focus_level = "Low"
            self.current_warnings = 0
            self.update_warning_ui()
        
        frm_level = tk.Frame(win, bg="#ecf0f1")
        frm_level.pack(pady=5)
        
        tk.Radiobutton(frm_level, text="High - 1 warning before strike", 
                      variable=v_level, value="High", command=update_warnings,
                      font=("Arial", 11), bg="#ecf0f1").pack(anchor=tk.W, padx=20, pady=5)
        tk.Radiobutton(frm_level, text="Mid - 2 warnings before strike", 
                      variable=v_level, value="Mid", command=update_warnings,
                      font=("Arial", 11), bg="#ecf0f1").pack(anchor=tk.W, padx=20, pady=5)
        tk.Radiobutton(frm_level, text="Low - 3 warnings before strike", 
                      variable=v_level, value="Low", command=update_warnings,
                      font=("Arial", 11), bg="#ecf0f1").pack(anchor=tk.W, padx=20, pady=5)
        
        # Separator
        tk.Frame(win, height=2, bg="#bdc3c7").pack(fill=tk.X, padx=20, pady=15)
        
        # Display settings
        tk.Label(win, text="Display Settings", font=("Arial", 14, "bold"), bg="#ecf0f1").pack(pady=(10, 10))
        
        v_show_cam = tk.BooleanVar(value=self.show_camera)
        
        def update_show_cam():
            self.show_camera = v_show_cam.get()
        
        frm_display = tk.Frame(win, bg="#ecf0f1")
        frm_display.pack(pady=5)
        
        tk.Checkbutton(frm_display, text="Show camera feed on screen", 
                      variable=v_show_cam, command=update_show_cam,
                      font=("Arial", 11), bg="#ecf0f1").pack(anchor=tk.W, padx=20, pady=5)
        
        # Save button
        def save():
            update_warnings()
            update_show_cam()
            win.destroy()
        
        tk.Button(win, text="Save", command=save, bg="#3498db", fg="white", 
                 font=("Arial", 12, "bold"), width=15, height=2).pack(pady=20)

    def toggle_mic(self):
        """Toggle microphone on/off"""
        self.use_mic = not self.use_mic
        self.btn_mic.config(text="Mic ON" if self.use_mic else "Mic OFF", 
                           bg="#27ae60" if self.use_mic else "#e74c3c", 
                           fg="white")
        print(f"[DEBUG] Mic toggled: {self.use_mic}")  # Debug output

    def toggle_cam(self):
        """Toggle camera on/off"""
        self.use_cam = not self.use_cam
        self.btn_cam.config(text="Cam ON" if self.use_cam else "Cam OFF", 
                           bg="#27ae60" if self.use_cam else "#e74c3c", 
                           fg="white")
        print(f"[DEBUG] Cam toggled: {self.use_cam}")  # Debug output
    
    def toggle_timer(self):
        curr_time = time.time()
        self.timer_enabled = not self.timer_enabled
        self.btn_timer.config(text="Timer ON" if self.timer_enabled else "Timer OFF", 
                             bg="#3498db" if self.timer_enabled else "#95a5a6", 
                             fg="white")
        if not self.timer_enabled:
            # Pause timer manually
            if not self.is_timer_paused:
                self.timer_paused_at = curr_time
                self.is_timer_paused = True
        else:
            # Resume timer manually
            if self.is_timer_paused and self.timer_paused_at is not None:
                self.timer_paused_time += curr_time - self.timer_paused_at
                self.is_timer_paused = False
                self.timer_paused_at = None

    def start_monitoring(self):
        self.frame_home.pack_forget()
        self.frame_active.pack(fill="both", expand=True)
        self.is_monitoring = True
        self.current_streak_start = time.time()
        self.timer_paused_time = 0
        self.timer_paused_at = None
        self.is_timer_paused = False
        self.current_warnings = 0
        self.warning_history = []  # Reset warning history
        self.slacking_start_time = None
        self.last_punish_time = 0
        self.t_eyes_start = None
        self.t_yawn_start = None # 하품 타이머 초기화 추가
        self.update_warning_ui()

    def stop_monitoring(self):
        self.frame_active.pack_forget()
        self.frame_home.pack(fill="both", expand=True)
        self.is_monitoring = False
        # DC 모터 정지 (level 0 호출)
        activate_feedback(0)

    def update_warning_ui(self):
        msg = f"Warning: {self.current_warnings}/{self.max_warnings} ({self.focus_level})"
        self.canvas.itemconfigure(self.id_warning, text=msg)
        
        # Update warning stack display
        if self.warning_history:
            # Show last 5 warnings
            recent_warnings = self.warning_history[-5:]
            stack_text = "Warnings: " + " | ".join([f"{w['reason']}" for w in recent_warnings])
            if len(self.warning_history) > 5:
                stack_text += f" (+{len(self.warning_history) - 5} more)"
        else:
            stack_text = "Warnings: None"
        self.canvas.itemconfigure(self.id_warning_stack, text=stack_text)

    # ---------------------------------------------------------
    # Main loop (enhanced drawing logic)
    # ---------------------------------------------------------
    def update_loop(self):
        raw_frame = self.cam_thread.get_frame()
        
        if raw_frame is not None:
            # Process based on camera display setting
            if self.show_camera:
                # 1. Copy image for display
                disp_frame = raw_frame.copy()
                # Resize from original (640x480) -> Camera box size (320x240)
                disp_frame = cv2.resize(disp_frame, (self.cam_box_w, self.cam_box_h))
                
                if self.is_monitoring and self.use_cam:
                    if self.frame_count % self.analysis_interval == 0:
                        # Analyze and draw on disp_frame (small version for camera box)
                        self.run_analysis_and_draw(raw_frame, disp_frame, is_small=True)
                elif not self.use_cam and self.is_monitoring:
                    cv2.putText(disp_frame, "CAM OFF", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100,100,100), 2)
                
                img = ImageTk.PhotoImage(Image.fromarray(disp_frame))
                if self.cam_image_id is None:
                    self.cam_image_id = self.canvas.create_image(self.cam_box_x, self.cam_box_y, image=img, anchor=tk.NW)
                    self.canvas.tag_lower(self.cam_image_id)
                else:
                    self.canvas.itemconfig(self.cam_image_id, image=img)
                self.canvas.image = img
                
                # Show camera border
                if self.cam_border_id is not None:
                    self.canvas.itemconfig(self.cam_border_id, state='normal')
            else:
                # Camera display off - hide camera box
                if self.cam_image_id is not None:
                    self.canvas.delete(self.cam_image_id)
                    self.cam_image_id = None
                # Hide camera border
                if self.cam_border_id is not None:
                    self.canvas.itemconfig(self.cam_border_id, state='hidden')
                
                # Continue analysis if monitoring (just not visible on screen)
                if self.is_monitoring and self.use_cam:
                    if self.frame_count % self.analysis_interval == 0:
                        # Perform analysis only (skip drawing)
                        self.run_analysis_only(raw_frame) 

        if self.is_monitoring:
            # Calculate elapsed time (accounting for pauses, violations, and manual disable)
            if self.timer_enabled:
                if not self.is_timer_paused:
                    elapsed = time.time() - self.current_streak_start - self.timer_paused_time
                    streak = int(max(0, elapsed))
                else:
                    # Timer is paused due to violation, show last value
                    if hasattr(self, 'last_streak_value'):
                        streak = self.last_streak_value
                    else:
                        streak = 0
            else:
                # Timer is manually disabled, show last value
                if hasattr(self, 'last_streak_value'):
                    streak = self.last_streak_value
                else:
                    streak = 0
            self.last_streak_value = streak
            self.canvas.itemconfigure(self.id_timer, text=f"{streak//60:02}:{streak%60:02}")

        self.frame_count += 1
        self.root.after(30, self.update_loop)

    # ---------------------------------------------------------
    # ★ Core: Enhanced visualization analysis function
    # ---------------------------------------------------------
    def run_analysis_and_draw(self, raw_rgb, draw_frame, is_small=False):
        import mediapipe as mp
        curr_time = time.time()
        
        # Coordinate conversion ratio
        if is_small:
            # For small camera box (320x240)
            scale_x = self.cam_box_w / 640
            scale_y = self.cam_box_h / 480
        else:
            # For full screen (800x480)
            scale_x = 800 / 640
            scale_y = 480 / 480 

        # 1. AI analysis
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=raw_rgb)
        timestamp_ms = int(curr_time * 1000)
        mp_result = self.mp_handler.process(mp_image, timestamp_ms) # MediaPipeHandler 사용
        audio_data = self.audio_handler.get_features()
        
        # Status Extraction
        ear_val = mp_result.get('ear', 0.0)
        mar_val = mp_result.get('mar', 0.0)
        is_face_detected = mp_result.get('face_detected', False)
        is_mouth_covered = mp_result.get('cover', False)
        
        # Check person detection first (before EdgeTPU analysis)
        detected_person = is_face_detected
        if mp_result.get('objects'):
            for det in mp_result['objects']:
                if det.categories[0].category_name == 'person':
                    detected_person = True
                    break
        
        # EdgeTPU
        tpu_in = cv2.resize(raw_rgb, (self.tpu_w, self.tpu_h))
        input_data = tpu_in[np.newaxis, ...]
        self.inter.set_tensor(self.in_det["index"], input_data)
        self.inter.invoke()
        logits = self.inter.get_tensor(self.out_det["index"]).squeeze()
        current_action = self.labels.get(np.argmax(logits), "Unknown")
        
        # If no person detected, set action to "No Person"
        if not detected_person:
            current_action = "No Person"

        # ----------------------------------------
        # ★ [Visual] Draw detected object boxes
        # ----------------------------------------
        
        is_phone_detected = False
        if mp_result.get('objects'):
            for det in mp_result['objects']:
                bbox = det.bounding_box
                name = det.categories[0].category_name
                
                # Coordinate conversion
                x = int(bbox.origin_x * scale_x)
                y = int(bbox.origin_y * scale_y)
                w = int(bbox.width * scale_x)
                h = int(bbox.height * scale_y)

                color = (0, 255, 0) # person = green
                if name == 'cell phone': 
                    color = (255, 0, 0) # phone = red
                    is_phone_detected = True
                if name == 'person': detected_person = True

                cv2.rectangle(draw_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(draw_frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Update action if phone is detected
        if is_phone_detected:
            current_action = "Using Phone"
        
        # Get values for main info display
        db_val = audio_data.get('db', -99)

        # ----------------------------------------
        # 3. Violation detection logic
        # ----------------------------------------
        is_bad = False
        reason = ""
        
        # (A) Phone detection
        if is_phone_detected:
            is_bad = True
            reason = "PHONE"
        
        # (B) Drowsiness detection
        elif is_face_detected and ear_val < config.THRESHOLDS['EAR']:
             if self.t_eyes_start is None: self.t_eyes_start = curr_time
             elif (curr_time - self.t_eyes_start) > config.DURATION['EYES']:
                 is_bad = True; reason = "SLEEP(EYES)"
        else:
            self.t_eyes_start = None

        # (C) Yawn detection
        if is_face_detected and mar_val > config.THRESHOLDS['MAR']:
            if self.t_yawn_start is None: self.t_yawn_start = curr_time
            elif (curr_time - self.t_yawn_start) > config.DURATION['YAWN']:
                is_bad = True; reason = "YAWN"
        else:
            self.t_yawn_start = None
            
        # (D) Noise detection
        if self.use_mic and audio_data['is_speech']:
            is_bad = True; reason = "NOISY"
            
        # (E) Sleeping pose
        if current_action == "sleep(lean)":
             is_bad = True; reason = "SLEEP(POSE)"
             
        # (F) Mouth Covered
        if is_mouth_covered:
            is_bad = True; reason = "COVERED"
        
        # No Person (최종 상태는 No Person이 아닐 때만 유효함)
        if not detected_person:
            is_bad = False # 사람이 없으면 딴짓으로 처벌하지 않음.
            reason = "NO_PERSON"
        

        # ----------------------------------------
        # 4. Warning system update & Timer control
        # ----------------------------------------
        status_text = "Focusing..."
        status_color = "#2ecc71" # Green

        if is_bad:
            # Pause timer when violation detected
            if self.timer_enabled and not self.is_timer_paused:
                self.timer_paused_at = curr_time
                self.is_timer_paused = True
            
            # ★★★ 모든 위반 유형을 즉시 실행하도록 통합 (1.5초 대기 로직 제거) ★★★
            if reason != "NO_PERSON": 
                # config.SERVO_COOLDOWN (2.0초) 쿨다운 체크
                if (curr_time - self.last_punish_time > config.SERVO_COOLDOWN): 
                    self.current_warnings += 1
                    # Add to warning history
                    self.warning_history.append({'reason': reason, 'time': curr_time, 'count': self.current_warnings})
                    self.update_warning_ui()
                    
                    if self.current_warnings >= self.max_warnings:
                        activate_feedback(2) # Punishment (DC 모터 1초 회전)
                        self.last_punish_time = curr_time
                        self.current_warnings = 0
                        self.warning_history = []  # Reset history after punishment
                        self.update_warning_ui()
                        status_text = "PUNISHED!"; status_color = "red"
                    else:
                        status_text = f"WARNING (+1, {reason})"; status_color = "orange"
                else:
                    # 쿨다운 중
                    status_text = f"{reason} DETECTED! (Cooldown)"; status_color = "orange"
            
            self.slacking_start_time = None # 1.5초 대기 로직이 없으므로 초기화

        else:
            # Resume timer when focusing again
            if self.timer_enabled and self.is_timer_paused:
                self.timer_paused_time += curr_time - self.timer_paused_at
                self.is_timer_paused = False
                self.timer_paused_at = None
            
            self.slacking_start_time = None
            # DC 모터 정지 유지 (level 0 호출)
            activate_feedback(0)
            
            if not detected_person:
                status_text = "No Person Detected"; status_color = "#95a5a6" # 회색

        # Update Canvas status text
        self.canvas.itemconfigure(self.id_status, text=status_text, fill=status_color)
        
        # Update info display on left side
        if self.is_monitoring:
            mar_val = mp_result.get('mar', 0.0) # MAR 값 가져오기
            db_val = audio_data.get('db', -99)
            
            self.canvas.itemconfigure(self.id_info_ear, text=f"EAR: {ear_val:.2f}")
            self.canvas.itemconfigure(self.id_info_mar, text=f"MAR: {mar_val:.2f}")
            self.canvas.itemconfigure(self.id_info_action, text=f"Action: {current_action}")
            # Show audio info with mic status
            mic_status = "ON" if self.use_mic else "OFF"
            self.canvas.itemconfigure(self.id_info_audio, text=f"Audio: {db_val:.1f} dB (Mic: {mic_status})")
            self.canvas.itemconfigure(self.id_info_person, text=f"Person: {'Yes' if detected_person else 'No'}")

    def run_analysis_only(self, raw_rgb):
        """Perform analysis only without screen display (when camera display is OFF)"""
        import mediapipe as mp
        curr_time = time.time()
        
        # 1. AI analysis
        mp_image = mp.Image(format=mp.ImageFormat.SRGB, data=raw_rgb)
        timestamp_ms = int(curr_time * 1000)
        mp_result = self.mp_handler.process(mp_image, timestamp_ms)
        audio_data = self.audio_handler.get_features()
        
        # Status Extraction
        ear_val = mp_result.get('ear', 0.0)
        mar_val = mp_result.get('mar', 0.0)
        is_face_detected = mp_result.get('face_detected', False)
        is_mouth_covered = mp_result.get('cover', False)
        
        # Check person detection first (before EdgeTPU analysis)
        detected_person = is_face_detected
        if mp_result.get('objects'):
            for det in mp_result['objects']:
                if det.categories[0].category_name == 'person':
                    detected_person = True
                    break
        
        # EdgeTPU
        tpu_in = cv2.resize(raw_rgb, (self.tpu_w, self.tpu_h))
        input_data = tpu_in[np.newaxis, ...]
        self.inter.set_tensor(self.in_det["index"], input_data)
        self.inter.invoke()
        logits = self.inter.get_tensor(self.out_det["index"]).squeeze()
        current_action = self.labels.get(np.argmax(logits), "Unknown")
        
        # If no person detected, set action to "No Person"
        if not detected_person:
            current_action = "No Person"

        # Check for phone detection and update action
        is_phone_detected = False
        for det in mp_result.get('objects', []):
            if det.categories[0].category_name == 'cell phone':
                is_phone_detected = True
                break
        
        # Update action if phone is detected
        if is_phone_detected:
            current_action = "Using Phone"

        # 2. Violation detection logic
        is_bad = False
        reason = ""

        # (A) Phone detection
        if is_phone_detected:
            is_bad = True
            reason = "PHONE"
        
        # (B) Drowsiness detection
        elif is_face_detected and ear_val < config.THRESHOLDS['EAR']:
             if self.t_eyes_start is None: 
                 self.t_eyes_start = curr_time
             elif (curr_time - self.t_eyes_start) > config.DURATION['EYES']:
                 is_bad = True
                 reason = "SLEEP(EYES)"
        else:
            self.t_eyes_start = None

        # (C) Yawn detection
        if is_face_detected and mar_val > config.THRESHOLDS['MAR']:
            if self.t_yawn_start is None: self.t_yawn_start = curr_time
            elif (curr_time - self.t_yawn_start) > config.DURATION['YAWN']:
                is_bad = True; reason = "YAWN"
        else:
            self.t_yawn_start = None

        # (D) Noise detection
        if self.use_mic and audio_data['is_speech']:
            is_bad = True
            reason = "NOISY"

        # (E) Sleeping pose
        if current_action == "sleep(lean)":
             is_bad = True
             reason = "SLEEP(POSE)"

        # (F) Mouth Covered
        if is_mouth_covered:
            is_bad = True; reason = "COVERED"
        
        # No Person
        if not detected_person:
            is_bad = False
            reason = "NO_PERSON"

        # 3. Warning system update & Timer control
        status_text = "Focusing..."
        status_color = "#2ecc71"  # Green

        if is_bad:
            # Pause timer when violation detected
            if self.timer_enabled and not self.is_timer_paused:
                self.timer_paused_at = curr_time
                self.is_timer_paused = True
            
            # ★★★ 모든 위반 유형을 즉시 실행하도록 통합 (1.5초 대기 로직 제거) ★★★
            if reason != "NO_PERSON": 
                if (curr_time - self.last_punish_time > config.SERVO_COOLDOWN):
                    self.current_warnings += 1
                    # Add to warning history
                    self.warning_history.append({'reason': reason, 'time': curr_time, 'count': self.current_warnings})
                    self.update_warning_ui()
                    
                    if self.current_warnings >= self.max_warnings:
                        activate_feedback(2); self.last_punish_time = curr_time; self.current_warnings = 0; self.warning_history = []; self.update_warning_ui()
                        status_text = "PUNISHED!"; status_color = "red"
                    else:
                        status_text = f"WARNING (+1, {reason})"; status_color = "orange"
                else:
                    status_text = f"{reason} DETECTED! (Cooldown)"; status_color = "orange"
            
            self.slacking_start_time = None #1.5초 대기 로직이 없으므로 초기화

        else:
            # Resume timer when focusing again
            if self.timer_enabled and self.is_timer_paused:
                self.timer_paused_time += curr_time - self.timer_paused_at
                self.is_timer_paused = False
                self.timer_paused_at = None
            
            self.slacking_start_time = None
            activate_feedback(0)
            
            if reason == "NO_PERSON":
                status_text = "No Person Detected"; status_color = "#95a5a6"

        # Update Canvas status text
        self.canvas.itemconfigure(self.id_status, text=status_text, fill=status_color)
        
        # Update info display on left side
        if self.is_monitoring:
            db_val = audio_data.get('db', -99)
            
            self.canvas.itemconfigure(self.id_info_ear, text=f"EAR: {ear_val:.2f}")
            self.canvas.itemconfigure(self.id_info_mar, text=f"MAR: {mar_val:.2f}")
            self.canvas.itemconfigure(self.id_info_action, text=f"Action: {current_action}")
            # Show audio info with mic status
            mic_status = "ON" if self.use_mic else "OFF"
            self.canvas.itemconfigure(self.id_info_audio, text=f"Audio: {db_val:.1f} dB (Mic: {mic_status})")
            self.canvas.itemconfigure(self.id_info_person, text=f"Person: {'Yes' if detected_person else 'No'}")


    def on_close(self):
        self.cam_thread.stop()
        self.audio_handler.close()
        self.mp_handler.close()
        # DC 모터 정리 코드를 호출
        cleanup_hardware() 
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartFocusApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()