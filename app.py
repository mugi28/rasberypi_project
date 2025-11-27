import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import time
import threading
import numpy as np

# ★ Picamera2 library
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
            config = self.picam2.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            self.picam2.configure(config)
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
        self.show_camera = True 
        
        # Focus level setting
        self.focus_level = "Mid"
        self.max_warnings = 2
        self.current_warnings = 0
        self.warning_history = []
        
        # Timer settings
        self.timer_enabled = True 
        self.current_streak_start = time.time()
        self.timer_paused_time = 0 
        self.timer_paused_at = None 
        self.is_timer_paused = False
        self.last_punish_time = 0
        
        # [수정] t_eyes_start 변수는 더 이상 쓰이지 않지만 호환성을 위해 남겨둠
        self.t_eyes_start = None

        # AI initialization
        print("[Init] Loading AI engine...")
        setup_hardware()
        self.audio_handler = AudioAnalyzer()
        self.mp_handler = MediaPipeHandler()
        
        self.labels = load_labels("labels.txt")
        self.inter = make_interpreter("model_edgetpu2.tflite")
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

        self.cam_box_x = self.width - 340 
        self.cam_box_y = 80 
        self.cam_box_w = 320 
        self.cam_box_h = 240 
        
        self.cam_border_id = self.canvas.create_rectangle(
            self.cam_box_x - 2, self.cam_box_y - 2,
            self.cam_box_x + self.cam_box_w + 2, self.cam_box_y + self.cam_box_h + 2,
            outline="white", width=2
        )
        
        self.id_status = self.canvas.create_text(self.width//2, 440, text="Ready", font=("Arial", 20, "bold"), fill="white")
        self.id_timer = self.canvas.create_text(self.width - 100, 50, text="00:00", font=("Arial", 30, "bold"), fill="white")
        self.id_warning = self.canvas.create_text(100, 440, text=f"Warning: 0/{self.max_warnings} ({self.focus_level})", font=("Arial", 16, "bold"), fill="yellow")
        
        self.info_text_y = 100
        self.id_info_ear = self.canvas.create_text(20, self.info_text_y, text="EAR: -- / MAR: --", font=("Arial", 12), fill="white", anchor=tk.W) 
        self.id_info_action = self.canvas.create_text(20, self.info_text_y + 25, text="Action: --", font=("Arial", 12), fill="white", anchor=tk.W)
        self.id_info_audio = self.canvas.create_text(20, self.info_text_y + 50, text="Audio: -- dB", font=("Arial", 12), fill="white", anchor=tk.W)
        self.id_info_person = self.canvas.create_text(20, self.info_text_y + 75, text="Person: --", font=("Arial", 12), fill="white", anchor=tk.W)
        
        self.warning_stack_y = self.info_text_y + 110
        self.id_warning_stack = self.canvas.create_text(20, self.warning_stack_y, text="Warnings: None", font=("Arial", 11), fill="yellow", anchor=tk.W)

    def open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.geometry("350x350")
        win.configure(bg="#ecf0f1")
        
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
        
        tk.Frame(win, height=2, bg="#bdc3c7").pack(fill=tk.X, padx=20, pady=15)
        
        tk.Label(win, text="Display Settings", font=("Arial", 14, "bold"), bg="#ecf0f1").pack(pady=(10, 10))
        
        v_show_cam = tk.BooleanVar(value=self.show_camera)
        
        def update_show_cam():
            self.show_camera = v_show_cam.get()
        
        frm_display = tk.Frame(win, bg="#ecf0f1")
        frm_display.pack(pady=5)
        
        tk.Checkbutton(frm_display, text="Show camera feed on screen", 
                      variable=v_show_cam, command=update_show_cam,
                      font=("Arial", 11), bg="#ecf0f1").pack(anchor=tk.W, padx=20, pady=5)
        
        def save():
            update_warnings()
            update_show_cam()
            win.destroy()
        
        tk.Button(win, text="Save", command=save, bg="#3498db", fg="white", 
                 font=("Arial", 12, "bold"), width=15, height=2).pack(pady=20)

    def toggle_mic(self):
        self.use_mic = not self.use_mic
        self.btn_mic.config(text="Mic ON" if self.use_mic else "Mic OFF", 
                           bg="#27ae60" if self.use_mic else "#e74c3c", fg="white")

    def toggle_cam(self):
        self.use_cam = not self.use_cam
        self.btn_cam.config(text="Cam ON" if self.use_cam else "Cam OFF", 
                           bg="#27ae60" if self.use_cam else "#e74c3c", fg="white")
    
    def toggle_timer(self):
        curr_time = time.time()
        self.timer_enabled = not self.timer_enabled
        self.btn_timer.config(text="Timer ON" if self.timer_enabled else "Timer OFF", 
                             bg="#3498db" if self.timer_enabled else "#95a5a6", fg="white")
        if not self.timer_enabled:
            if not self.is_timer_paused:
                self.timer_paused_at = curr_time
                self.is_timer_paused = True
        else:
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
        self.warning_history = []
        self.last_punish_time = 0
        self.update_warning_ui()

    def stop_monitoring(self):
        self.frame_active.pack_forget()
        self.frame_home.pack(fill="both", expand=True)
        self.is_monitoring = False
        activate_feedback(0)

    def update_warning_ui(self):
        msg = f"Warning: {self.current_warnings}/{self.max_warnings} ({self.focus_level})"
        self.canvas.itemconfigure(self.id_warning, text=msg)
        
        if self.warning_history:
            recent_warnings = self.warning_history[-5:]
            stack_text = "Warnings: " + " | ".join([f"{w['reason']}" for w in recent_warnings])
            if len(self.warning_history) > 5:
                stack_text += f" (+{len(self.warning_history) - 5} more)"
        else:
            stack_text = "Warnings: None"
        self.canvas.itemconfigure(self.id_warning_stack, text=stack_text)

    def update_loop(self):
        raw_frame = self.cam_thread.get_frame()
        
        if raw_frame is not None:
            if self.show_camera:
                disp_frame = raw_frame.copy()
                disp_frame = cv2.resize(disp_frame, (self.cam_box_w, self.cam_box_h))
                
                if self.is_monitoring and self.use_cam:
                    if self.frame_count % self.analysis_interval == 0:
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
                
                if self.cam_border_id is not None:
                    self.canvas.itemconfig(self.cam_border_id, state='normal')
            else:
                if self.cam_image_id is not None:
                    self.canvas.delete(self.cam_image_id)
                    self.cam_image_id = None
                if self.cam_border_id is not None:
                    self.canvas.itemconfig(self.cam_border_id, state='hidden')
                
                if self.is_monitoring and self.use_cam:
                    if self.frame_count % self.analysis_interval == 0:
                        self.run_analysis_only(raw_frame) 
                
                elif self.is_monitoring and not self.use_cam:
                    if self.frame_count % self.analysis_interval == 0:
                        self.run_analysis_only(raw_frame)
                        
        if self.is_monitoring and (raw_frame is None or self.frame_count % self.analysis_interval != 0):
            if self.use_mic:
                audio_data = self.audio_handler.get_features()
                if audio_data['is_speech']:
                    self.check_and_process_audio_violation(audio_data)

        if self.is_monitoring:
            if self.timer_enabled:
                if not self.is_timer_paused:
                    elapsed = time.time() - self.current_streak_start - self.timer_paused_time
                    streak = int(max(0, elapsed))
                else:
                    if hasattr(self, 'last_streak_value'):
                        streak = self.last_streak_value
                    else:
                        streak = 0
            else:
                if hasattr(self, 'last_streak_value'):
                    streak = self.last_streak_value
                else:
                    streak = 0
            self.last_streak_value = streak
            self.canvas.itemconfigure(self.id_timer, text=f"{streak//60:02}:{streak%60:02}")

        self.frame_count += 1
        self.root.after(30, self.update_loop)
    
    def check_and_process_audio_violation(self, audio_data):
        curr_time = time.time()
        is_bad = False
        reason = ""
        
        if self.use_mic and audio_data['is_speech']:
            is_bad = True; reason = "NOISY"
            
        if is_bad:
            if self.timer_enabled and not self.is_timer_paused:
                self.timer_paused_at = curr_time
                self.is_timer_paused = True
                
            if (curr_time - self.last_punish_time > config.SERVO_COOLDOWN):
                self.current_warnings += 1
                self.warning_history.append({'reason': reason, 'time': curr_time, 'count': self.current_warnings})
                self.update_warning_ui()
                
                if self.current_warnings >= self.max_warnings:
                    activate_feedback(2)
                    self.last_punish_time = curr_time
                    self.current_warnings = 0
                    self.warning_history = []
                    self.update_warning_ui()
                    status_text = "PUNISHED!"
                    status_color = "red"
                else:
                    status_text = f"WARNING (+1)"
                    status_color = "orange"
            else:
                status_text = f"{reason} DETECTED! (Cooldown)"
                status_color = "orange"

        else:
            if self.timer_enabled and self.is_timer_paused:
                self.timer_paused_time += curr_time - self.timer_paused_at
                self.is_timer_paused = False
                self.timer_paused_at = None
            
            activate_feedback(0)
            status_text = "Focusing..."
            status_color = "#2ecc71"

        self.canvas.itemconfigure(self.id_status, text=status_text, fill=status_color)
        self.canvas.itemconfigure(self.id_info_audio, text=f"Audio: {audio_data.get('db', -99):.1f} dB (Mic: ON)")
        self.canvas.itemconfigure(self.id_info_ear, text="EAR: -- / MAR: --")
        self.canvas.itemconfigure(self.id_info_action, text="Action: --")
        self.canvas.itemconfigure(self.id_info_person, text="Person: No")


    # ---------------------------------------------------------
    # ★ Core: run_analysis_and_draw
    # ---------------------------------------------------------
    def run_analysis_and_draw(self, raw_rgb, draw_frame, is_small=False):
        import mediapipe as mp
        curr_time = time.time()
        
        if is_small:
            scale_x = self.cam_box_w / 640
            scale_y = self.cam_box_h / 480
        else:
            scale_x = 800 / 640
            scale_y = 480 / 480 

        # 1. AI analysis
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=raw_rgb)
        mp_result = self.mp_handler.process(mp_image, int(curr_time * 1000))
        audio_data = self.audio_handler.get_features() 
        
        detected_person_vis = False
        if mp_result.get('objects'):
            for det in mp_result['objects']:
                if det.categories[0].category_name == 'person':
                    detected_person_vis = True
                    break
        if mp_result.get('ear', 0.0) > 0.01:
            detected_person_vis = True
        
        current_action = "No Person"
        is_phone_detected = False
        ear_val = 0.0
        mar_val = 0.0
        
        if detected_person_vis:
            tpu_in = cv2.resize(raw_rgb, (self.tpu_w, self.tpu_h))
            input_data = tpu_in[np.newaxis, ...]
            self.inter.set_tensor(self.in_det["index"], input_data)
            self.inter.invoke()
            logits = self.inter.get_tensor(self.out_det["index"]).squeeze()
            current_action = self.labels.get(np.argmax(logits), "Unknown")
            
            if mp_result.get('objects'):
                for det in mp_result['objects']:
                    bbox = det.bounding_box
                    name = det.categories[0].category_name
                    
                    x = int(bbox.origin_x * scale_x)
                    y = int(bbox.origin_y * scale_y)
                    w = int(bbox.width * scale_x)
                    h = int(bbox.height * scale_y)
    
                    color = (0, 255, 0) # person = green
                    if name == 'cell phone': 
                        color = (255, 0, 0) # phone = red
                        is_phone_detected = True
    
                    cv2.rectangle(draw_frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(draw_frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            if is_phone_detected:
                current_action = "Using Phone"

            ear_val = mp_result.get('ear', 0.0)
            mar_val = mp_result.get('mar', 0.0)

        # ----------------------------------------
        # 3. Violation detection logic (수정됨)
        # ----------------------------------------
        is_bad = False
        reason = ""

        # 1. 시각적/자세 위반
        if detected_person_vis:
            if is_phone_detected:
                is_bad = True; reason = "PHONE"
    
            if mar_val > config.THRESHOLDS['MAR']:
                 is_bad = True; reason = "YAWN"
    
            if current_action == "sleep(lean)":
                 is_bad = True; reason = "SLEEP(POSE)"
                 
            # Distracted 로직
            if "distracted" in current_action.lower():
                 is_bad = True; reason = "DISTRACTED"
        
            # ★ [수정] 5초 이상 눈 감았을 때만 작동하도록 변경
            if ear_val < config.THRESHOLDS['EAR']:
                # 눈을 처음 감았으면 시간 기록 시작
                if self.t_eyes_start is None:
                    self.t_eyes_start = curr_time
                # 감은 지 얼마나 지났는지 계산
                elif (curr_time - self.t_eyes_start) >= config.DURATION['EYES']:
                    is_bad = True
                    reason = "SLEEP(EYES)"
            else:
                # 눈을 뜨면 타이머 리셋
                self.t_eyes_start = None

        # 2. 음성 위반
        db_val = audio_data.get('db', -99)
        if self.use_mic and audio_data['is_speech']:
            is_bad = True; reason = "NOISY"
            
        # ----------------------------------------
        # 4. Warning system update (즉시 반응)
        # ----------------------------------------
        status_text = "Focusing..."
        status_color = "#2ecc71" # Green

        if is_bad:
            if self.timer_enabled and not self.is_timer_paused:
                self.timer_paused_at = curr_time
                self.is_timer_paused = True
            
            # Threshold 대기 시간 없음 -> 즉시 체크
            if (curr_time - self.last_punish_time > config.SERVO_COOLDOWN):
                self.current_warnings += 1
                self.warning_history.append({
                    'reason': reason,
                    'time': curr_time,
                    'count': self.current_warnings
                })
                self.update_warning_ui()
                
                if self.current_warnings >= self.max_warnings:
                    activate_feedback(2) # Punishment
                    self.last_punish_time = curr_time
                    self.current_warnings = 0
                    self.warning_history = []
                    self.update_warning_ui()
                    status_text = "PUNISHED!"
                    status_color = "red"
                else:
                    status_text = f"WARNING (+1)"
                    status_color = "orange"
            else:
                status_text = f"{reason} DETECTED! (Cooldown)"
                status_color = "orange"

        else:
            if self.timer_enabled and self.is_timer_paused:
                self.timer_paused_time += curr_time - self.timer_paused_at
                self.is_timer_paused = False
                self.timer_paused_at = None
            
            activate_feedback(0)

        self.canvas.itemconfigure(self.id_status, text=status_text, fill=status_color)
        
        if self.is_monitoring:
            detected_person_check = detected_person_vis
            self.canvas.itemconfigure(self.id_info_ear, text=f"EAR: {ear_val:.2f} / MAR: {mar_val:.2f}")
            self.canvas.itemconfigure(self.id_info_action, text=f"Action: {current_action}")
            mic_status = "ON" if self.use_mic else "OFF"
            self.canvas.itemconfigure(self.id_info_audio, text=f"Audio: {db_val:.1f} dB (Mic: {mic_status})")
            self.canvas.itemconfigure(self.id_info_person, text=f"Person: {'Yes' if detected_person_check else 'No'}")

    # ---------------------------------------------------------
    # ★ Core: run_analysis_only
    # ---------------------------------------------------------
    def run_analysis_only(self, raw_rgb):
        import mediapipe as mp
        curr_time = time.time()
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=raw_rgb)
        mp_result = self.mp_handler.process(mp_image, int(curr_time * 1000))
        audio_data = self.audio_handler.get_features()
        
        detected_person_vis = False
        ear_val = 0.0
        mar_val = 0.0
        current_action = "No Cam"
        is_phone_detected = False

        if self.use_cam:
            if mp_result.get('objects'):
                for det in mp_result['objects']:
                    if det.categories[0].category_name == 'person':
                        detected_person_vis = True
                        break
            if mp_result.get('ear', 0.0) > 0.01:
                detected_person_vis = True
            
            if detected_person_vis:
                tpu_in = cv2.resize(raw_rgb, (self.tpu_w, self.tpu_h))
                input_data = tpu_in[np.newaxis, ...]
                self.inter.set_tensor(self.in_det["index"], input_data)
                self.inter.invoke()
                logits = self.inter.get_tensor(self.out_det["index"]).squeeze()
                current_action = self.labels.get(np.argmax(logits), "Unknown")

                for det in mp_result.get('objects', []):
                    if det.categories[0].category_name == 'cell phone':
                        is_phone_detected = True
                        current_action = "Using Phone"
                        break
                
                ear_val = mp_result.get('ear', 0.0)
                mar_val = mp_result.get('mar', 0.0)

        is_bad = False
        reason = ""

        if detected_person_vis:
            if is_phone_detected:
                is_bad = True; reason = "PHONE"
            
            
                
            if mar_val > config.THRESHOLDS['MAR']:
                 is_bad = True; reason = "YAWN"

            if current_action == "sleep(lean)":
                 is_bad = True; reason = "SLEEP(POSE)"

            # Distracted 로직
            if "distracted" in current_action.lower():
                 is_bad = True; reason = "DISTRACTED"

           # ★ [수정] 5초 이상 눈 감았을 때만 작동하도록 변경
            if ear_val < config.THRESHOLDS['EAR']:
                # 눈을 처음 감았으면 시간 기록 시작
                if self.t_eyes_start is None:
                    self.t_eyes_start = curr_time
                # 감은 지 얼마나 지났는지 계산
                elif (curr_time - self.t_eyes_start) >= config.DURATION['EYES']:
                    is_bad = True
                    reason = "SLEEP(EYES)"
            else:
                # 눈을 뜨면 타이머 리셋
                self.t_eyes_start = None

        db_val = audio_data.get('db', -99)
        if self.use_mic and audio_data['is_speech']:
            is_bad = True; reason = "NOISY"

        status_text = "Focusing..."
        status_color = "#2ecc71"

        if is_bad:
            if self.timer_enabled and not self.is_timer_paused:
                self.timer_paused_at = curr_time
                self.is_timer_paused = True
            
            if (curr_time - self.last_punish_time > config.SERVO_COOLDOWN):
                self.current_warnings += 1
                self.warning_history.append({
                    'reason': reason,
                    'time': curr_time,
                    'count': self.current_warnings
                })
                self.update_warning_ui()
                
                if self.current_warnings >= self.max_warnings:
                    activate_feedback(2)
                    self.last_punish_time = curr_time
                    self.current_warnings = 0
                    self.warning_history = []
                    self.update_warning_ui()
                    status_text = "PUNISHED!"
                    status_color = "red"
                else:
                    status_text = f"WARNING (+1)"
                    status_color = "orange"
            else:
                status_text = f"{reason} DETECTED! (Cooldown)"
                status_color = "orange"
        else:
            if self.timer_enabled and self.is_timer_paused:
                self.timer_paused_time += curr_time - self.timer_paused_at
                self.is_timer_paused = False
                self.timer_paused_at = None
            
            activate_feedback(0)

        self.canvas.itemconfigure(self.id_status, text=status_text, fill=status_color)
        
        if self.is_monitoring:
            self.canvas.itemconfigure(self.id_info_ear, text=f"EAR: {ear_val:.2f} / MAR: {mar_val:.2f}")
            self.canvas.itemconfigure(self.id_info_action, text=f"Action: {current_action}")
            mic_status = "ON" if self.use_mic else "OFF"
            self.canvas.itemconfigure(self.id_info_audio, text=f"Audio: {db_val:.1f} dB (Mic: {mic_status})")
            self.canvas.itemconfigure(self.id_info_person, text=f"Person: {'Yes' if detected_person_vis else 'No'}")

    def on_close(self):
        self.cam_thread.stop()
        self.audio_handler.close()
        self.mp_handler.close()
        cleanup_hardware() 
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartFocusApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()      