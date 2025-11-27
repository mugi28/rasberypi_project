import argparse
import time
import cv2
import numpy as np
import sys
import mediapipe as mp

# MediaPipe Solutions 임포트
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Task API 임포트
import mediapipe.tasks as tasks 
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode

# 모듈 임포트
import config
from tpu_utils import make_camera, load_labels, make_interpreter
from audio_utils import AudioAnalyzer
from servo import setup_hardware, activate_feedback, cleanup_hardware

# Face Mesh 부분 추적을 위한 연결 목록
FACE_MESH_EYES_LIPS_CONNECTIONS = (
    mp_face_mesh.FACEMESH_LIPS |
    mp_face_mesh.FACEMESH_LEFT_EYE |
    mp_face_mesh.FACEMESH_LEFT_IRIS |
    mp_face_mesh.FACEMESH_RIGHT_EYE |
    mp_face_mesh.FACEMESH_RIGHT_IRIS |
    mp_face_mesh.FACEMESH_LEFT_EYEBROW |
    mp_face_mesh.FACEMESH_RIGHT_EYEBROW
)

# ================== 보조 함수: MediaPipe 결과 처리 ==================

def calculate_ear(landmarks, ear_indices):
    """EAR (Eye Aspect Ratio) 계산"""
    if not landmarks:
        return 0.0

    # config.IDXS['L_EYE'] 또는 R_EYE의 6개 랜드마크 사용
    p = np.array([[landmarks[i].x, landmarks[i].y] for i in ear_indices])
    
    # 6개 점 중 3쌍의 유클리드 거리 계산
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    
    # EAR 공식
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_mar(landmarks, mar_indices):
    """MAR (Mouth Aspect Ratio) 계산 (입 벌림) - Normalized Ratio (수정됨)"""
    if not landmarks:
        return 0.0

    # 1. 수직 거리 (A) - 기존 config.IDXS['MOUTH_V'] 사용 (예: 13, 14)
    p_vert = np.array([[landmarks[i].x, landmarks[i].y] for i in mar_indices])
    A = np.linalg.norm(p_vert[0] - p_vert[1])

    # 2. 수평 거리 (B) - 입술 양 끝 (Face Mesh  랜드마크 61, 291)
    MOUTH_H_INDICES = [61, 291]
    p_horiz = np.array([[landmarks[i].x, landmarks[i].y] for i in MOUTH_H_INDICES])
    B = np.linalg.norm(p_horiz[0] - p_horiz[1])
    
    # 3. MAR Ratio 계산 (A / B)
    if B == 0: return 0.0 
    
    mar = A / B
    return mar

def get_hand_cover_status(landmarks, face_landmarks, hand_tips_indices):
    """손으로 입을 가렸는지 확인 (간단화)"""
    if not face_landmarks or not landmarks:
        return False
        
    # 입술 중앙 랜드마크 (예: 13번)
    mouth_center_idx = 13
    mouth_x = face_landmarks[mouth_center_idx].x
    
    # 손가락 끝 랜드마크 (4, 8, 12, 16, 20)
    tip_coords = np.array([[landmarks[i].x, landmarks[i].y] for i in hand_tips_indices])
    
    # 입술 근처에 손가락 끝이 있는지 확인
    # 손가락 끝과 입술 중심의 평균 거리 계산
    distances = np.linalg.norm(tip_coords - np.array([mouth_x, face_landmarks[mouth_center_idx].y]), axis=1)
    
    # 모든 손가락 끝이 입 근처에 있다면 입 가림으로 간주
    if np.mean(distances) < config.THRESHOLDS['COVER']:
        return True
        
    return False

# ================== Args ==================
def parse_args():
    ap = argparse.ArgumentParser(description="Pi + EdgeTPU + MediaPipe Solutions + Audio + DC Motor")
    ap.add_argument("--model", type=str, default="./model_edgetpu.tflite", help="EdgeTPU model path")
    ap.add_argument("--labels", type=str, default="./labels.txt", help="labels.txt path")
    ap.add_argument("--res", type=str, default="1640x1232", help="Resolution")
    ap.add_argument("--topk", type=int, default=3, help="Top-K predictions")
    ap.add_argument("--headless", action="store_true", help="No GUI mode")
    return ap.parse_args()

# ================== Main Loop ==================
def main():
    args = parse_args()
    
    # 1. 설정 및 초기화
    labels = load_labels(args.labels)
    
    try:
        picam2, (cam_w, cam_h) = make_camera(args.res)
    except Exception as e:
        print(f"❌ 카메라 초기화 실패: {e}")
        return

    try:
        inter = make_interpreter(args.model)
        in_det = inter.get_input_details()[0]
        out_det = inter.get_output_details()[0]
        in_h, in_w = in_det["shape"][1], in_det["shape"][2]
    except Exception as e:
        print(f"❌ EdgeTPU 초기화 실패: {e}")
        return
    
    # --- MediaPipe Solutions 초기화 ---
    print("[Init] MediaPipe Solutions 및 Task API 초기화 중...")
    
    # Solutions (Face Mesh, Hands)
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, 
        refine_landmarks=True,
        min_detection_confidence=0.7, # 감지 민감도 상향 조정
        min_tracking_confidence=0.7)  # 추적 민감도 상향 조정

    hands = mp_hands.Hands(
        model_complexity=1, 
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # Task API Object Detector
    object_detector = None 
    try:
        base_options = tasks.BaseOptions(model_asset_path=config.OBJECT_MODEL)
        detector_options = vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=RunningMode.IMAGE,
            max_results=5,
            score_threshold=config.THRESHOLDS['OBJECT'])
        object_detector = vision.ObjectDetector.create_from_options(detector_options)
    except Exception as e:
        print(f"❌ MediaPipe Task API Object Detector 초기화 실패: {e}. 'object_detector'를 사용할 수 없습니다.")
        object_detector = None 

    # --- 오디오 및 하드웨어 초기화 ---
    try:
        audio_handler = AudioAnalyzer()
    except Exception as e:
        print(f"❌ 오디오 초기화 실패: {e}")
        # Solutions 객체 정리
        face_mesh.close()
        hands.close()
        if object_detector: object_detector.close()
        return

    print("[Init] DC 모터(Hardware) 초기화 중...")
    setup_hardware()

    print("[INFO] 멀티모달 모니터링 시작... (종료: 'q' 키)")
    
    if not args.headless:
        cv2.namedWindow("Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Monitor", 960, 720)

    # === 변수 초기화 ===
    start_time = time.time()
    fps_time = cv2.getTickCount()
    frames = 0
    fps = 0.0
    
    # 시간 체크 변수
    t_eyes_start = None
    t_yawn_start = None
    t_action_start = None
    prev_action = None
    
    last_hit_time = 0

    while True:
        # 1. 이미지 캡처 및 RGB 변환 (MediaPipe용)
        frame_rgb = picam2.capture_array()
        curr_time = time.time()  
        
        # --- (A) Face Mesh (EAR/MAR) ---
        face_results = face_mesh.process(frame_rgb)
        
        # --- (B) Hand Tracking (Mouth Cover) ---
        hand_results = hands.process(frame_rgb)
        
        # --- (C) Object Detection (Phone) ---
        # Task API 사용: mp.Image 객체로 변환 필요
        object_results = None
        if object_detector:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            object_results = object_detector.detect(mp_image)
        
        # --- (D) Audio ---
        audio_data = audio_handler.get_features()              

        # [터미널 확인용]
        print(f"\r🎤 Speech: {audio_data['is_speech']} | dB: {audio_data['db']:.1f} | Sound: {audio_data['detected_sound']}    ", end="")
        
        # --- (E) EdgeTPU 추론 (Action Recognition) ---
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        input_img = cv2.resize(frame_bgr, (in_w, in_h))
        
        # ... (EdgeTPU 추론 로직은 동일) ...
        if in_det["dtype"] == np.float32:
            input_data = (input_img.astype(np.float32) / 255.0)[None, ...]
        else:
            input_data = input_img.astype(np.uint8)[None, ...]
            
        inter.set_tensor(in_det["index"], input_data)
        inter.invoke()
        
        logits = inter.get_tensor(out_det["index"]).squeeze()
        if logits.ndim == 0: logits = np.array([logits])
        logits = logits.astype(np.float32)
        ex = np.exp(logits - np.max(logits))
        probs = ex / (np.sum(ex) + 1e-9)
        
        top_idx = np.argmax(probs)
        current_action = labels.get(top_idx, "Unknown")
        
        # === 3. 상태 감지 및 처벌 판단 ===
        
        status_text = "Normal"
        color = (0, 255, 0)
        trigger_punishment = False
        sound_alert = ""
        is_action_stable = False 
        
        # --- Solutions 결과 추출 및 변수 계산 ---
        
        is_person_present = False
        ear = 0.0
        mar = 0.0
        is_mouth_covered = False
        is_phone_detected = False
        
        face_landmarks = None
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0].landmark
            ear = (calculate_ear(face_landmarks, config.IDXS['L_EYE']) + 
                   calculate_ear(face_landmarks, config.IDXS['R_EYE'])) / 2
            mar = calculate_mar(face_landmarks, config.IDXS['MOUTH_V']) # 수정된 MAR 함수 호출
            
            # 사람이 인식된 것으로 간주
            is_person_present = True 
            
            # 입 가림 감지 (hand landmarks 필요)
            if hand_results.multi_hand_landmarks:
                hand_landmarks = hand_results.multi_hand_landmarks[0].landmark
                is_mouth_covered = get_hand_cover_status(
                    hand_landmarks, face_landmarks, config.IDXS['HAND_TIPS']
                )

        # Object Detection for Phone (Task API 사용)
        if object_results and object_results.detections:
            # Task API 결과는 .detections[i].categories[j].category_name 형식
            for detection in object_results.detections:
                for category in detection.categories:
                    # config 파일의 OBJECT_MODEL에 따라 'cell phone' 레이블이 있는지 확인
                    if category.category_name == 'cell phone':
                        is_phone_detected = True
                        break
                if is_phone_detected:
                    break

        # EdgeTPU에서 'person'이 감지되면 사람 있는 것으로 간주
        if current_action == 'person' or (object_results and object_results.detections):
             is_person_present = True

        # 사람이 없으면 모든 감지 로직 건너뜐 후 변수 초기화
        if not is_person_present:
            status_text = "NO PERSON DETECTED"
            color = (128, 128, 128) # 회색
            t_eyes_start = None
            t_yawn_start = None
            t_action_start = None
            
        else:
            # --- 사람이 있을 때만 감지 로직 실행 ---
            
            # [Audio] 말소리/소음
            is_talking = False
            if audio_data['is_speech']:
                sound_alert = "Talking Detected!"
                is_talking = True
                
            if audio_data['detected_sound'] != 'None' and audio_data['sound_conf'] > 0.3:
                sound_alert = f"SOUND: {audio_data['detected_sound']}"

            # [Visual] 눈 감음
            is_eyes_closed = False
            if ear > 0.01 and ear < config.THRESHOLDS['EAR']:
                if t_eyes_start is None: t_eyes_start = curr_time
                elif (curr_time - t_eyes_start) > config.DURATION['EYES']:
                    is_eyes_closed = True
            else:
                t_eyes_start = None

            # [Visual] 하품
            is_yawning = False
            if mar > config.THRESHOLDS['MAR']: # 이제 이 MAR은 Normalized Ratio입니다.
                if t_yawn_start is None: t_yawn_start = curr_time
                elif (curr_time - t_yawn_start) > config.DURATION['YAWN']:
                    is_yawning = True
            else:
                t_yawn_start = None

            # [Visual] 행동 안정화
            if current_action != prev_action:
                t_action_start = curr_time
                prev_action = current_action
            is_action_stable = (t_action_start and (curr_time - t_action_start) > config.DURATION['ACTION'])

            # --- 처벌 조건 체크 ---
            
            # (A) 스마트폰
            if is_phone_detected:
                status_text = "PHONE DETECTED! (MP)"
                color = (0, 0, 255)
                trigger_punishment = True

            # (B) 졸음
            elif is_eyes_closed:
                status_text = "DROWSINESS"
                color = (0, 0, 255)
                trigger_punishment = True
                if "Snoring" in audio_data['detected_sound']:
                    status_text = "DEEP SLEEP (Snoring)"

            # (C) 떠들기
            elif is_talking:
                status_text = "NO TALKING!"
                color = (0, 0, 255)
                trigger_punishment = True

            # (D) 엎드려 잠
            elif current_action == "sleep(lean)" and is_action_stable:
                status_text = "SLEEPING (Leaning)"
                color = (0, 0, 255)
                trigger_punishment = True

            # (E) 하품
            elif is_yawning:
                status_text = "Yawning"
                color = (0, 165, 255)
                trigger_punishment = True
            
            if is_mouth_covered:
                status_text += " + Mouth Covered"


        # === 5. DC 모터 작동 ===
        if trigger_punishment:
            # 쿨다운이 지나면 처벌 실행
            if (curr_time - last_hit_time > config.SERVO_COOLDOWN):
                print(f"\n[ACTION] 🛑 처벌 발동: {status_text} / {sound_alert} -> DC Motor!")
                activate_feedback(2)
                last_hit_time = curr_time
                cv2.rectangle(frame_bgr, (0, 0), (cam_w, cam_h), (0, 0, 255), 15)
        else:
            activate_feedback(0) # 대기


        # === 6. 화면 출력 ===
        if not args.headless:
            frames += 1
            if frames >= 10:
                t1 = cv2.getTickCount()
                fps = frames * cv2.getTickFrequency() / (t1 - fps_time)
                fps_time = t1
                frames = 0
            
            # Solutions Drawing: 얼굴 드로잉 (눈, 입 주변만)
            if face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame_bgr,
                    landmark_list=face_results.multi_face_landmarks[0],
                    connections=FACE_MESH_EYES_LIPS_CONNECTIONS, 
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1))

            
            if hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame_bgr,
                    landmark_list=hand_results.multi_hand_landmarks[0],
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2))
            
            # 객체 박스 (Task API 결과 사용)
            if object_results and object_results.detections:
                from mediapipe.tasks.python.components.containers import Detection
                for detection in object_results.detections:
                    # Bounding Box 정보 추출 (정규화된 좌표)
                    bbox = detection.bounding_box
                    x_min = int(bbox.origin_x)
                    y_min = int(bbox.origin_y)
                    x_max = x_min + int(bbox.width)
                    y_max = y_min + int(bbox.height)
                    
                    # 라벨 및 신뢰도
                    category_name = detection.categories[0].category_name
                    score = detection.categories[0].score
                    
                    box_col = (255, 255, 0)
                    if category_name == 'cell phone': box_col = (0, 0, 255)
                    elif category_name == 'person': box_col = (0, 255, 0)
                    
                    cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), box_col, 2)
                    cv2.putText(frame_bgr, f"{category_name} ({score:.2f})", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_col, 2)

            # 정보 오버레이
            cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 오디오 바
            bar_len = int((audio_data['db'] + 60) * 5)
            bar_len = max(0, min(bar_len, 200))
            cv2.rectangle(frame_bgr, (10, 80), (10 + bar_len, 95), (0, 255, 0), -1)

            if is_person_present:
                voice_status = "Voice: YES" if is_talking else "Voice: NO"
                voice_col = (0, 0, 255) if is_talking else (200, 200, 200)
                cv2.putText(frame_bgr, voice_status, (220, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.6, voice_col, 2)
                
                if sound_alert:
                    cv2.putText(frame_bgr, sound_alert, (320, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    if "SOUND:" in sound_alert: audio_handler.reset_sound_status()

            eye_dur = (curr_time - t_eyes_start) if t_eyes_start else 0.0
            cv2.putText(frame_bgr, f"EAR: {ear:.2f} ({eye_dur:.1f}s)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            act_disp = current_action if is_action_stable else f"{current_action}?"
            cv2.putText(frame_bgr, f"Pose: {act_disp}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)
            
            cv2.putText(frame_bgr, status_text, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            
            cv2.imshow("Monitor", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    print("[Exit] 시스템 종료 중...")
    audio_handler.close()
    
    # MediaPipe Solutions 객체 정리
    face_mesh.close()
    hands.close()
    if object_detector: object_detector.close()
    
    picam2.stop()
    cleanup_hardware()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass