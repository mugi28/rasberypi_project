import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import config

class MediaPipeHandler:
    def __init__(self):
        print("🔹 MediaPipe (Face / Hand / Object) 초기화 중...")
        try:
            # 1. Face Landmarker 설정
            base_opts_face = python.BaseOptions(model_asset_path=config.FACE_MODEL)
            opts_face = vision.FaceLandmarkerOptions(
                base_options=base_opts_face,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5)
            self.face_landmarker = vision.FaceLandmarker.create_from_options(opts_face)

            # 2. Hand Landmarker 설정
            base_opts_hand = python.BaseOptions(model_asset_path=config.HAND_MODEL)
            opts_hand = vision.HandLandmarkerOptions(
                base_options=base_opts_hand,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3)
            self.hand_landmarker = vision.HandLandmarker.create_from_options(opts_hand)

            # 3. Object Detector 설정
            base_opts_obj = python.BaseOptions(model_asset_path=config.OBJECT_MODEL)
            opts_obj = vision.ObjectDetectorOptions(
                base_options=base_opts_obj,
                running_mode=vision.RunningMode.VIDEO,
                max_results=5,  # 최대 5개의 물체 감지
                score_threshold=config.THRESHOLDS['OBJECT']
            )
            self.object_detector = vision.ObjectDetector.create_from_options(opts_obj)

            print("✅ MediaPipe 로드 성공")
        except Exception as e:
            print(f"❌ MediaPipe 로드 에러: {e}")
            raise e

    def _dist(self, p1, p2):
        """ 두 점 사이의 유클리드 거리 계산 """
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def get_ratios(self, lm):
        """ EAR(눈)과 MAR(입) 비율 계산 """
        # EAR (Eye Aspect Ratio)
        def get_ear(idxs):
            v1 = self._dist(lm[idxs[1]], lm[idxs[5]])
            v2 = self._dist(lm[idxs[2]], lm[idxs[4]])
            h = self._dist(lm[idxs[0]], lm[idxs[3]])
            return (v1 + v2) / (2.0 * h) if h != 0 else 0.0
        
        ear = (get_ear(config.IDXS['L_EYE']) + get_ear(config.IDXS['R_EYE'])) / 2.0
        
        # MAR (Mouth Aspect Ratio) - [13, 14] 수직, [61, 291] 수평
        v = self._dist(lm[config.IDXS['MOUTH_V'][0]], lm[config.IDXS['MOUTH_V'][1]])
        h = self._dist(lm[config.IDXS['MOUTH_H'][0]], lm[config.IDXS['MOUTH_H'][1]])
        mar = v / h if h != 0 else 0.0
        
        return ear, mar # ★ MAR 추가

    def check_cover(self, lm, hand_res):
        """ 손이 입을 가리고 있는지 확인 """
        if not hand_res.hand_landmarks: return False
        
        min_dist = 1.0
        # 13번: 윗입술 중앙 랜드마크
        mouth_center = lm[13]
        
        for hand in hand_res.hand_landmarks:
            for idx in config.IDXS['HAND_TIPS']:
                # 손가락 끝과 입 사이 거리 계산
                d = self._dist(mouth_center, hand[idx])
                if d < min_dist: min_dist = d
        
        return min_dist < config.THRESHOLDS['COVER']

    def process(self, mp_image, timestamp):
        """ 이미지 처리 및 결과 반환 """
        # 3가지 모델 병렬 실행 (Video 모드)
        face_res = self.face_landmarker.detect_for_video(mp_image, timestamp)
        hand_res = self.hand_landmarker.detect_for_video(mp_image, timestamp)
        obj_res = self.object_detector.detect_for_video(mp_image, timestamp)
        
        result = {
            'ear': 0.0, 
            'mar': 0.0, # ★ MAR 초기화 추가
            'cover': False, 
            'face_detected': False,
            'objects': [] # 감지된 객체 리스트
        }
        
        # 얼굴 분석 (졸음/하품)
        if face_res.face_landmarks:
            result['face_detected'] = True
            lm = face_res.face_landmarks[0]
            result['ear'], result['mar'] = self.get_ratios(lm) # ★ MAR 값 저장
            result['cover'] = self.check_cover(lm, hand_res)
            
        # 객체 분석 결과 저장
        if obj_res.detections:
            result['objects'] = obj_res.detections

        return result
    
    def close(self):
        """ 리소스 해제 """
        self.face_landmarker.close()
        self.hand_landmarker.close()
        self.object_detector.close()