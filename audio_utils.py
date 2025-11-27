import pyaudio
import numpy as np
import torch
import threading
import time
import config

class AudioAnalyzer:
    def __init__(self):
        # --- 설정 로드 ---
        AC = config.AUDIO_CONFIG
        self.MIC_RATE = AC['MIC_RATE']       # 48000
        self.SAMPLE_RATE = AC['SAMPLE_RATE'] # 16000
        self.CHUNK_SIZE = AC['CHUNK_SIZE']   # 512
        self.MIC_INDEX = AC['MIC_INDEX']
        
        # 다운샘플링 비율 (예: 3)
        self.DOWN_SAMPLE_RATIO = int(self.MIC_RATE / self.SAMPLE_RATE)
        
        print(f"[Audio] 초기화: 마이크({self.MIC_RATE}Hz) -> VAD({self.SAMPLE_RATE}Hz)")
        
        # 1. Silero-VAD (말소리 감지용) - 이것만 남김!
        self.vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
        print("[Audio] VAD 모델 준비 완료! (YAMNet 제거됨)")

        self.lock = threading.Lock()
        self.running = True
        
        # 결과 저장소 (detected_sound, sound_conf 제거)
        self.features = {
            'is_speech': False,
            'db': -100.0,
            'detected_sound': 'None' # main.py 호환성을 위해 None으로 고정
        }
        
        self.p = pyaudio.PyAudio()
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16, 
                channels=1, 
                rate=self.MIC_RATE, 
                input=True, 
                frames_per_buffer=self.CHUNK_SIZE * self.DOWN_SAMPLE_RATIO, 
                input_device_index=self.MIC_INDEX
            )
        except Exception as e:
            print(f"[Error] 마이크 열기 실패: {e}")
            raise e
        
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        # 한 번에 읽을 크기
        read_chunk = self.CHUNK_SIZE * self.DOWN_SAMPLE_RATIO 
        
        while self.running:
            try:
                # 1. 읽기
                data = self.stream.read(read_chunk, exception_on_overflow=False)
                audio_int16 = np.frombuffer(data, dtype=np.int16)
                
                # 2. 다운샘플링 (48000 -> 16000)
                if self.DOWN_SAMPLE_RATIO > 1:
                    audio_int16 = audio_int16[::self.DOWN_SAMPLE_RATIO]

                # 3. 정규화
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                
                # [dB 계산]
                rms = np.sqrt(np.mean(audio_float32**2))
                if rms > 0:
                    db = 20 * np.log10(rms)
                else:
                    db = -100

                # [Silero VAD] 말소리 감지
                vad_prob = self.vad_model(torch.from_numpy(audio_float32), self.SAMPLE_RATE).item()
                is_speech = vad_prob > 0.5
                
                with self.lock:
                    self.features['is_speech'] = is_speech
                    self.features['db'] = float(db)
                    self.features['detected_sound'] = 'None' # 항상 None

            except Exception:
                pass
            time.sleep(0.01)

    def get_features(self):
        with self.lock:
            return self.features.copy()

    def close(self):
        self.running = False
        self.thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()