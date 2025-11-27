import pyaudio
import numpy as np
import config  # config.py 설정 불러오기
import time
import os

# --- 설정 로드 ---
try:
    AC = config.AUDIO_CONFIG
    MIC_INDEX = AC['MIC_INDEX']
    MIC_RATE = AC['MIC_RATE']
    CHUNK = AC['CHUNK_SIZE']
except AttributeError:
    print("❌ 오류: config.py 파일을 찾을 수 없거나 설정이 잘못되었습니다.")
    exit()

def test_microphone():
    p = pyaudio.PyAudio()

    print("\n" + "="*40)
    print(f"🎤 마이크 테스트 시작")
    print(f"   - 장치 번호: {MIC_INDEX}")
    print(f"   - 샘플 레이트: {MIC_RATE} Hz")
    print("="*40)
    print("소리를 내보세요! (게이지가 움직여야 합니다)\n")

    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=MIC_RATE,
            input=True,
            input_device_index=MIC_INDEX,
            frames_per_buffer=CHUNK
        )
    except Exception as e:
        print(f"\n❌ 마이크 열기 실패: {e}")
        print("config.py의 'MIC_INDEX'나 'MIC_RATE'가 맞는지 확인하세요.")
        return

    try:
        while True:
            # 1. 데이터 읽기
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)

            # 2. 소리 크기(RMS) 계산
            rms = np.sqrt(np.mean(audio_data**2))
            
            # 3. 데시벨(dB) 변환
            if rms > 0:
                db = 20 * np.log10(rms)
            else:
                db = -100

            # 4. 시각화 (막대 그래프 그리기)
            # 소리가 없으면 db는 보통 -50 ~ -40, 시끄러우면 -20 ~ 0
            # 시각화를 위해 게이지 계산
            bar_len = int((db + 60) / 2) 
            if bar_len < 0: bar_len = 0
            bar = "█" * bar_len
            
            # 한 줄에 계속 출력 (애니메이션 효과)
            print(f"\r🔊 소리 크기: {db:.1f} dB  |{bar:<30}|", end="")
            
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\n🛑 테스트 종료.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    test_microphone()