# servo.py (극약 처방 버전: 쓸 때만 연결하고 바로 끊기)

import time

# 하드웨어 라이브러리 (없으면 가상 모드)
try:
    from gpiozero import Servo
    IS_RPI = True
except ImportError:
    IS_RPI = False

# 핀 번호 설정
SERVO_PIN = 26

# 시간 설정 (사용자 환경에 맞게 조절하세요)
HIT_TIME = 1    # 타격하러 가는 시간
RETURN_TIME = 0.5   # 복귀하는 시간
WAIT_TIME = 1     # 중간 대기 시간

def setup_hardware():
    # 여기서는 아무것도 안 합니다. (때릴 때만 켤 거니까요)
    if IS_RPI:
        print(f"[HW] 서보 대기 모드 (GPIO {SERVO_PIN}) - 평소엔 전원 차단됨")

def hit_action():
    """
    모터를 켜고 -> 때리고 -> 끄는 함수
    """
    if not IS_RPI:
        return

    # 1. ★ [핵심] 동작 직전에 모터 연결 (이때만 전기가 통함)
    servo = Servo(SERVO_PIN)

    try:
        # 2. 타격 (ㅡ)
        # print("[HW] 타격!")
        servo.min()
        time.sleep(HIT_TIME)

        # 3. 정지
        servo.mid()
        time.sleep(WAIT_TIME)

        # 4. 복귀 (|)
        # print("[HW] 복귀!")
        servo.max()
        time.sleep(RETURN_TIME)
        
    finally:
        # 5. ★ [핵심] 일이 끝나자마자 모터 연결 해제 (SW적으로 선을 뽑는 효과)
        # 이렇게 하면 대기 상태에서 신호가 아예 없으므로 절대 안 움직입니다.
        servo.close()


def activate_feedback(level):
    """
    main.py에서 호출하는 함수
    """
    if level == 2:
        print(f"[HW] 딴짓 감지! 타격 시작...")
        hit_action() # 타격 함수 호출
        print("[HW] 타격 완료.")
        
    elif level == 0:
        # 평소에는 아무것도 안 합니다. 
        # (이미 hit_action 끝날 때 servo.close()를 했기 때문에 완벽히 정지 상태임)
        pass

def cleanup_hardware():
    # 이미 닫혀있으므로 할 게 없음
    pass