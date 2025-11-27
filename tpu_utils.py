# tpu_utils.py
import sys
import platform
from tflite_runtime.interpreter import Interpreter, load_delegate

def make_camera(res):
    """Picamera2 초기화 및 설정"""
    try:
        from picamera2 import Picamera2
        w, h = [int(x) for x in res.lower().split("x")]
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(
            main={"size": (w, h), "format": "RGB888"}
        ))
        picam2.start()
        sw, sh = picam2.sensor_resolution
        # 전체 화각 사용 (ScalerCrop)
        picam2.set_controls({"ScalerCrop": (0, 0, sw, sh)})
        return picam2, (w, h)
    except ImportError:
        print("❌ Picamera2 라이브러리가 설치되지 않았습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 카메라 초기화 오류: {e}")
        sys.exit(1)

def load_labels(path):
    """라벨 파일 로드"""
    labels = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                s = line.strip()
                if not s: continue
                parts = s.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    labels[int(parts[0])] = parts[1]
                else:
                    labels[i] = s
        return labels
    except FileNotFoundError:
        print(f"❌ 라벨 파일({path})을 찾을 수 없습니다.")
        sys.exit(1)

def make_interpreter(model_path):
    """Edge TPU Interpreter 생성 (TPU 실패 시 종료)"""
    try:
        delegate = load_delegate('libedgetpu.so.1')
        print("✅ Edge TPU Delegate 로드 성공")
    except Exception as e:
        print(f"❌ Edge TPU 로드 실패: {e}")
        print("   Coral USB Accelerator가 연결되어 있는지 확인하세요.")
        sys.exit(1)

    try:
        inter = Interpreter(model_path=model_path, experimental_delegates=[delegate])
        inter.allocate_tensors()
        return inter
    except Exception as e:
        print(f"❌ 모델 로드 실패({model_path}): {e}")
        sys.exit(1)