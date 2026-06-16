# 🧺 Laundry Symbol Detection Model

의류 세탁 라벨 이미지에서 **세탁 기호를 자동으로 탐지**하고, 탐지된 기호를 바탕으로 세탁 방법을 안내하기 위한 YOLO 기반 객체 탐지 프로젝트입니다.

<p align="center">
  <img src="https://img.shields.io/badge/Model-YOLOv8-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Export-TFLite-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Target-Android-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Dataset-Roboflow-purple?style=for-the-badge" />
</p>

---

## 📌 Project Overview

이 프로젝트는 옷에 붙어 있는 세탁 라벨 사진을 입력받아 다음과 같은 세탁 기호를 탐지합니다.

- 세탁 가능 / 손세탁 / 물세탁 불가
- 표백 가능 / 표백 불가
- 다림질 가능 / 다림질 불가
- 자연건조 / 건조기 사용 / 건조기 사용 불가
- 드라이클리닝 / 웻클리닝 관련 기호

최종 목표는 학습된 모델을 **TFLite 모델로 변환**하여 Android 앱에서 사용하는 것입니다.

---

## 🛠 Tech Stack

| Category | Tool |
|---|---|
| Model | YOLOv8 / Ultralytics |
| Dataset | Roboflow |
| Training Environment | Google Colab |
| Export Format | TensorFlow Lite `.tflite` |
| Target Platform | Android |
| Language | Python, Kotlin/Java |

---

## 📁 Dataset

데이터셋은 Roboflow에서 관리하며, YOLOv8 형식으로 export하여 사용했습니다.

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 70
```

클래스는 크게 두 국가/표준 기준으로 구성되어 있습니다.

```text
ER_ : Europe 기준 세탁 기호
KR_ : Korea 기준 세탁 기호
```

예시 클래스명:

```text
KR_wash_warm_30_neutral
KR_iron_low_temp
KR_hang_dry_in_shade
ER_dry_clean_chemical
ER_dry_clean_hydrocarbon
ER_no_wet_clean
```

> ⚠️ 참고  
> `wash-care-symbols` 클래스는 실제 세탁 기호라기보다 잘못 들어간 라벨일 가능성이 있습니다.  
> 모델 성능을 더 안정적으로 만들려면 해당 클래스가 실제로 필요한지 확인하는 것이 좋습니다.

---

## 🏷 Label Naming Rule

라벨 이름은 다음 규칙을 기준으로 정리했습니다.

```text
[국가/표준]_[기능]_[세부조건]
```

예시:

```text
KR_wash_warm_30_neutral
KR_iron_low_temp
KR_hang_dry_in_shade
ER_dry_clean_chemical_mild
ER_tumble_dry_low_heat
```

### Prefix

| Prefix | Meaning |
|---|---|
| `KR` | Korea care symbol |
| `ER` | Europe care symbol |

### Common Keywords

| Keyword | Meaning |
|---|---|
| `wash` | 물세탁 |
| `hand_wash` | 손세탁 |
| `no_wash` | 물세탁 불가 |
| `bleach` | 표백 |
| `iron` | 다림질 |
| `dry_clean` | 드라이클리닝 |
| `tumble_dry` | 건조기 사용 |
| `flat_dry` | 뉘어서 건조 |
| `line_dry` / `hang_dry` | 걸어서 건조 |
| `in_shade` | 그늘에서 건조 |
| `mild` | 약한 처리 |
| `very_mild` | 매우 약한 처리 |

---

## 🚀 Model Training

Google Colab에서 Ultralytics YOLO를 사용하여 학습했습니다.

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="/content/drive/MyDrive/laundry_symbols.yolov8/data.yaml",
    epochs=50,
    imgsz=640,
    device=0,
    project="/content/drive/MyDrive/yolo_results",
    name="laundry_symbols_exp1"
)
```

학습 결과는 다음 경로에 저장됩니다.

```text
/content/drive/MyDrive/yolo_results/laundry_symbols_exp1/
```

주요 가중치 파일:

```text
weights/best.pt
weights/last.pt
```

---

## 🔍 Prediction Test

학습된 모델을 사용해 테스트 이미지를 예측할 수 있습니다.

```python
from ultralytics import YOLO
import cv2
from google.colab.patches import cv2_imshow

best_model = YOLO("/content/drive/MyDrive/yolo_results/laundry_symbols_exp1/weights/best.pt")

test_results = best_model.predict(
    source="/content/drive/MyDrive/android termproject/test2.jpg",
    save=True,
    conf=0.25
)

save_dir = test_results[0].save_dir
predicted_img = cv2.imread(f"{save_dir}/test2.jpg")
cv2_imshow(predicted_img)
```

예측 결과는 다음 정보를 포함합니다.

```text
box 좌표
confidence score
class id
```

예시:

```text
class_id = 67
confidence = 0.88
label = KR_wash_warm_30_neutral
```

---

## 📦 Export to TFLite

Android 앱에서 사용하기 위해 YOLO 모델을 TFLite 형식으로 변환합니다.

```python
from ultralytics import YOLO

model = YOLO("/content/drive/MyDrive/yolo_results/laundry_symbols_exp1/weights/best.pt")

model.export(
    format="tflite",
    imgsz=640,
    nms=True
)
```

변환 후 생성되는 파일 예시:

```text
best_float32.tflite
```

Android에서 사용하기 위해 보통 다음 위치에 넣습니다.

```text
app/src/main/assets/best_float32.tflite
app/src/main/assets/labels.txt
```

---

## 🧾 labels.txt

TFLite 모델은 문자열 라벨을 직접 출력하지 않고, 보통 `class_id`를 출력합니다.

따라서 Android 앱에서 다음처럼 매핑해야 합니다.

```text
class_id → labels.txt[class_id] → 실제 라벨명
```

예시:

```text
67 → KR_wash_warm_30_neutral
49 → KR_iron_low_temp
46 → KR_hang_dry_in_shade
```

`labels.txt` 생성 코드:

```python
import yaml

data_yaml = "/content/drive/MyDrive/laundry_symbols.yolov8/data.yaml"
labels_path = "/content/drive/MyDrive/yolo_results/laundry_symbols_exp1/labels.txt"

with open(data_yaml, "r") as f:
    data = yaml.safe_load(f)

names = data["names"]

if isinstance(names, dict):
    names = [names[i] for i in sorted(names.keys())]

with open(labels_path, "w") as f:
    for name in names:
        f.write(name + "\n")

print("labels.txt 생성 완료:", labels_path)
```

---

## 📱 Android Usage Flow

Android 앱에서는 다음 흐름으로 모델을 사용합니다.

```text
1. Camera 또는 Gallery에서 이미지 입력
2. Bitmap을 640x640 크기로 전처리
3. TFLite 모델 실행
4. 출력값에서 box, confidence, class_id 추출
5. labels.txt를 이용해 class_id를 라벨명으로 변환
6. 라벨명에 맞는 세탁 안내 문구 표시
```

예시 데이터 클래스:

```kotlin
data class Detection(
    val label: String,
    val confidence: Float,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
)
```

라벨명을 설명으로 바꾸는 예시:

```kotlin
val careDescription = mapOf(
    "KR_wash_warm_30_neutral" to "약 30도에서 중성세제로 세탁하세요.",
    "KR_iron_low_temp" to "저온으로 다림질하세요.",
    "KR_hang_dry_in_shade" to "그늘에서 옷걸이에 걸어 말리세요.",
    "KR_no_tumble_dry" to "건조기 사용은 피하세요."
)
```

---

## ✅ Example Output

모델이 세탁 라벨 이미지를 분석하면 다음과 같은 결과를 얻을 수 있습니다.

```text
KR_wash_warm_30_neutral 0.88
KR_iron_low_temp 0.92
KR_hang_dry_in_shade 0.91
```

이를 앱에서는 다음처럼 사용자 친화적인 문장으로 변환할 수 있습니다.

```text
- 약 30도에서 중성세제로 세탁하세요.
- 저온으로 다림질하세요.
- 그늘에서 옷걸이에 걸어 말리세요.
```

---

## ⚠️ Notes

- TFLite 모델만으로 라벨명이 자동 출력되는 것은 아닙니다.
- 모델 출력의 `class_id`를 `labels.txt`와 매핑해야 합니다.
- `labels.txt`의 순서는 학습 당시 `data.yaml`의 `names` 순서와 반드시 같아야 합니다.
- 클래스가 너무 많으면 학습이 느려질 수 있으므로, 필요하다면 `KR_` 또는 `ER_` 기준으로 데이터셋을 분리할 수 있습니다.
- CPU 학습은 매우 느리므로 Colab GPU 사용을 권장합니다.

---

## 🧠 Future Improvements

- 잘못 들어간 클래스 제거
- KR / ER 데이터셋 분리 학습
- 비슷한 기호 클래스 데이터 추가
- 흐릿하거나 잘린 이미지 제거
- Android 실시간 카메라 탐지 기능 추가
- 탐지된 세탁 기호를 자연어 세탁 가이드로 변환

---

## 👩‍💻 Author

**seooyy**

Laundry care symbol detection project for Android term project.

