# AI- 실제 블랙박스 환경을 위한 YOLOv11 기반 도로 위험물 탐지 고도화 및 전처리 한계 분석


## 목차
- [1. 프로젝트 개요 (Abstract)]
- [2. 연구 배경 및 문제 정의 (Introduction)]
- [3. 데이터셋 및 환경 분석 (Data Analysis)]
- [4. 제안 방법론 및 코드 구현 (Methodology & Code)]
- [5. 실험 결과 및 심층 분석 (Results & Analysis)]
- [6. 결론 및 향후 과제 (Conclusion & Future Work)]

----

## 1. 프로젝트 개요 (Abstract)

> **팀원**: 이찬민(23619019), 김우준(22619005), 박채원(24619019), 한지은(24619040)

본 프로젝트는 주행 중 치명적인 사고를 유발하는 도로 위 위험물(포트홀, 균열 등)을 실시간으로 탐지하기 위해 최신 객체 탐지 알고리즘인 **YOLOv11**을 적용한 연구입니다. 

특히 본 연구는 단순히 모델의 성능을 높이는 것을 넘어, **"정제된 해외 오픈소스 데이터셋으로 학습된 모델이, 왜 빛 반사와 노이즈가 심한 국내 블랙박스 실환경에서는 성능이 급감하는가?"**라는 도메인 갭(Domain Gap) 문제에 집중했습니다. 

이를 해결하기 위해 컴퓨터 비전 기반의 전처리 기법(ROI Crop, CLAHE 등)을 파이프라인에 적용해보았으나, 이러한 인위적인 픽셀 조작이 오히려 모델의 원근감 인식을 방해하고 노이즈를 증폭시킨다는 실증적 한계를 확인했습니다. 결론적으로 실제 상용화 레벨의 블랙박스 탐지 시스템을 위해서는 알고리즘 튜닝을 넘어선 **국내 규격에 맞는 데이터 중심(Data-centric)의 고도화**가 필수적임을 제안합니다.

---

## 2. 연구 배경 및 문제 정의 (Introduction)

### 2.1 연구 배경
매년 포트홀(Pothole) 및 도로 균열(Crack)로 인한 타이어 파손 및 교통사고가 급증하고 있습니다. 특히 운전자의 시야가 극도로 제한되는 야간이나 우천 시, 블랙박스나 내비게이션을 통해 위험물을 사전에 감지하고 경고해주는 시스템의 필요성이 대두되고 있습니다.

### 2.2 기존 기술의 문제점 (Domain Gap)
AI 기술이 발전했음에도 현재 상용화된 블랙박스에서 포트홀 탐지가 어려운 이유는 **학습 환경과 실전 환경의 좁힐 수 없는 차이** 때문입니다.
1. **학습 환경 (A급 데이터)**: 공개된 연구용 데이터셋은 대부분 정지 상태에서 촬영되었거나, 정면 앵글에 화질이 매우 깨끗합니다.
2. **실전 환경 (블랙박스)**: 주행 중 발생하는 렌즈 블러(Blur), 차량 앞유리의 난반사, 와이퍼 움직임, 저조도 등 환경 노이즈가 극심합니다. 

본 연구는 이 간극을 확인하고, 실전 블랙박스 화면에서도 탐지가 가능하도록 데이터 전처리 파이프라인을 구축해 검증하는 것을 목표로 삼았습니다.

---

## 3. 데이터셋 및 환경 분석 (Data Analysis)

### 3.1 사용 데이터셋 구성
글로벌 오픈소스 데이터 플랫폼인 Kaggle에서 제공하는 공인 데이터셋을 베이스라인으로 활용했습니다.
* **데이터셋명**: Road Damage Dataset: Potholes, Cracks and Manholes
* **버전 및 시기**: Version 4 (2024~2025)
* **데이터 규모**: 총 2,009장의 주행 환경 원본 이미지 (Raw Image)
* **라벨링 객체 (총 4,737개)**
  * `Crack` (균열): 2,519개
  * `Pothole` (포트홀): 1,261개
  * `Manhole` (맨홀): 957개

### 3.2 Yaml 클래스 불일치 한계 (Label Mismatch)
학습 데이터셋(Kaggle)과 테스트 데이터셋(직접 수집한 국내 블랙박스 데이터) 간의 `Yaml` 구성 파일 내 클래스 인덱스 순서가 일치하지 않는 문제가 발생했습니다. 이를 코드로 강제 변환할 경우 바운딩 박스 라벨링의 치명적인 왜곡이 발생할 우려가 있어, 정량적 Test mAP 측정 대신 **Val 데이터셋 기준의 정량 평가**와 **실제 영상 추론을 통한 정성 평가**를 병행하는 전략을 택했습니다.

---

## 4. 제안 방법론 및 코드 구현 (Methodology & Code)

데이터 환경의 차이를 알고리즘적으로 극복하기 위해 `Dataset ➔ Data Processing (ROI / CLAHE) ➔ Model (YOLOv11) ➔ Inference` 구조의 파이프라인을 설계했습니다. 

<img width="742" height="336" alt="KakaoTalk_20260616_152939433" src="https://github.com/user-attachments/assets/e5cf32a0-1eb2-4d82-8db1-893626ea9076" />


### 4.1. 관심 영역 추출 (ROI Crop)
블랙박스 영상의 상단 절반은 보통 하늘, 가로수, 건물 등 탐지와 전혀 무관한 배경입니다. 이 배경이 모델의 연산량을 늘리고 오탐지를 유발한다고 가설을 세워, **이미지의 하단 60% (실제 도로면)만 잘라내어 모델에 입력**하는 ROI Crop 기법을 구현했습니다.

```python
import cv2

def apply_roi_crop(image):
    """
    원본 이미지에서 불필요한 상단 배경을 제거하고 도로가 위치한 하단 영역만 추출
    """
    height, width = image.shape[:2]
    
    # y축 기준 상단 40%는 버리고, 40% 지점부터 맨 아래(height)까지 슬라이싱
    start_y = int(height * 0.4) 
    roi_image = image[start_y:height, 0:width]
    
    return roi_image
```

### 4.2. 적응형 히스토그램 평활화 (CLAHE)
어두운 아스팔트와 포트홀의 명암(Contrast) 차이가 뚜렷하지 않아 탐지를 못하는 문제를 해결하기 위해 도입했습니다. 전체 이미지의 밝기를 한 번에 올리면 빛 번짐이 심해지므로, 이미지를 작은 그리드(8x8)로 나누어 국지적으로 대비를 강조하는 CLAHE 기법을 적용했습니다.

```python
import cv2

def apply_clahe(image):
    """
    이미지의 명암만 조절하기 위해 LAB 색상 공간으로 변환 후 밝기(L) 채널에 CLAHE 적용
    """
    # 1. BGR 이미지를 LAB 색상 공간으로 변환
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # 2. 채널 분리 (L: 밝기, A: 녹-적, B: 청-황)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 3. CLAHE 객체 생성 (대비 제한값 2.0 지정하여 과도한 노이즈 증폭 방지)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # 4. 평활화된 L 채널을 기존 색상 채널과 병합 후 다시 BGR로 복원
    merged = cv2.merge((cl, a_channel, b_channel))
    enhanced_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    return enhanced_img
```

적용 후 분석: 포트홀의 윤곽은 진해졌지만, 치명적인 부작용이 발생했습니다. 아스팔트 특유의 자글자글한 질감과 앞유리에 반사된 미세한 빛 반사 노이즈까지 함께 명암이 증폭되면서, 모델이 멀쩡한 도로를 포트홀(False Positive)로 오인하는 현상이 급증했습니다.

### 4.3. YOLOv11 학습 및 블랙박스 추론 (Inference)
전처리 파이프라인을 통과한 프레임을 YOLOv11 모델에 태워 실시간으로 바운딩 박스를 렌더링합니다.

```python
from ultralytics import YOLO

# 1. 커스텀 학습이 완료된 최적 가중치 로드
model = YOLO('runs/detect/train/weights/best.pt')

# 2. 국내 주행 블랙박스 영상 테스트 진행
results = model.predict(
    source='data/test_blackbox_driving.mp4', 
    conf=0.25,        # Confidence Score (0.25 이상인 객체만 탐지)
    iou=0.45,         # NMS 겹침 허용치
    imgsz=640,        # 640x640 리사이즈
    save=True         # 결과 동영상 렌더링 및 저장
)
```
---

## 5. 실험 결과 및 심층 분석 (Results & Analysis)
전처리 기법들의 실제 효용성을 검증하기 위해 Val 데이터를 기준으로 mAP(Mean Average Precision)를 측정하여 비교했습니다.

<img width="744" height="948" alt="image" src="https://github.com/user-attachments/assets/f4df486f-2f62-476e-a053-5b449c7841a7" />



---

## 6. 결론 및 향후 과제 (Conclusion & Future Work)

최종 결론
본 프로젝트를 통해 해외 고화질 데이터셋으로 학습된 객체 탐지 모델은 국내 블랙박스 실환경에 곧바로 적용하기 어렵다는 점을 확인했습니다

무엇보다 성능을 강제로 끌어올리기 위해 도입했던 전통적인 컴퓨터 비전 전처리(ROI Crop, CLAHE)가 최신 딥러닝 모델에서는 오히려 독이 되어 딥러닝 특유의 문맥(Context) 파악 능력을 방해하고 노이즈를 극대화한다는 사실을 실험적으로 증명해냈습니다.

----
## 향후과제

1. 데이터 중심 AI (Data-centric Approach) 전환
   
 -알고리즘의 파라미터 튜닝이나 인위적 전처리에 집착하는 것을 멈추고, 본질적인 해결을 위해 국내 도로 환경의 빛 반사, 렌즈 블러가 그대로 담긴 블랙박스 규격의 자체 데이터셋을 새롭게 수집하여 파인튜닝할 계획입니다.

2. Yaml 불일치 해결 및 파이프라인 자동화

-서로 출처가 다른 데이터셋을 병합할 때 발생하는 클래스 꼬임 문제를 원천 차단하기 위해, 레이블 구조를 자동으로 통합하고 정제해주는 전처리 자동화 스크립트를 개발할 것입니다.
 
3. 엣지 디바이스 환경 최적화

-추후 차량용 블랙박스나 저사양 내비게이션 등 임베디드 장비 내에서 지연 시간(Latency) 없이 실시간으로 작동할 수 있도록, 지식 증류(Knowledge Distillation)나 양자화(Quantization)를 통한 모델 경량화 연구를 이어갈 것입니다.








