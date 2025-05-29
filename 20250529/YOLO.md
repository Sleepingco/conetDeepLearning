1. CNN (Convolutional Neural Network)
– 기본 기반 기술
• 등장 배경:이미지 분류 문제를 해결하기 위한 기본 구조
• 고정된 위치에 있는 객체를 분류만 할 수 있음
– 기술적 특이점:
정확한 위치(localization) 불가능
전체 이미지를 보고 "고양이다"는 말은 가능하지만, "어디에 고양이가 있다"는 건 못함
기본 이미지 분류에만 사용됨
– 객체 탐지의 시작점, 하지만 "어디 있는지"는 모름

2. R-CNN (Regions with CNN features, 2014)
– 등장 배경:
• "분류"만 하던 CNN에 위치 정보(객체 박스)를 붙이고 싶다
• Selective Search를 이용해 후보영역(region proposals)을 만든 뒤, 각 영역마다 CNN을 적용
– 기술적 특이점:
• 두 단계 방식 (Two-stage Detection)
• 후보 박스 생성 (Selective Search)
• CNN으로 각 후보 영역 분류
• 정확도는 높지만 속도가 엄청 느림 (한 이미지당 수천 번 CNN 실행)
– CNN으로 객체 위치까지 찾기 시작했지만 너무 느렸음

R-CNN은 object가 있을 법한 후보 영역을 뽑아내는 "Region proposal"
알고리즘과 후보 영역을 분류하는 CNN을 사용,
예시)약 2,000개에 달하는 후보 이미지 각각에 대해서 convolution 연산을 수행
Proposal을 만들어내는 데에는 Selective search라는 비 신경망 알고리즘이
사용됩니다. 이후에 후보 영역의 Classification과 Bounding Box의 regression을
위해 신경망을 사용

3. Fast R-CNN (2015)
– 등장 배경:
• R-CNN의 느린 속도와 중복된 CNN 연산 문제 해결 목적
– 기술적 특이점:
• 전체 이미지에 CNN을 한 번만 적용
• 그 후, ROI(Region of Interest) Pooling을 통해 후보영역 추출
• 단일 CNN feature map을 재활용 → 속도 향상
• 학습도 end-to-end로 가능
–이미지 한 번만 보고, 후보만 쪼개서 분류하자
R-CNN의 경우, region proposal을 selective search로 수행한 뒤 많은 수의 후보 이미지
각각에 대해서 convolution 연산을 수행. 이 경우 한 이미지에서 feature을 반복해서
추출하기 때문에 비효율적이고 느리다는 단점
Fast R-CNN에서는 입력 이미지를 한 번만 CNN에 통과시키고, 객체 탐지를 위한
후보 영역 추출과 특징 추출을 분리하여 수행. 이로써 앵커 박스와 특징 맵의 일부
영역을 공유하므로, RCNN보다 효율적인 속도와 좋은 탐지 성능을 제공

One stage Detector : 객체의 검출과 분류, 그리고 바운딩 박스 regression을 한 번에
Two stage Detector : object 위치 후보(Region proposals) 들을 뽑아내는 단계,
 객체 검출, 이후 object가 있는지를 Classification과
 정확한 바운딩 박스를 구하는 Regression을 수행하는 단계.
 4. Faster R-CNN (2015)
– 등장 배경:
• Fast R-CNN은 여전히 Selective Search에 의존 (느림)
• 그래서 후보영역 생성조차도 신경망으로 대체하자
– 기술적 특이점:
• Region Proposal Network (RPN) 도입
• RPN이 직접 anchor 기반으로 후보 영역을 예측
• 완전한 end-to-end 학습 가능
• 정확도와 속도 모두 뛰어남
– “후보 영역도 CNN으로 만들자” → 진짜 완성된 객체 탐지 모델
Region Proposal Network (RPN) 이란?
CNN feature map 위에 슬라이딩 윈도우를 돌리면서
Anchor Box라는 다양한 크기/비율의 박스를 겹쳐놓고
각각의 박스가 물체가 있는지 없는지, 좌표는 어디인지 예측
즉, 후보영역 뽑기마저 CNN으로 직접 처리!
5. YOLO (You Only Look Once, 2016)
– 등장 배경:
• R-CNN 계열은 정확하지만 실시간에 부적합 (2-stage 구조)
• 객체 탐지를 한 번에 끝내고 싶다
– 기술적 특이점:
• 이미지를 grid로 분할 후, 각 셀에서 객체 존재 여부 + 위치 + 클래스 예측
• 단일 CNN만으로 전체 처리
• 실시간 처리 가능속도는 훌륭하지만, 초기 버전(YOLOv1)은 작은 객체 탐지에 약했음
–딱 한 번 본다
– 빠르고 직관적인 객체 탐지기
6. DETR (DEtection TRansformer)
2020년 Facebook AI 발표, "Transformer를 이용해 객체 탐지를 End-to-End로 처리하는 모델“
등장 배경
기존 객체 탐지 알고리즘들은 거의 다 CNN + 복잡한 후처리(Anchor, NMS) 구조였음
anchor 설정, region proposal, NMS 등의 "핸드메이드 규칙"이 너무 많았음
“이걸 다 없애고, 순수하게 딥러닝(Transformer)으로만 처리할 수 없을까?”
기술적 특이점
1. Object Detection을 "Sequence Prediction"으로 봄
기존 탐지: “여기 뭔가 있어!” 박스를 예측
DETR: “이 이미지는 N개의 객체가 있고, 각각의 class와 위치는 이거야” 라고 문장처럼 예측
2. Transformer 구조 사용 (Encoder-Decoder)
Encoder: 이미지 → CNN으로 feature map 추출 후 positional encoding 추가
Decoder: 객체를 나타내는 learnable queries (예: 100개) 를 넣어 Transformer가 어떤
객체들이 있는지 예측
3. No Anchors, No NMS, No Region Proposal, NMS 전부 제거됨!
대신 예측한 box들과 실제 box를 Hungarian Matching 알고리즘으로 매칭해서 학습
이 덕분에 완전히 end-to-end 학습 가능함
7. Vision Transformer (ViT) : 이미지 분류(Image Classification)
자연어 처리(NLP)에서 성공을 거둔 Transformer 구조를 컴퓨터 비전 분야에 적용하여,
CNN 없이도 이미지 분류에서 높은 성능을 달성하고자 함
구조 및 특징:
입력 처리: 이미지를 고정 크기의 패치로 분할하고, 각 패치를 시퀀스로 변환
포지셔널 인코딩: 패치의 순서 정보를 보존하기 위해 위치 정보를 추가
Transformer 인코더: Self-Attention을 통해 패치 간의 관계를 학습
분류 헤드: 특별한 [CLS] 토큰을 통해 전체 이미지의 표현을 얻고, 이를 분류기로 사용
장점:
전역적인 정보 처리 능력: Self-Attention을 통해 이미지 전체의 문맥을 고려한 특징 추출 가능
확장성: 대규모 데이터셋에서의 학습을 통해 높은 성능 달성
단점:
데이터 효율성 부족: CNN에 비해 적은 데이터로는 성능이 저하될 수 있음
로컬 정보 처리의 한계: CNN이 갖는 지역적인 특성을 활용하지 못함

비전기반의 자율주행의 한계
라이다/레이더 센싱(물리적감) vs 비전센셍(신경망을 통한감지)
점유네트워크로 개선한 비젼 기반 자율 주행 문제점
-조감도(버드아이뷰):2d조감도를 3d조감도로 업데이트
-고정 직사각형(픽스드 렉탱글)복셀(볼륨+픽셀)로 나눈 이미지로 디테일 확장 불규칙한 모양과 돌출된 부분도 감지
-겍체 감지(오브젝트 디텍션):학습된 객체만 인식하는 한계를 극복 객체의 인식보다 물체의 존재여부에 집중
움직이는 객체와 않움직이는 객체를 구분하여 메모리 효율성 향상

Occupancy Network : 점유 그리드 매핑, 탐지보다는 점유여부 체크
차량 주변 환경을 3D 격자 형태로 표현하고, 각 격자에 물체가 존재하는지 여부를 예측하는
딥러닝 모델, 실시간으로 차량 주변의 장애물과 공간을 인식, 멀티뷰 사용가능
Occupancy Network의 작동 방식
1.데이터(라이다, 레이더, 카메라 등)를 입력
2.데이터를 기반으로 차량 주변 환경을 3D 격자로 나눕니다.
3.각 격자에 대해 물체의 존재 여부를 0(빈 공간) 또는 1(물체 존재)로 예측
4.예측된 3D 격자 데이터를 활용하여 주행 경로를 계획하고 장애물을 회피
예를 들어:차량 전방에 다른 차량과 보행자가 있고, 우측에는 가로수
이 장면을 3D 격자로 표현하고, 각 격자에서 물체 존재 여부를 예측
1(물체 존재)로 예측되고, 그 외 격자는 0(빈 공간)으로 예측
자율주행 시스템은 전방 차량과 보행자, 가로수를 인식하고 이를 회피 경로를 계획
실시간으로 복잡한 환경을 정확히 인식,
자율주행/로봇 내비게이션/증강현실 등 다양한 분야에서 활용

오큐펀시 플로우 2d이미지를 3d로 변환하여 움직이는 물체와 움직이지 않는 물체의 구분(vs 다른차량등을 통해 작성된 3d지도:nerfs)
실시간 비교 분석> 주행 차량과의 주행관계를 인식
오큐펀시 볼륨: 물체가 차지하는 부피를 산출
날씨가 흐리거나 어두운밤들의 영향으로 입력이미지가 좋지 않은 경우: 차량의 여러 이미지, 여러차량의 동일 이미지를 고려하여 평균값을 산출/ 예측하여 활용

재구성한 이미지 지도vs점유네트워크로 생성된 실제 상황(현실 이미지)
차이를 비교/분석하여 fds(autopilot)에 활용

점유 흐름은 부피를 차지하는 물체가 움직이는 여부와 방향, 속도를 알려줍니다. 앞차가 정지해 있는지, 갑자기 멈췄는지 등을 감지하여 반응할 수 있습니다.
점유네트워크 출력(실제 상황)과 재구성 장면을 비교하여 지속적으로 학습 시킵니다. 차량내에서 실시간으로 작동하며 끊임없이 주변 환경을 확인하고 검증합니다. 
– Occupancy Network
“공간 안에 어떤 점이 물체 내부냐 외부냐”를 신경망이 판단하게 하는 방식
핵심 개념
• 입력: 어떤 3D 공간의 좌표 (𝑥,𝑦,𝑧)(x,y,z)
• 출력: 해당 좌표가 물체 안인지 밖인지 (occupancy: 0 또는 1)
• 이걸 하나의 MLP(다층 퍼셉트론)가 학습해서 공간 전체를 묘사함
• 학습 후에는 임의의 해상도로 표면을 추출(Marching Cubes 등) 가능
Occupancy Networks란?
Occupancy Networks는 전통적인 객체 감지 방식을 넘어서는 새로운 접근 방식
•3D 공간 분할: 공간을 작은 3D 격자(보통 voxel)로 나누고, 각 voxel이 점유되었는지를 예측.
•형상 기반 인식: 객체의 존재 여부를 판단하는 데 초점며, 객체의 종류나 클래스에 의존 않음.
•동적 객체 인식: 정적 및 동적 객체를 구분 가능.
•고속 처리: 100 FPS 이상의 속도로 실행되어 실시간 처리에 적합.
• 입력: 8개의 카메라 영상
• 특징 추출: RegNet 및 BiFPN을 사용하여 이미지에서 특징을 추출
• Attention 모듈: 위치 정보를 포함한 특징을 사용, Occupancy Feature Volume을 생성
• 시간적 융합: 이전 프레임들과의 정보를 융합하여 4D Occupancy Grid를 생성
• 출력: Occupancy Volume과 Occupancy Flow를 생성

RegNet (Regularized Network)
• 등장 배경
ResNet, DenseNet 등은 성능은 좋지만 구조가 사람 손으로 설계된 비정형적 구조
Facebook AI는 실험적으로 수천 개의 네트워크를 생성해봤더니 성능 좋은 네트워크들은
일정한 규칙에 따라 폭과 깊이가 증가하는 구조를 가짐
수학적으로 간단한 함수로 커지는 네트워크(RegNet) 설계
• 구조적 특징
Block의 채널 수 (width), 깊이 (depth) 를 함수로 정의
하드웨어 최적화 :
 연산량(Flops) 대비 정확도 우수, 범용성 백본(backbone)으로 쓰기에 안정적
예시: RegNetY-16GF: COCO나 ImageNet에서 우수한 성능을 내는 대표 모델
 Tesla도 Occupancy Network에서 이 구조를 backbone으로 사용

 BiFPN (Bidirectional Feature Pyramid Network)
• 등장 배경
기존 FPN (Feature Pyramid Network)은 top-down (큰→작은) 방향만 정보 전달하지만,
객체는 큰 것과 작은 것이 같이 있기 때문에 작은 해상도 정보도 위로 전달하는
구조가 필요했음
• BiFPN의 특징
양방향 흐름: Top-down + Bottom-up
Skip 연결: 중복된 feature들끼리 병합
• 가중치 학습:
여러 scale에서 들어오는 feature를 단순히 더하지 않고,
각각의 importance를 학습하여 더함 (Weighted sum)

NeRF (Neural Radiance Fields)
3D 공간의 한 점에서의 색(RGB)과 밀도(α)를 예측해, 2D 이미지로 렌더링 가능하게
• 등장 배경
3D 재구성에서 photorealistic한 이미지 렌더링이 어려움
"이미지를 쏴서 다시 이미지를 만들자!"라는 접근으로 시작됨 (볼륨 렌더링 기반)
• 핵심 개념
입력: 3D 좌표 (x,y,z) + viewing direction (𝜃,𝜑)
출력: 그 점에서 보는 RGB 색상과 밀도 값
 이를 통해 volume rendering 알고리즘을 이용해 이미지를 재구성

 Bounding Box
Bounding Box는 바운딩 박스, Bbox
corner coordinates : bounding box를 좌상점과 우하점으로 나타낸다 ,
center coordinates : bounding box를 중심점과 width, height 로 나타낸다
•코너 좌표 방식: (10, 20), (50, 70)
•중심 좌표 방식: (30, 45), 너비 40, 높이 50

COCO dataset(common objects in context)
약 80개의 객체 카테고리와 약 330,000개의 이미지로 구성.
바운딩 박스 정보와 객체 분할을 위한 픽셀-레벨 마스크 정보가 포함.
객체 탐지 및 분할 알고리즘의 성능 평가를 위해 많이 사용.(YOLO의 학습데이터!!)
평균 정밀도(AP), 평균 재현율(AR) 등의 지표를 통해 알고리즘의 정확도와 성능을 평가

mAP(means of Average Precision)
Data set내의 전체 클래스에 대해 각 클래스에 대한 AP(Average Precision)을
계산하고 그 숫자의 평균(means)
단일 클래스의 모델성능정보 대비 전체 Data set의 모델 성능정보 기준 !!

IoU(Intersection over Union)
정답영역(Ground Truth bounding box)과 예측영역(predicted bounding box),
박스의 차이를 상대적으로 평가하기 위한 방법 , 교차하는 영역을 합친 영역으로 나눈 값
정답 영역과 예측 영역이 겹쳐진 부분이 클 수록 IoU의 값이 커진다

단순히 이미지를 분류하는 것이 classification,
car라는 object의 위치를 알아내는 것이 Classification with localization,
localization을 통해 여러 object를 인식하는 것을 Detection

Object Detection이란
물체를 분류하는 Classification,
물체가 박스를 통해 (Bounding box) 위치 정보 찾는Localization
Object Detection = Classification + Localization
Object Detection = Multi-Labeled Classification+ Bounding Box Regression(Localization)

Image Segmentation :이미지를 픽셀 단위의 다양한 segments로 분할 task.
 이미지의 모든 픽셀에 라벨을 할당하는 task.
Segmentation :
- 동일한 클래스에 해당하는 픽셀을 같은 색으로 칠하는 Semantic Segmentation.
- 동일한 클래스/다른 사물의 픽셀이면 다른 색으로 칠하는 Instance Segmentation
슬라이딩 윈도우(Sliding window)
Localization network의 입력으로 만들기 위해 원본 이미지에서 잘라내는 크기를
window 크기로 영역을 이동시키며(sliding) 이미지 탐색을 수행하는 방식
이미지 내의 오브젝트가 어떤 크기로 어떤 종횡비를 갖는지 알 수 없기 때문에 매우
많은 종류의 Windows를 생성해야 하기 때문에 매우 비효율적인 방법일 수 있다.

앵커 박스(Anchor Box)는 사전에 정의된 크기와 종횡비를 가지는 사각형. 일반적으로
이미지 내에서 객체가 나타날 것으로 예상되는 여러 위치에 앵커 박스들을 미리 배치.
바운딩 박스는 실제로 탐지된 객체의 경계를 나타내는 사각형. 객체의 경계선을 둘러싸는
최소한의 크기로 설정되며, 바운딩 박스의 좌표와 크기 정보를 통해 객체의 위치와 형태를
추론할 수 있다.
객체 탐지는 앵커 박스를 기반으로 예측한 객체의 위치 및 크기를 바운딩 박스로 표현

Anchor box 정리
•y의 label을 보면 Anchor box가 2개가 됨에 따라서 output dimension이 두 배가 되었다.
그리고 각각은 정해진 Anchor box에 매칭된 object를 책임지게 됩니다.
• grid cell에서 Anchor box에 대한 object 할당은 IoU로 할 수 있다. 인식 범위 내에
object가 있고 두 개의 Anchor box 경우 IoU가 더 높은 Anchor box에 object를 할당.

NMS(Non-Max Suppression), 비-최대 억제
object detection 알고리즘은 탐지된 객체에 여러 개의 bounding boxes를 생성.
이 중 하나의 bounding box만을 선택해야 하는데 이때 적용하는 기법
NMS는 겹친 박스들이 있을 경우 가장 확률이 높은 박스를 기준으로 기준이 되는
IoU 이상인 것들을 없앤다.

ROI(Region of Interest) pooling
주로 Fast R-CNN에서 적용. 추출된 특징 맵에서 앵커 박스에 해당하는 부분을
잘라내고 고정된 크기의 특징 맵으로 변환하는 과정을 수행.
예를 들어, 자동차 탐지를 수행하는 Fast R-CNN에서 추출된 특징 맵과 앵커 박스가 주어졌을 때, ROI
pooling은 앵커 박스를 일정한 크기로 분할하여 각 분할 영역에서 최댓값을 선택하고, 선택된 값들을
모아 고정된 크기의 특징 벡터로 변환

YOLO는
이미지를 grid로 나누고, Sliding window 기법을 Convolution 연산으로 대체해
Fully Convolutional Network 연산을 통해 grid cell별로 Bbox를 얻어낸 뒤,
Bbox들에 대해 NMS를 한 방식

Bounding box 조정과 Classification을 동일 신경망 구조를 통해
동시에 실행하는 통합 인식(Unified Detection)을 구현
S*S개의 Grid 구현 > Grid내 B개의 Bounding Box 예측 > Bbox내의 물체가 존재할
확률 계산(Confidence) > Grid마다 특정물체의 클래스 확률 계산

mAP (mean Average Precision) @0.5
모델의 **정확도(Precision)**와 **재현율(Recall)**의 조합을 종합적으로 측정하는 지표
"모델이 객체를 얼마나 잘 탐지하고, 정확히 위치시켰는가?".
"IoU ≥ 0.5일 때" 정확도(Precision)의 평균 값을 의미
예측 박스와 정답 박스가 50% 이상 겹치면 성공(True Positive), 모든 클래스의 평균 정확도.

1. 학습 출력 관련 주요 용어
바운딩 박스 :객체를 감싸는 사각형 영역
바운딩 박스는 (x, y) 좌표와 (width, height)로 정의.
컨피던스스코어:모델이 특정 객체를 탐지했을 때 해당 객체가 있을 확률
클래스 확률바운딩 박스 안에서 탐지된 객체가 특정 클래스에 속할 확률
IOU예측한 바운딩 박스와 실제 정답 바운딩 박스의 겹치는
부분의 비율. IoU는 (겹치는 영역의 면적) / (두 박스의 합집합 면적)으로 계산.
IoU Threshold: IoU가 일정 임계값 이상인 경우, 객체 탐지가 정확하다고 간주
예를 들어, IoU가 0.5 이상이면 예측된 객체가 실제 객체와 일치한다고 판단
AP:모델의 정확도를 평가하는 지표로, Precision-Recall Curve(정밀도재현율 곡선)를 기반으로 계산됩니다. AP는 여러 IoU 임계값에 대해 계산된 평균값.
mAP:AP를 여러 클래스에 대해 평균화한 값

2. 모델 평가 관련 주요 용어
• Precision (정밀도): 모델이 예측한 객체 중에서 실제로 객체가 맞는 비율.
Precision = TP / (TP + FP)TP(True Positive): 실제 객체를 정확히 탐지한 경우
• FP(False Positive): 객체가 없는데 객체가 있다고 잘못 예측한 경우
• Recall (재현율): 실제 객체 중에서 모델이 정확히 탐지한 비율.
Recall = TP / (TP + FN)
• FN(False Negative): 실제 객체가 있는데 모델이 이를 탐지하지 못한 경우
• F1-Score: Precision과 Recall의 조화 평균으로, 두 지표의 균형을 평가
두 값이 매우 다르면 F1-Score가 낮습니다.
F1 = 2 * (Precision * Recall) / (Precision + Recall)

inference speed
모델이 실제 환경에서 얼마나 빠르게 동작하는지를 평가하는 지표
mAP@ 0.5
일반적으로 객체 탐지 모델의 성능을 평가하는 주요 지표로,
IoU 임계값을 0.5로 설정하고 평균 정밀도를 계산.

FP16 : 부동소수점 16비트 연산(Floating Point 16-bit)
6.3ms, V100 :YOLO 모델이 약 6.3밀리초의 추론 속도,
 NVIDIA V100 GPU 하드웨어를 사용 기준
28.4mAP : 모델이 COCO 데이터셋에 대해 28.4%의 평균 정밀도

Small Models:YOLOv5s with smaller configuration
속도가 빠르고 경량화된 모델로, 작은 객체를 탐지하는 데 유용.
작은 모델은 리소스 제약이 있는 환경이나 실시간 처리가 필요한 경우에 유용.
Medium Models: YOLOv5m
중간 크기의 입력 이미지에 대한 객체 탐지 작업에 적합하며,
정확도와 속도 사이의 균형을 유지.
Large Models:YOLOv5l, YOLOv5x
입력 이미지의 크기에 상관없이 다양한 크기의 객체를 정확하게 탐지할 수 있으며,
더 많은 컴퓨팅 리소스가 필요.
정확도를 우선시하는 복잡한 객체 탐지 작업에 적합

BackBone(특징 추출부)
• BottleNeckCSP
"Bottleneck": CNN 블록 구조로, 연산량을 줄이면서도 특징을 잘 추출하게 해줌
"CSP" (Cross Stage Partial): 특징 정보 일부만 전달, 계산량은 줄이고 성능은 유지
•SPP(Spatial Pyramid Pooling)
여러 크기의 커널로 풀링하여 다양한 스케일의 정보를 확보
PANet(Path Aggregation Network, 특징 통합부)
Output(출력부)