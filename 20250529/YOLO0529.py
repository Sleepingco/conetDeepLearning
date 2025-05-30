# %%
!pip install ultralytics

# %%
from ultralytics import YOLO

# 모델 로딩
model = YOLO('yolov8n.pt')

# 이미지 경로 지정
image_path = '/content/man_car_cat.jpg'

# 예측 수행 (오타 수정!)
result_image = model.predict(source=image_path, save=True, show=True)

# 결과 출력
print("Image predict result:", result_image)


# %%
print(type(model.names),len(model.names))
print(model.names)

# %%
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    print("웹캠 작동 OK!")
else:
    print("웹캠 연결 문제!")
cap.release()

# %%
!curl -L "https://public.roboflow.com/ds/c7SehbkfK4?key=5BCTbB5LTm" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

# %%
!cat data.yaml

# %%
model.train(data='/content/data.yaml', epochs=10, batch=32, imgsz=640, lr0=0.001, augment=True, patience=10, save=True)

# %%
print(type(model.names),len(model.names))
print(model.names)

# %%
from ultralytics import YOLO
# 1️⃣best.pt 모델 로드
model = YOLO('runs/detect/train/weights/best.pt') # 경로를 본인의 best.pt로 수정!
# 2️⃣예측할 이미지 경로
image_path = '/content/jellyfish.jpg' # 예측할 이미지 경로
# 3️⃣예측 수행
results = model.predict(
 source=image_path, # 이미지 파일 경로 (또는 폴더 경로, 비디오 경로, 0: 웹캠)
 save=True, # 예측 결과 이미지 저장
show=True, # 팝업 창에 결과 출력
conf=0.5, # 탐지 신뢰도 threshold
 device=0 # GPU: 0 / CPU: 'cpu'
)
# 4️⃣결과 확인
for result in results:
 print(result.boxes) # 감지된 객체의 경계 상자 정보

# %%
!curl -L "https://universe.roboflow.com/ds/dzCaW2YwM0?key=x1owEIg2f7" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

# %%
!cat data.yaml

# %%
model.train(data='/content/data.yaml', epochs=50, batch=32, imgsz=640, lr0=0.001, augment=True, patience=10, save=True)

# %%
from ultralytics import YOLO
# 1️⃣best.pt 모델 로드
model = YOLO('runs/detect/train2/weights/best.pt') # 경로를 본인의 best.pt로 수정!
# 2️⃣예측할 이미지 경로
image_path = '/content/download.jpg' # 예측할 이미지 경로
image_path = '/content/istockphoto-647672134-612x612.jpg' # 예측할 이미지 경로
# 3️⃣예측 수행
results = model.predict(
 source=image_path, # 이미지 파일 경로 (또는 폴더 경로, 비디오 경로, 0: 웹캠)
 save=True, # 예측 결과 이미지 저장
show=True, # 팝업 창에 결과 출력
conf=0.5, # 탐지 신뢰도 threshold
 device=0 # GPU: 0 / CPU: 'cpu'
)
# 4️⃣결과 확인
for result in results:
 print(result.boxes) # 감지된 객체의 경계 상자 정보

# %%
model = YOLO('yolov8s.pt')

# %%
!cat data.yaml

# %%
model.train(
    data='/content/data.yaml',
    epochs=50,
    batch=32,
    imgsz=720,
    lr0=0.001,
    augment=True,
    auto_augment='randaugment',
    mosaic=1.0,
    copy_paste=0.5,
    cutmix=0.3,
    erasing=0.4,
    patience=10,
    save=True,
    optimizer='AdamW'
)


# %%
from ultralytics import YOLO
# 1️⃣best.pt 모델 로드
model = YOLO('runs/detect/train42/weights/best.pt') # 경로를 본인의 best.pt로 수정!
# 2️⃣예측할 이미지 경로
image_path = '/content/download.jpg' # 예측할 이미지 경로
image_path = '/content/istockphoto-647672134-612x612.jpg' # 예측할 이미지 경로
# 3️⃣예측 수행
results = model.predict(
 source=image_path, # 이미지 파일 경로 (또는 폴더 경로, 비디오 경로, 0: 웹캠)
 save=True, # 예측 결과 이미지 저장
show=True, # 팝업 창에 결과 출력
conf=0.25, # 탐지 신뢰도 threshold
 device=0 # GPU: 0 / CPU: 'cpu'
)
# 4️⃣결과 확인
for result in results:
 print(result.boxes) # 감지된 객체의 경계 상자 정보


