'''This script detects if a person is drowsy or not,using dlib and eye aspect ratio
calculations. Uses webcam video feed as input.'''

#Import necessary libraries
from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import pygame #For playing sound
import time
import dlib
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

#Initialize Pygame and load music(wav 파일 지원)
pygame.mixer.init()     #믹서 모듈의 초기화 함수
pygame.mixer.music.load('audio/alert.wav')  #음악 로딩
guideSound = pygame.mixer.Sound("audio/start_guide_wav.wav")   # 안내 음성
video_file = "./video/eye_test.mp4"

#Minimum threshold of eye aspect ratio below which alarm is triggerd
#눈의 EAR의 THRESH 기본값을 0.3으로 설정
#졸음운전을 판단할 때 사용하는 임곗값(눈)
EYE_ASPECT_RATIO_THRESHOLD = 0.3
#하품을 판단할 때 사용하는 임곗값(입)
MOUTH_THRESHOLD = 0.37
#고개숙임 판단 시 사용하는 임계값
HEAD_DOWN_THRESHOLD = 0.3

#Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
EYE_ASPECT_RATIO_CONSEC_FRAMES = 10


#COunts no. of consecutuve frames below threshold value
#프레임 카운터(눈) 초기화
EYE_COUNTER = 0

#프레임 카운터(입) 초기화
COUNTER_MOUTH = 0

#Load face cascade which will be used to draw a rectangle around detected faces.
#얼굴 인식: 정면 얼굴에 대한 Haar_Cascade 학습 데이터 (직사각형 그리는 용도)
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

#눈 종횡비(EAR) 계산 함수
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])  #두 점사이의 거리를 구하는데 사용하는 함수
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A+B) / (2.0 * C)
    return ear

#입 거리비 계산 함수
def mouth_rate(lip):
    leftPoint = lip[0]  #48
    rightPoint = lip[6]  #54
    topPoint = lip[3]  #51
    rowPoint = lip[9]  #57

    mouthWidth= distance.euclidean(leftPoint, rightPoint)   #입 너비
    mouthHeight = distance.euclidean(topPoint, rowPoint)  # 입 높이

    mouthRate = mouthHeight / mouthWidth

    return mouthRate

#고개 숙임 계산 함수
def head_rate(head):
    headX = distance.euclidean(head[27], head[30])   #코 시작~끝(27, 30)
    headY = distance.euclidean(head[30], head[8])   #코 끝~턱 끝(30, 8)

    headRate = headX / headY

    return headRate

#Load face detector and predictor, uses dlib shape predictor file
#68개의 얼굴 랜드마크 추출
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#Extract indexes of facial landmarks for the left and right eye
#위에서 추출한 랜드마크에서 왼쪽 눈과 오른쪽 눈의 랜드마크 좌표 추출
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
#랜드마크에서 입 좌표 추출
(mStart, mEnd) = 48, 68

#Start webcam video capture
#첫번째(0) 카메라를 VideoCapture 타입의 객체로 얻어옴
video_capture = cv2.VideoCapture(0)
#video_capture = cv2.VideoCapture(video_file)

#Give some time for camera to initialize(not required)
time.sleep(2)


# 현재시각
currentTime = time.strftime("%Y%m%d_%H%M%S")
outputFileName = "./output/" + currentTime + ".avi"

# 웹캠의 속성 값을 받아오기
# 정수 형태로 변환하기 위해 round
w = round(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_capture.get(cv2.CAP_PROP_FPS) # 카메라에 따라 값이 정상적, 비정상적

print(fps) #30.0

# fourcc 값 받아오기, *는 문자를 풀어쓰는 방식, *'DIVX' == 'D', 'I', 'V', 'X'
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

# 1프레임과 다음 프레임 사이의 간격 설정
delay = round(1000/fps)

# 웹캠으로 찰영한 영상을 저장하기
# cv2.VideoWriter 객체 생성, 기존에 받아온 속성값 입력
out = cv2.VideoWriter(outputFileName, fourcc, fps/10, (w, h))

X = 0


START = False

#실시간 그래프 그리기 위함
plt.show()  #1번만 호출해야함
plt.xticks(rotation=45) #x축 라벨 -45

dataX=[]
dataY=[]

while(True):
    #Read each frame and flip it, and convert to grayscale
    # ret : frame capture결과(boolean)
    # frame : Capture한 frame
    ret, frame = video_capture.read()   #비디오 읽기
    frame = cv2.flip(frame,1)           #프레임 좌우반전(flipcode > 0)
    #BGR 이미지: / GrayScale 이미지: 색상정보X,밝기 정보로만 구성(0~255, 검~흰)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #color space 변환(BGR->Grayscale로 변환)

    #Detect facial points through detector function
    faces = detector(gray, 0)

    #Detect faces through haarcascade_frontalface_default.xml
    # grayscale 이미지를 입력해 얼굴 검출하여 위치를 리스트로 리턴(x, y, w, h)
    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)    #(ScaleFactor, minNeighbor)

    #키보드 입력으로 시작하기
    if(cv2.waitKey(1) & 0xFF == ord('s')):
        START = True

    if(START):
        # 블랙박스처럼 현재 시각 나타내기
        currentTime = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, currentTime, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 얼굴 사각형 그리기
        # (x, y): 좌상단 위치, (w, h): 이미지 크기
        for (x, y, w, h) in face_rectangle:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Detect facial points
        # 검출된 얼굴 들에 대한 반복
        for face in faces:

            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            # 왼쪽,오른쪽 눈 index 리스트로 얻기
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            # 입 index 리스트로 얻기
            innerMouth = shape[mStart:mEnd]

            # 양쪽 눈의 EAR 계산
            leftEyeAspectRatio = eye_aspect_ratio(leftEye)
            rightEyeAspectRatio = eye_aspect_ratio(rightEye)

            # 입 거리비 계산
            mouthRate = mouth_rate(innerMouth)

            # 고개 숙임 Rate 계산
            headRate = head_rate(shape)

            # 양쪽 눈 EAR 값의 평균 계산
            eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

            # Use hull to remove convex contour discrepencies and draw eye shape around eyes
            # 눈 윤곽선에서 블록껍질 검출(경계면 둘러싸는 다각형 구하기)
            # convexHull(윤곽선:윤곽선 검출 함수에서 반환되는 구조, 방향:True/False[시계/반시계])
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            # 입 윤곽선에서 블록껍질 검출
            innerMouthHull = cv2.convexHull(innerMouth)

            # 눈 경계선 그리기
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            # 입 경계선 그리기
            cv2.drawContours(frame, [innerMouthHull], -1, (0, 255, 0), 1)

            # 얼굴 랜드마크 68 point 찍기
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            # 좌측 상단 기준 text 표시
            # 눈 EAR 표시
            cv2.putText(frame, "EAR : {:.2f}".format(eyeAspectRatio), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 30, 20), 2)

            # 입 거리비 표시
            cv2.putText(frame, "mouthRate : {:.2f}".format(mouthRate), (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 30, 20), 2)

            # 고개 숙임비 표시
            cv2.putText(frame, "headRate : {:.2f}".format(headRate), (0, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 30, 20), 2)

            # Detect if eye aspect ratio is less than threshold
            # 졸음판단(현재 EAR이 임계값보다 작은지 확인)
            if (eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD):
                # 임계값보다 EAR 작으면(눈 감고 있는 상태)
                EYE_COUNTER += 1
                # If no. of frames is greater than threshold frames,
                if EYE_COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                    # 눈을 깜빡인건지 졸고있는건지 확인
                    pygame.mixer.music.play(-1)
                    cv2.putText(frame, "You are Drowsy", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            else:
                # 임계값보다 EAR 크면(눈 뜨고 있는 상태)
                pygame.mixer.music.stop()  # 소리 출력 정지
                EYE_COUNTER = 0  # 카운터 초기화

            # 하품판단
            if (mouthRate > MOUTH_THRESHOLD * 2):
                COUNTER_MOUTH += 1
                # 10 프레임 연속 조건 충족 시, 하품 판단
                if (COUNTER_MOUTH >= 10):
                    cv2.putText(frame, "You are Yawn", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            else:
                COUNTER_MOUTH = 0

            # 고개 숙임 판단
            if (headRate > HEAD_DOWN_THRESHOLD * 1.8):
                cv2.putText(frame, "You are HEAD DOWN", (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

        # 실시간 그래프그리기
        X += 1
        dataX.append(X)
        dataY.append(eyeAspectRatio)
        # plt.scatter(X, eyeAspectRatio)
        plt.plot(dataX, dataY)
        plt.pause(0.000000001)

    # 비디오 저장
    out.write(frame)  # 영상 데이터만 저장. 소리는 X

    #Show video feed
    cv2.imshow('Video', frame)

    #키보드 입력으로 중지시키기
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

#Finally when video capture is over, release the video capture and destroyAllWindows
video_capture.release()
cv2.destroyAllWindows()