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

#Initialize Pygame and load music
pygame.mixer.init()     #믹서 모듈의 초기화 함수
pygame.mixer.music.load('audio/alert.wav')  #음악 로딩

#Minimum threshold of eye aspect ratio below which alarm is triggerd
#눈의 EAR의 THRESH 기본값을 0.3으로 설정
#졸음운전을 판단할 때 사용하는 임곗값
EYE_ASPECT_RATIO_THRESHOLD = 0.3

#Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
EYE_ASPECT_RATIO_CONSEC_FRAMES = 20

#COunts no. of consecutuve frames below threshold value
#프레임 카운터 초기화
COUNTER = 0

#Load face cascade which will be used to draw a rectangle around detected faces.
#얼굴 인식 직사각형 그리는 용도
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

#This function calculates and return eye aspect ratio
#눈 종횡비(EAR) 계산 함수
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])  #두 점사이의 거리를 구하는데 사용하는 함수
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A+B) / (2.0 * C)
    return ear

#Load face detector and predictor, uses dlib shape predictor file
#68개의 얼굴 랜드마크 추출
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#Extract indexes of facial landmarks for the left and right eye
#위에서 추출한 랜드마크에서 왼쪽 눈과 오른쪽 눈의 랜드마크 좌표 추출
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

#Start webcam video capture
#첫번째(0) 카메라를 VideoCapture 타입의 객체로 얻어옴
video_capture = cv2.VideoCapture(0)

#Give some time for camera to initialize(not required)
time.sleep(2)

while(True):
    #Read each frame and flip it, and convert to grayscale
    # ret : frame capture결과(boolean)
    # frame : Capture한 frame
    ret, frame = video_capture.read()   #비디오 읽기
    frame = cv2.flip(frame,1)           #프레임 좌우반전(flipcode > 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #color space 변환(BGR->Grayscale로 변환)

    #Detect facial points through detector function
    faces = detector(gray, 0)

    #Detect faces through haarcascade_frontalface_default.xml
    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

    #Draw rectangle around each face detected
    #얼굴 사각형 그리기
    for (x,y,w,h) in face_rectangle:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    #Detect facial points
    for face in faces:

        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        #Get array of coordinates of leftEye and rightEye
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        #Calculate aspect ratio of both eyes
        #양쪽 눈의 EAR 계산
        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)

        #양쪽 눈 EAR 값의 평균 계산
        eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

        #Use hull to remove convex contour discrepencies and draw eye shape around eyes
        #눈의 경계선 그리기
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #눈 EAR 표시
        cv2.putText(frame, "EAR : {:.2f}".format(eyeAspectRatio), (300, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 30, 20), 2)

        #Detect if eye aspect ratio is less than threshold
        #졸음판단(현재 EAR이 임계값보다 작은지 확인)
        if(eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD):
            #임계값보다 EAR 작으면(눈 감고 있는 상태)
            COUNTER += 1
            #If no. of frames is greater than threshold frames,
            if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                #눈을 깜빡인건지 졸고있는건지 확인
                pygame.mixer.music.play(-1)
                cv2.putText(frame, "You are Drowsy", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        else:
            #임계값보다 EAR 크면(눈 뜨고 있는 상태)
            pygame.mixer.music.stop()   #소리 출력 정지
            COUNTER = 0                 #카운터 초기화
    
    #Show video feed
    cv2.imshow('Video', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

#Finally when video capture is over, release the video capture and destroyAllWindows
video_capture.release()
cv2.destroyAllWindows()
