# -*- coding: utf-8 -*-

# reserch reference : 
#1) https://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html(open cv 설명서)
#2) METHODS FOR HEMODYNAMIC PARAMETERSMEASUREMENT USING THE LASER SPECKLE EFFECT IN MACRO AND MICROCIRCULATION p.15~47
#3) https://webnautes.tistory.com/1180(broadcasting)
#4) https://blog.naver.com/PostView.nhn?blogId=phj8498&logNo=221271984956(color map)
#5) https://jangjy.tistory.com/m/337(uint8 환경에서 한국어 사용)
#6) 진단초음파 영상처리를 위한 적응 pseudimedian 필터. 전기전자학회논문지 vol7. no.2.허은석

# version info
# python 3.6
# open cv 3.4.2
# numpy 1.16.2

import numpy as np
import cv2
import time
import datetime
import tensorflow as tf
import os

def nothing(x):
    pass

def mean_imaging(image,mask):
    output=cv2.filter2D(image, -1, mask)
    return output
#refernce 2)의 3.30의 식을 적용하였음.
#원 영상 mean convolution
    
def std_imaging(image1,image2,mask):
    sequence1 = (image1-image2)**2
    output = mean_imaging(sequence1, mask)
    output = (output)**(1/2)
    return output
#refence 2)의 3.31의 식을 적용하였음.
#(원영상-mean convolution)의 제곱을 mean convolution 후 sqrt

def rescale1(image,i, j):
    retval, rescale_image = cv2.threshold(image, i, j, cv2.THRESH_TOZERO_INV)
    #만약 image의 픽셀값이 i 이상이면 0, 아니면 그대로
    reval2, rescale_image = cv2.threshold(rescale_image, j, i, cv2.THRESH_TOZERO)
    #rescale_image의 픽셀값이 j 이상이면 그대로 아니면 0
    return rescale_image
    #return되는 값은 i~j사이의 영역만 남는 rescale_image가 된다.
 
def running_avg(input_vid,output_vid,power):

    temp = cv2.accumulateWeighted(input_vid,output_vid,1/(power+1))
    run_avg_video = cv2.convertScaleAbs(temp)
    return run_avg_video
# cv2.accumulateWeighted(src, dst, alpha[, mask]) is opencv function to find running average
# when alpha raise, function update speed (how fast the accumulator “forgets” about earlier images).
    
def make_folder(folder_name):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
#영상이 저장될 폴더를 만드는 함수
#실행될때마다 동일한 폴더가 생성되지 않도록 동일한 이름의 폴더 유무를 확인 후 없는 경우만 폴더 생성

def imwrite_kor(name, img, params=None):
    try:
        ext = os.path.splitext(name)[1]
        result, n = cv2.imencode(ext, img, params)
        #한글 경로를 인식할 수 있도록 imencode함수를 사용해 저장
        #imencode(buffer,flags)
        #buffer : 배열 또는 바이트 벡터 형태 입력
        #flags : 읽어들이는 이미지 이름
        if result:
            with open(name,mode='w+b') as f:
                n.tofile(f)
            return True
        
        else:
            return False
        
    except Exception as e:
        print(e)
        return False
    
#opencv가 파일경로에 한글이 존재하는 경우 인식하지 못하는 문제를 해결하기위한 함수
#출처 : https://jangjy.tistory.com/m/337(개인블로그)

def applyCustomColormap(image):
    
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    
    lut[:, 0, 2] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,55,55,55,55,55,55,55,55,55,
       55,55,55,55,55,55,55,109,109,109,109,109,109,109,109,109,109,109,109,109,
       109,109,109,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,
       11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,65,65,65,65,65,65,65,65,
       65,65,65,65,65,65,65,65,119,119,119,119,119,119,119,119,119,119,119,119,
       119,119,119,119,173,173,173,173,173,173,173,173,173,173,173,173,173,173,
       173,173,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
       255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
       255,255,255,255,255,255,255,255,255,255,255,255,255,255,237,237,237,237,
       237,237,237,237,237,237,237,237,237,237,237,237,255,255,255,255,255,255,
       255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
       255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
       255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
       255,255,255,255,255]
    #R components
    lut[:, 0, 1] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,108,108,108,108,108,108,108,108,108,108,108,108,108,108,108,108,162,162,162,162,162,162,162,162,162,162,162,162,162,162,162,201,201,201,201,201,201,201,201,201,201,201,201,201,201,201,201,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,223,223,223,223,223,223,223,223,223,223,223,223,223,223,223,223,169,169,169,169,169,169,169,169,169,169,169,169,169,169,169,169,144,144,144,144,144,144,144,144,144,144,144,144,144,144,144,144,144,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #G components
    lut[:, 0, 0] = [237,237,237,237,237,237,237,237,237,237,237,237,237,237,237,237,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,112,112,112,112,112,112,112,112,112,112,112,112,112,112,112,112,166,166,166,166,166,166,166,166,166,166,166,166,166,166,166,166,144,144,144,144,144,144,144,144,144,144,144,144,144,144,144,144,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,144,144,144,144,144,144,144,144,144,144,144,144,144,144,144,144,144,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #B components
    colorimage = cv2.LUT(image, lut)
    
    # 파랑 :(0,0,237),(55,54,255),(109,108,255),(163,162,255)
    #초록 : (11,201,4),(65,255,58),(119,255,112),(173,255,166)
    #주황 : (255,255,144),(255,255,90),(255,223,36),(237,169,0)
    #빨강 : (255,144,144),(255,90,90),(255,36,36),(255,0,0) 8가지 색으로 맵핑
    #ref : https://github.com/spmallick/learnopencv/blob/master/Colormap/custom_colormap.py
    #https://mrsnake.tistory.com/142
    #https://www.ft.unicamp.br/docentes/magic/khoros/html-dip/c4/s10/front-page.html
    return colorimage

def mouse_callback(event, x, y, flags, param):
    global clicked_x, clicked_y,mouse_is_pressing

    if event == cv2.EVENT_LBUTTONDOWN:

        mouse_is_pressing = True
        clicked_x, clicked_y = x, y
        #마우스를 클릭했을때부터 좌표가 찍히도록 설정

    elif event == cv2.EVENT_LBUTTONUP:

        mouse_is_pressing = False 
        roi = lsci_copy[ clicked_y:y, clicked_x:x ]
        #lsci 특정 영역만을 가져온다. 
        v= cv2.mean(roi)
        list(v)
        velo = v[0]
        velo = (1/velo)**2
        #속도 = (1/lsci)^2 임을 이용해서 속도를 구한다. v= [roi내부의 lsci평균값]이다.
        print(velo)
        cv2.imshow('roi for velocity',roi)
# 원본 영역에서부터 두 점 (start_y, start_x), (x,y)로 구성되는 사각영역을 잘라내어 
#image_cut에 저장하고 그 영역을 image_cut이라는 윈도우에 보여준다.


#############기초 설정#####################


mean_mask = np.array((
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]), dtype="int")/25
#영상의 평균을 구하기위한 mean filter 생성, 5x5사이즈 사용

cv2.namedWindow('color_lsci_image', cv2.WINDOW_NORMAL)
cv2.createTrackbar('High', 'color_lsci_image',0, 255, nothing)
cv2.setTrackbarPos('High', 'color_lsci_image', 68)
cv2.createTrackbar('Low', 'color_lsci_image',0, 255, nothing)
cv2.setTrackbarPos('Low', 'color_lsci_image', 2)
cv2.createTrackbar('moving_avg', 'color_lsci_image',0, 14, nothing)
cv2.setTrackbarPos('moving_avg', 'color_lsci_image', 1)


mouse_is_pressing = False
clicked_x, clicked_y = 1, 1
cv2.setMouseCallback('velo', mouse_callback)
#영상처리에 필요한 트랙바를 만들기 위한 초기설정
#namedWindow로 트랙바가 들어갈 창의 이름설정
#createTrackbar로 트랙바를 i,j이라는 이름으로 color_lsci_image 윈도우에 나타내게함
        
patient_name = input("환자의 이름을 입력해주세요")
patient_age = input("환자의 나이를 입력해주세요")
patient_sex = input("환자의 성별을 입력해주세요")    
patient_info = patient_name + "(" + patient_sex + patient_age + ")"
#환자의 정보를 입력받고 조합하여 저장

now = datetime.datetime.now().strftime("%Y-%m-%d")
#이미지가 저장되는 날짜 확인

folder_root =  'C:/Users/LeeSeunghoon/Desktop'
path = folder_root + "/" + now
make_folder(path)
#이미지가 저장되는 날짜와 동일한 날짜를 이름으로 가진 폴더 생성

with tf.device('/gpu:1'):
#tensorflow를 통한 gpu사용

    vid = cv2.VideoCapture(1)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT,960)
    _, buffer = vid.read()
    buffer_vid = np.float32(buffer)
    
    prev_time=0
    capture_count = 0
    #영상 입력 시작 시점을 0초로 기준하여 재생시간 카운트할 수 있도록 prev_time 설정
    #영상 저장 시 순차적으로 번호를 메겨 저장할 수 있도록 capture_count 설정
    raw_buffer = cv2.cvtColor(buffer, cv2.COLOR_BGR2GRAY)
    raw_buffer = cv2.flip(raw_buffer, 1)
    
    while True:
        ret, frame = vid.read()
        key = cv2.waitKey(3)
        
        if ret:
            raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            raw = cv2.flip(raw, 1)

            lsc_mean = mean_imaging(raw, mean_mask)
            #원 영상의 평균으로 lsci의 분모역할을 할 영상
            
            lsc_std = std_imaging(raw, lsc_mean, mean_mask)
            #원 영상의 분산으로 lsci의 분자 역할을 할 영상
           
            lsc = lsc_std/lsc_mean
            #위에서 구한 두개의 이미지를 나누어 각 value의 lsci 계산
            #배열간의 나눗셈은 실제로 불가능하나 matlab, python등 프로그램은 배열간 나눗셈을 각 인자간의 나눗셈으로 계산하여 결과를 나타내준다
            #lsci1은 dense_OF에 사용되며 lsci2는 직접영상처리에 사용된다.
            
            lsci_copy = lsc.copy()
            #속도 계산에 사용될 LSCI_image 복사본
            
            i = cv2.getTrackbarPos('High', 'color_lsci_image')
            j = cv2.getTrackbarPos('Low','color_lsci_image')
            m = cv2.getTrackbarPos('moving_avg', 'color_lsci_image')
            #초기설정해둔 트랙바의 값을 불러온다.
            
            lsc = cv2.normalize(lsc, lsc, 255,0, cv2.NORM_MINMAX)
            lsc = rescale1(lsc, i, j)
            #lsci2 이미지를 255~0까지 정규화시킨 후 정규화 범위를 (i, j) 값을 제외한 모든값을 saturation 시킨다.
            cv2.normalize(lsc, lsc, 255, 0, cv2.NORM_MINMAX)
            #(j,i)영역만 다시 255~0까지 정규화시켜준다(min이 255, max가 0)
            
            lsc2 = cv2.cvtColor(lsc.astype('uint8'), cv2.COLOR_GRAY2BGR)
            
            color_lsc = applyCustomColormap(lsc2)
            
            color_lsci = running_avg(color_lsc, buffer_vid, m)
            color_lsci = cv2.bitwise_not(color_lsci)
            
            current_time = time.time()
            sec = current_time - prev_time
            prev_time = current_time
            
            vid_fps = 1/(sec)
            fps_str = "FPS : %0.1f" % vid_fps
            #영상의 프레임 계산
           
            cv2.putText(color_lsci, fps_str, (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            #계산된 프레임을 lsci영상과 DOF영상에 나타냄
            
            cv2.imshow('RAW', raw)
            cv2.imshow('color_lsci_image', color_lsci)
            
            cv2.imshow('velocity', lsci_copy)
            #영상재생
            
            cv2.setMouseCallback('velocity', mouse_callback)
            #veloctiy 윈도우에서 마우스를 이용해 속도를 구할 수 있도록함.
            
        else:
            print('에러가 발생하였습니다.')
            break
            #영상 입력이 잘못된경우 에러메세지 출력 및 종료
            
        if key == 27:
            break
            #ESC를 누르면 while문을 탈출한다.(실행종료)
        
        elif key == ord('s'):
            print('사진 저장')
            capture_count = capture_count+1
            
            imwrite_kor(os.path.join(path, patient_info + '_' + str(capture_count) + '.bmp'), raw)
            imwrite_kor(os.path.join(path, patient_info + '_' + str(capture_count) + '_speckle' + '.bmp'), color_lsci)
           
            #키보드의 's'를 누르면 사진 2장이 저장된다.
       
    vid.release()
    cv2.destroyAllWindows()