# File name: disease_detect.py
# Date: 21-6-2020
# Version: python 3.8.3 , opencv 3.4.9(pip install opencv-python==3.4.9.33)
# Explanation: This program Detects eyes,mouth,nose,frontface using Haar casscade and predicts disease based on detection..
# assumption: only one person is in front of webcam,make sure you hide at least one eye for headache detection
# usage: python disease_detect.py
# created by: Bhavisha Parmar

import cv2
import time
import numpy as np
from os import listdir
from os.path import isfile, join

eyecascadePath = "./xmlfiles/haarcascade_eye.xml" #make sure these files are in "xmlfiles" folder in same folder as this code or change the path 
nosecascadePath = "./xmlfiles/Nariz.xml"#make sure these files are in "xmlfiles" folder in same folder as this code or change the path
mouthcascadePath = "./xmlfiles/Mouth.xml"#make sure these files are in "xmlfiles" folder in same folder as this code or change the path
facecascadePath = "./xmlfiles/haarcascade_frontalface_default.xml"#make sure these files are in "xmlfiles" folder in same folder as this code or change the path


smile_cascade = cv2.CascadeClassifier('./xmlfiles/haarcascade_smile.xml')
eyeCascade = cv2.CascadeClassifier(eyecascadePath);
noseCascade = cv2.CascadeClassifier(nosecascadePath);
mouthCascade = cv2.CascadeClassifier(mouthcascadePath);
faceCascade = cv2.CascadeClassifier(facecascadePath);


font = cv2.FONT_HERSHEY_SIMPLEX


video_capture = cv2.VideoCapture(0)#for recorded video write name of the video in single quote instead of 0

if video_capture.isOpened(): 
    width  = int(video_capture.get(3)) # float
    height = int(video_capture.get(4)) # float

out = cv2.VideoWriter('result.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (640,480)) #for saving output in result.avi file, change name everytime you want to save new result
while True:
	ret, frame = video_capture.read()
	if ret == True:
		start = time.time()
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		eyes = eyeCascade.detectMultiScale(gray, 1.2,5)
		nose = noseCascade.detectMultiScale(gray, 1.2,11)
		mouth = mouthCascade.detectMultiScale(gray, 3,23)
		faces = faceCascade.detectMultiScale(gray, 1.3,3)
		if len(faces)!=0:	
			for (x, y, w, h) in faces: 
				cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2) 
				roi_gray = gray[y:y + h, x:x + w] 
				roi_color = frame[y:y + h, x:x + w] 
				smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
		else:
			smiles=[]		
		
		if (len(eyes)<=1):
			cv2.putText(frame,"you have headache", (width-250,height-10), font, 0.8,(100,0,200),2) # last for arguments are position,size,color in BGR format and bold/thikness
		elif (len(nose)<1 or len(mouth)<1):
			cv2.putText(frame,"you have cough", (width-250,height-10), font, 0.8,(100,0,200),2)# last for arguments are position,size,color in BGR format and bold/thikness
		elif (len(smiles)==0):
			cv2.putText(frame,"you are sad or depressed", (width-250,height-10), font, 0.8,(100,0,200),2)# last for arguments are position,size,color in BGR format and bold/thikness
		else:
			cv2.putText(frame,"you are alright :)", (width-250,height-10), font, 0.8,(100,0,200),2)# last for arguments are position,size,color in BGR format and bold/thikness
		
		cv2.imshow('Result',frame)
		out.write(frame)
		end = time.time()
		print("time = %.6f",(end-start))
		if cv2.waitKey(33) & 0xFF == ord('q'): # on press 'q' the process will terminates
			cv2.destroyAllWindows()
			out.release()
			break