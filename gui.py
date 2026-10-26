# File name: gui.py
# Date: 21-6-2020
# Version: python 3.8.3 , opencv 3.4.9(pip install opencv-python==3.4.9.33)
# Explanation: This program Detects eyes,mouth,nose,frontface using Haar casscade and predicts disease based on detection.. 
################################on startbuttton click webcam will be on and detection will be start,on stop video will be off
# assumption: only one person is in front of webcam,make sure you hide at least one eye for headache detection
# usage: python gui.py
# created by: Bhavisha Parmar

import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import os
import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk

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

out = cv2.VideoWriter('result.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (640,480)) #for saving output in result.avi file, change name everytime you want to save new result



################ functions ######################
def stop():
	running = False
	cap.release()
	out.release()
	img = Image.open('images/0.png')#default image when video is not running put whatever image you like in images folder named 0.png
	imgtk = ImageTk.PhotoImage(image=img)
	lmain.imgtk = imgtk
	lmain.configure(image=imgtk)
############ fire #############
def OpenFile_s():
	global cap
	global running
	global width
	global height
	running = True
	cap = cv2.VideoCapture(0)#open webcam
	if cap.isOpened(): 
		width  = int(cap.get(3))
		height = int(cap.get(4))
	show_frame()
    
def show_frame():
	if running==True:
		ret, frame = cap.read()
		if ret==True:
			frame = disease(frame)
			out.write(frame)
			cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
			img = Image.fromarray(cv2image)
			imgtk = ImageTk.PhotoImage(image=img)
			lmain.imgtk = imgtk
			lmain.configure(image=imgtk)
			lmain.after(10, show_frame)
		else:
			img = Image.open('images/0.png')#default image when video is not running put whatever image you like in images folder named 0.png
			imgtk = ImageTk.PhotoImage(image=img)
			lmain.imgtk = imgtk
			lmain.configure(image=imgtk)
def disease(frame):
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
		w3['text']="you have headache"
		cv2.putText(frame,"H", (width-50,height-10), font, 0.8,(100,0,200),2) # last for arguments are position,size,color in BGR format and bold/thikness
	elif (len(nose)<1 or len(mouth)<1):
		w3['text']="you have cough"
		cv2.putText(frame,"C", (width-50,height-10), font, 0.8,(100,0,200),2)# last for arguments are position,size,color in BGR format and bold/thikness
	elif (len(smiles)==0):
		w3['text']="you are sad/depressed"
		cv2.putText(frame,"S/D", (width-50,height-10), font, 0.8,(100,0,200),2)# last for arguments are position,size,color in BGR format and bold/thikness
	else:
		w3['text']="you are alright :)"
		cv2.putText(frame,"you are alright :)", (width-250,height-10), font, 0.8,(100,0,200),2)# last for arguments are position,size,color in BGR format and bold/thikness
	return frame


################ layout ######################
    
#Set up GUI
window = tk.Tk()  #Makes main window
window.wm_title("hedache and cough detection")
window.config(background="#d9d9d9")#background color
#window.geometry("500x500") #You want the size of the app to be 500x500

#Graphics window
window.grid_rowconfigure(8, weight=1)
window.grid_rowconfigure(9, weight=1)
window.grid_columnconfigure(2, weight=1)
window.grid_columnconfigure(3, weight=1)

# Title  Raw =0
w2 = tk.Label(window, text="Hedache and Cough Detection",font=("Helvetica", 26),background="#d9d9d9",justify='center')#heading
w2.grid(row=0, column=0,columnspan =5)

#Capture video frames Raw =1 col=0
imageFrame = tk.Frame(window,background="#d9d9d9")
imageFrame.grid(row=1, column=0,sticky = N+S,rowspan=6,pady=(10,10),columnspan =5)
img = Image.open('images/0.png')
imgtk = ImageTk.PhotoImage(image=img)
lmain = tk.Label(imageFrame)
lmain.grid(row=1, column=0,sticky = N+S,rowspan=6,pady=(10,10),columnspan =5)
lmain.imgtk = imgtk
lmain.configure(image=imgtk)
#Raw =1 col=1
w3 = tk.Label(window, text="",font=("Helvetica", 26),background="#d9d9d9",justify='center')#result letter
w3.grid(row=8, column=0,columnspan =5)
B1=tk.Button(window,text = "Start",command= OpenFile_s,bg='#df9f9f',activebackground='#ff3399',justify='center',width=15,relief='raised',font=("Helvetica", 15))#start button
B1.grid(row=9,column=2,pady=(10,10))
#B1.place(relx=0.5,rely=0.5,anchor=CENTER)
B2=tk.Button(window,text = "Stop",command= stop,bg='#df9f9f',activebackground='#ff3399',justify='center',width=15,relief='raised',font=("Helvetica", 15))#stopbutton
B2.grid(row=9,column=3,pady=(10,10))
#B2.place(relx=0.5,rely=0.5,anchor=CENTER)

window.mainloop()  #Starts GUI

