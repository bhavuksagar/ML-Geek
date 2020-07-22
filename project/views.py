from django.shortcuts import render,redirect
from django.http import HttpResponse
import cv2
from PIL import Image
import numpy as np
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle
from sklearn.naive_bayes import MultinomialNB
from .models import blog
from django.core.mail import send_mail

# Create your views here.
def homePage(request):
    return render(request,'index.html')
def aboutPage(request):
    return render(request,'about.html')

def contactPage(request):
    return render(request,'contact.html')

def facePage(request):
    return render(request,'face.html')

def languagePage(request):
    return render(request,'language.html')

def projectPage(request):
    return render(request,'project.html')

def reviewPage(request):
    return render(request,'review.html')

def smilePage(request):
    return render(request,'smile.html')

def thugPage(request):
    return render(request,'thug.html')

def spamPage(request):
    return render(request,'spam.html')

def thugrun(request):
    maskPath = "/Users/sahilsagar/Desktop/ML/Project/Snapchat/mask.png"
    cascPath = "/Users/sahilsagar/Desktop/ML/Project/Snapchat/face.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    mask = Image.open(maskPath)
    def thug_mask(image):
    	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    	faces = faceCascade.detectMultiScale(gray, 1.15)
    	background = Image.fromarray(image)
    	for (x,y,w,h) in faces:
    		resized_mask = mask.resize((w,h), Image.ANTIALIAS)
    		offset = (x,y)
    		background.paste(resized_mask, offset, mask=resized_mask)
    	return np.asarray(background)
    cap = cv2.VideoCapture(cv2.CAP_ANY)
    while True:
    	ret, frame = cap.read()
    	if ret == True:
    		cv2.imshow('Live', thug_mask(frame))
    		if cv2.waitKey(1) == 27:
    			break
    cap.release()
    cv2.destroyAllWindows()

    return redirect('/')

def langrun(request):
    if request.method =='POST':
        input=request.POST['input']
        path='/Users/sahilsagar/Desktop/ML-Geeks/main/geek/ML/model'
        model=pickle.load(open(path,'rb'))
        cv_input=model[0].transform([input])
        out=model[1].predict(cv_input)
        return render(request,'language.html',{'out':out[0]})
    else:
        return render(request,'language.html',{})

def spamrun(request):
    if request.method == 'POST':
        input=request.POST['input']
        path="/Users/sahilsagar/Desktop/ML-Geeks/main/geek/ML/spamModel"
        model=pickle.load(open(path,'rb'))
        cv_input=model[1].transform([input])
        out=model[0].predict(cv_input)
        if out=='ham':
            out='Not Spam'
        else:
            out='Spam'
        return render(request,'spam.html',{'out':out})
    else:
        return render(request,'spam.html',{})

def reviewrun(request):
    if request.method =='POST':
        input=request.POST['input']
        path='/Users/sahilsagar/Desktop/ML-Geeks/main/geek/ML/review'
        model=pickle.load(open(path,'rb'))
        cv_input=model[0].transform([input])
        out=model[1].predict(cv_input)
        return render(request,'review.html',{'out':out[0]})
    else:
        return render(request,'review.html',{})

def facerun(request):
    cascPath = "/Users/sahilsagar/Desktop/ML/Project/Snapchat/face.xml"

    face=cv2.CascadeClassifier(cascPath)
    cam=cv2.VideoCapture(0)

    while True:
        check,frame=cam.read()
        faces=face.detectMultiScale(frame,1.3,5)
        for x,y,w,h in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,220),2)
        cv2.imshow("face",frame)
        key=cv2.waitKey(1)
        if key==ord('q'):
            break
    cam.release()
    cv2.destroyAllWindow()
    return redirect('/')

def smilerun(request):
    facePath = "/Users/sahilsagar/Desktop/ML/Project/Snapchat/face.xml"
    smilePath = "smile.xml"
    faceCascade = cv2.CascadeClassifier(facePath)
    smileCascade = cv2.CascadeClassifier(smilePath)

    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)

    sF = 1.05

    while True:

        ret, frame = cap.read() # Capture frame-by-frame
        img = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor= sF,
            minNeighbors=8,
            minSize=(55, 55),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # ---- Draw a rectangle around the faces

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            smile = smileCascade.detectMultiScale(
                roi_gray,
                scaleFactor= 1.7,
                minNeighbors=22,
                minSize=(25, 25),
                flags=cv2.CASCADE_SCALE_IMAGE
                )

            # Set region of interest for smiles
            for (x, y, w, h) in smile:
                print ("Found", len(smile), "smiles!")
                cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 1)
                #print "!!!!!!!!!!!!!!!!!"

        #cv2.cv.Flip(frame, None, 1)
        cv2.imshow('Smile Detector', frame)

        if (cv2.waitKey(1) & 0xff) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def blogPage(request):
    model= blog.objects.order_by('id')
    context={'model':model}
    return render(request,'blogs.html',context)

def contectmail(request):
    if request.method == "POST":
        message_name= request.POST['name']
        message_email= request.POST['email']
        message= request.POST['message']

        send_mail(
        'Mail From :' +message_name,
        message,
        message_email,
        ['cse16311.sbit@gmail.com']
        )
        context= {'message_name':message_name}

        return render(request,'contact.html',context)

    else:
        return render(request,'contact.html')
