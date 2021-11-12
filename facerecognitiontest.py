import cv2
import face_recognition
import numpy as np

imgkrish = face_recognition.load_image_file('Training_images/Krish Naik.jpg')
imgkrish = cv2.cvtColor(imgkrish,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('Training_images/KrishTest.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgkrish)[0]
encodeface = face_recognition.face_encodings(imgkrish)[0]
cv2.rectangle(imgkrish,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

facetest = face_recognition.face_locations(imgTest)[0]
encodetest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(facetest[3],facetest[0]),(facetest[1],facetest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeface], encodetest)
facedist = face_recognition.face_distance([encodeface], encodetest)
cv2.putText(imgTest,f'{results},{round(facedist[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
print(results, facedist)


cv2.imshow('Krish',imgkrish)
cv2.imshow('KrishTest',imgTest)
cv2.waitKey(0)





