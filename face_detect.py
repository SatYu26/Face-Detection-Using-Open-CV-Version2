import cv2
import matplotlib.pyplot as plt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") #path for haar classifier

cap = cv2.VideoCapture(0) #video capturing in camera
while True:
    ret,img = cap.read() #to read the image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #to convert the image in gray

    faces = face_cascade.detectMultiScale(gray,1.3,5) #to read the faces using the haar classifier

    for x,y,w,h in faces:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),3) #to draw the blue color (B,G,R) rectangle around the face of thickness 3
    cv2.imshow('Face Detection',gray)
    plt.show()

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()