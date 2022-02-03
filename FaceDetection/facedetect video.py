import cv2

# Loading pre trained data on face frontals from opencv
trained_face_data= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose a video to detect faces in, 0 for default webcam
webcam=cv2.VideoCapture(0, cv2.CAP_DSHOW)

#Every face has a unoique user id
face_id= 1

count= 0
while True:
    successful_frame_read, frame = webcam.read()
    #Converting to grayscale
    grayscaled_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)




    #Detect Face

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
   
    #print(face_coordinates)

    #Draw rectangles around the face

    for(x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,0),2)

        



    
    cv2.imshow("rifat's face detector",frame)
    # we use this so that the image doesn't close instantly
    key=cv2.waitKey(1)
    
    if key== 81 or key ==113:  #press q to stop
        break


webcam.release()

print('code completed')