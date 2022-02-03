import cv2

# Loading pre trained data on face frontals from opencv
trained_face_data= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose an image to detect faces in
img = cv2.imread('pic1.jpg')

#shows the image and the projectors's name is rifat's face detector
#cv2.imshow("rifat's face detector",img)

# we use this so that the image doesn't close instantly
#cv2.waitKey()

#Converting to grayscale
grayscaled_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#shows the image and the projectors's name is rifat's face detector
#cv2.imshow("rifat's face detector",grayscaled_img)
# we use this so that the image doesn't close instantly
#cv2.waitKey()

#Detect Face

face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

print(face_coordinates)

#Draw rectangles around the face

for(x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,0),2)

cv2.imshow("rifat's face detector",img)
# we use this so that the image doesn't close instantly
cv2.waitKey()


print('code completed')