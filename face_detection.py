import cv2
import numpy as npy
import face_recognition as face_rec
def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)
mayank = face_rec.load_image_file(r"/Users/mss17082004gmail.com/Desktop/Python projects/Face_recognition/sample_images/mayank.jpeg")
mayank = cv2.cvtColor(mayank, cv2.COLOR_BGR2RGB)
mayank = resize(mayank,0.50)
mayank_test = face_rec.load_image_file(r"/Users/mss17082004gmail.com/Desktop/Python projects/Face_recognition/sample_images/mayank_test.jpeg")
mayank_test = cv2.cvtColor(mayank_test, cv2.COLOR_BGR2RGB)
mayank_test = resize(mayank_test,0.50)
facelocation_mayank = face_rec.face_locations(mayank)[0]
encode_mayank = face_rec.face_encodings(mayank)[0]
cv2.rectangle(mayank, (facelocation_mayank[3], facelocation_mayank[0]), (facelocation_mayank[1], facelocation_mayank[2]), (255, 0, 255), 3)
faceLocation_mayank_test = face_rec.face_locations(mayank_test)[0]
encode_mayank_test = face_rec.face_encodings(mayank_test)[0]
cv2.rectangle(mayank_test, (facelocation_mayank[3], facelocation_mayank[0]), (facelocation_mayank[1], facelocation_mayank[2]), (255, 0, 255), 3)
results = face_rec.compare_faces([encode_mayank], encode_mayank_test)
print(results)
cv2.putText(mayank_test, f'{results}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2 )
cv2.imshow('main_img', mayank)
cv2.imshow('test_img', mayank_test)
cv2.waitKey(0)
cv2.destroyAllWindows()


