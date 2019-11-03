import tensorflow as tf
import keras
import cv2
print('\n')
print('Libraries Imported.')
print('\n')

#reading the image
img = cv2.imread('C:\\Users\\ayush\\PycharmProjects\\tf_oo\\image1.jpg', 1) #the (1) indicates that the image has been read in RGB format
print(img)

print ('\n')

#reading the image in grayscale
img_1 = cv2.imread('C:\\Users\\ayush\\PycharmProjects\\tf_oo\\image1.jpg', 0)
print(img_1)

#knowing the shape of the image
##as we are dealing with numpy array, we will just perform numpy array

print (img.shape)
print ('\n')
print(img_1.shape)

#displaying the image
#cv2.imshow('Test Image', img) #Test Image is the name of the file
#cv2.imshow('Test Image in Grayscale', img_1) #Test Image in Grayscale is the name of the file
#cv2.waitKey(2000) #this zero indicates that the window that will display the image will only be closed if we press anything and if there is any number then it will wait for that amount of miliseconds
#cv2.destroyAllWindows()

##if we want to resize the image
#resizing = cv2.resize(img_1,(363,924))
#cv2.imshow("Resizing the grayscale image", resizing)
#cv2.waitKey(2000)
#cv2.destroyAllWindows()

## as we see from the above function that our image wasnt reduced to symmetric size, so we usually divide the function by two, three or according to our dimensions
resize_it_again = cv2.resize(img_1, (int(img_1.shape[1]/2), int(img_1.shape[0]/2)))
cv2.imshow('Reducing the image to half', resize_it_again)
cv2.waitKey(0)
cv2.destroyAllWindows()

