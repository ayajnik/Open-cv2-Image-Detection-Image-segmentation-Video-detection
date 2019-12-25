import numpy as np
import cv2
print('\n')
print('Libraries Imported.')
print('\n')



##OpenCv has BGR format

##########################################################displaying the image#################################################################
#img = cv2.imread('delhi.jpg')
#print(img.shape)

#cv2.imshow('Image', img)
#cv2.waitKey(0)    ##zero in the argument tells us that if we press anywhere on the window then only the image will be closed
#cv2.destroyAllWindows()

###################################################displaying the numpy arrays of each color channels##########################################
#blue = img[:,:,0]
#print(blue)
#print('\n')
#green = img[:,:, 1]
#print(green)
#print('\n')
#red = img[:,:, 2]
#print(red)
#print('\n')
#cv2.imshow("Image", img)
#cv2.imshow("Blue", blue)
#cv2.imshow("Green", green)
#cv2.imshow("Red", red)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
##############################################################viewing video####################################################################

#cap = cv2.VideoCapture(0)
#if cap.isOpened() == False:
#    print("The file could not be opened properly.")

#while True:
#    ret, frame = cap.read()

#    if ret == True:
#        cv2.imshow('Frame', frame)

#    if cv2.waitKey(25) & 0xFF == 27:
#        break

#    else:
#        break
#cap.release()
#cv2.destroyAllWindows()
#################################################################classification of image#######################################################
#img = cv2.imread('delhi.jpg')
#print(img.shape)

#all_rows = open('synset_words.txt').read().strip().split("\n")
#classes = [r[r.find(' ') + 1:] for r in all_rows]
#net = cv2.dnn.readNetFromCaffe('bvlc_googlenet.prototxt','bvlc_googlenet.caffemodel')
#blob = cv2.dnn.blobFromImage(img, 1, (224,224))
#net.setInput(blob)
#outp = net.forward()
#print(outp)
#idx = np.argsort(outp[0])[::-1][:5]
#print("The image shown can be : ")
#print("\n")
#for (i,id) in enumerate(idx):
#    print('{}. {} ({}): Probability {:.3}%'.format(i+1, classes[id], id, outp[0][id]*100))

#cv2.imshow('Image', img)
#cv2.waitKey(0)    ##zero in the argument tells us that if we press anywhere on the window then only the image will be closed
#cv2.destroyAllWindows()

################################################################video classification###########################################################
cap = cv2.VideoCapture(0)   ##---->> if we put 0 in the argument, that means we are using the webcam or else we pass in the path for the video

all_rows = open('synset_words.txt').read().strip().split("\n")

classes = [r[r.find(' ') + 1:] for r in all_rows]

net = cv2.dnn.readNetFromCaffe('bvlc_googlenet.prototxt','bvlc_googlenet.caffemodel')

if cap.isOpened() == False:
    print('Cannot open file or video stream')

while True:
    ret, frame = cap.read()

    blob = cv2.dnn.blobFromImage(frame, 1, (224,224))

    net.setInput(blob)

    outp = net.forward()

    r=1
    for i in np.argsort(outp[0])[::-1][:5]:
        txt = ' "%s" probability "%.3f" ' % (classes[i], outp[0][i] * 100)
        cv2.putText(frame, txt, (0, 25 + 40*r), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        r+=1

    if ret == True:
        cv2.imshow('Frame', frame)

        if cv2.waitKey(25) & 0xFF == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
