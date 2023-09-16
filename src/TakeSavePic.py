#!/usr/bin/env python
# coding: utf-8

# In[22]:


#!pip install opencv-python


# In[23]:


import cv2


# In[24]:


# intialize the webcam and pass a constant which is 0
cam = cv2.VideoCapture(0)


# In[25]:


# title of the app
cv2.namedWindow('python webcam screenshot app')


# In[26]:


# let's assume the number of images gotten is 0
img_counter = 0


# In[27]:


def take_pic(img_counter):
    img_name = f'opencv_frame_{img_counter}.jpg'
    # saves the image as a png file
    cv2.imwrite(img_name, frame)
    print('screenshot taken')
    img_counter += 1


# In[28]:


# while loop
while True:
    # intializing the frame, ret
    ret, frame = cam.read()
    # if statement
    if not ret:
        print('failed to grab frame')
        break
    # the frame will show with the title of test
    cv2.imshow('test', frame)
    #to get continuous live video feed from my laptops webcam
    k  = cv2.waitKey(1)
    # if the escape key is been pressed, the app will stop
    if k%256 == 27:
        print('escape hit, closing the app')
        break
    # if the spacebar key is been pressed
    # screenshots will be taken
    elif k%256  == 32:
        take_pic(img_counter)


# In[29]:


# release the camera
cam.release()


# In[ ]:




