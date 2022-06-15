import cv2
vidcap = cv2.VideoCapture('data/Victoria/IMG_5095.MOV')
# success, image = vidcap.read()
# image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
# cv2.imshow('img', image)
# vidcap.set(cv2.CAP_PROP_POS_FRAMES, 100)
# success, image = vidcap.read()
# image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
# cv2.imshow('img2', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# exit()

success = True
frame = 0
while success:
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    success, image = vidcap.read()
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    cv2.imwrite("data/all/victoria%d.jpg" % frame, image)
    frame += 30
