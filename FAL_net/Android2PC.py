# https://stackoverflow.com/questions/53130370/how-to-access-phone-camera-using-python-script/53131060
import urllib3
import urllib.request
import cv2
import numpy as np
import ssl
import time

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = 'http://192.168.137.153:8080/video'
cv2.startWindowThread()
cv2.namedWindow("preview")
cap = cv2.VideoCapture(url)
while True:
    # imgResp = urllib3.urlopen(url)
    # imgResp = urllib.request.urlopen(url)
    # imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    # img = cv2.imdecode(imgNp, -1)
    ret, frame = cap.read()
    if frame is not None:
        cv2.imshow("preview", cv2.resize(frame,(600,400)))
        # cv2.imshow("preview", frame)
    q = cv2.waitKey(1)

    if q == ord("q"):
        break;
cv2.destroyAllWindows()
