#!/usr/bin/python3
from PIL import Image, ImageSequence
import numpy as np
import warnings
import pprint
#from bluetooth import *

############ For bluetooth! #####################3
# target_name = "ArduinoNanoBLE"
# target_address = None

# nearby_devices = bluetooth.discover_devices()

# for bdaddr in nearby_devices:
#     if target_name == bluetooth.lookup_name( bdaddr ):
#         target_address = bdaddr
#         break

# if target_address is not None:
#     print "found target bluetooth device with address ", target_address
# else:
#     print "could not find target bluetooth device nearby"


# port = 3
# s = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
# s.connect((serverMACAddress, port))

#######################################################3
im = Image.open("teaser.gif")
print(im)
# resizeImg = imresize(depthImg,[300 410]); % 320 x 240 pixels
# width = 410; height = 300;
index = 1

height = 390
width = 480
label = np.zeros((390, 480))

lev = np.zeros((480, 10))
level = np.zeros((10,10))



for frame in ImageSequence.Iterator(im):
    
    index += 1
    print(index)
    # index % 10 !=0: continue
    
    # frame.save("frame%d.png" % index)
    width, height = frame.size
    channels = 1

    pixel_values = list(frame.getdata())
    pixel_values = np.array(pixel_values).reshape((width, height, channels))
    # pprint.pprint(pixel_values.shape)
    pixeling = pixel_values[:,:,0]
    # pprint.pprint(pixel_values)
    for posy in range(width):
        for posx in range(height):
            if pixel_values[posx][posy] > 230 :
                label[posx,posy] = 1
            elif pixel_values[posx][posy] <= 230 & pixel_values[posx][posy] > 200 :
                label[posx,posy] = 2
            elif pixel_values[posx][posy] <= 200 & pixel_values[posx][posy] > 170 :
                label[posx,posy] = 3
            elif pixel_values[posx][posy] <=170 & pixel_values[posx][posy] > 140 :
                label[posx,posy] = 4
            elif pixel_values[posx][posy] <=140 & pixel_values[posx][posy] > 110 :
                label[posx,posy] = 5
            elif pixel_values[posx][posy] <=110 & pixel_values[posx][posy] > 80 :
                label[posx,posy] = 6
            elif pixel_values[posx][posy] <=80 & pixel_values[posx][posy] > 50 :
                label[posx,posy] = 7
            else :
                label[posx,posy] = 8

    for posy in range(0, width, 48) :
        for posx in range(0, height, 39) :
            level[posx//48,posy//39] = np.around(np.mean(label[posx:posx+48,posy:posy+39]))
			# s.send(level) ############# for bluetooth!
    

    pprint.pprint(level)



