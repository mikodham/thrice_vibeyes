#!/usr/bin/python3
# from PIL import Image, ImageSequence
# import matplotlib.pyplot as plt
import numpy as np
# import warnings
# from bluetooth import *

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


def dep2level(frame):  # input is disparity
    width, height = frame.shape  # 480, 140
    label = np.zeros((width, height))
    level = np.zeros((10,10))
    many = (frame < 60).astype(int)
    for boundary in [100, 150, 200, 220, 240, 250, 252]:
        many += (frame < boundary).astype(int)

    for posx in range(0, width, width//10) :
        for posy in range(0, height, height//10):
            mean = np.around(np.mean(many[posx:posx+(width//10), posy:posy+(height//10)]))
            level[posx // (width // 10), posy // (height // 10)] = mean
            many[posx:posx+(width//10),posy:posy+(height//10)] = mean
    print(level)
    return many

    # print(many)
    # for posx in range(width-1):  # 0~479
    #     for posy in range(height-1):  # 0~259
    #         if pixel_values[posx][posy] > 230 :
    #             label[posx,posy] = 0
    #         elif pixel_values[posx][posy] <= 230 and pixel_values[posx][posy] > 200 :
    #             label[posx,posy] = 1
    #         elif pixel_values[posx][posy] <= 200 and pixel_values[posx][posy] > 170 :
    #             label[posx,posy] = 2
    #         elif pixel_values[posx][posy] <=170 and pixel_values[posx][posy] > 140 :
    #             label[posx,posy] = 3
    #         elif pixel_values[posx][posy] <=140 and pixel_values[posx][posy] > 120 :
    #             label[posx,posy] = 4
    #         elif pixel_values[posx][posy] <=120 and  pixel_values[posx][posy] > 100 :
    #             label[posx,posy] = 5
    #         elif pixel_values[posx][posy] <=100 and  pixel_values[posx][posy] > 80 :
    #             label[posx,posy] = 6
    #         elif pixel_values[posx][posy] <=80 and  pixel_values[posx][posy] > 60 :
    #             label[posx,posy] = 7
    #         else :
    #             label[posx,posy] = 8
    # print(label.shape)



    # np.savetxt("level.csv", level, delimiter=",")
    # plt.imshow(frame)
    # plt.pause(0.00000001)




