import time, pickle, requests, threading
from picamera import PiCamera
from picamera.array import PiRGBArray

import time, pickle, requests, threading,os,datetime,glob
from picamera import PiCamera
from picamera.array import PiRGBArray

class sendImg:
  def __init__(self):
    self.count = 0

    # Camera initialisation
    self.camera = PiCamera()
    self.camera.exposure_mode = "sports"
    self.camera.resolution = (640, 480)
    self.output = PiRGBArray(self.camera)

    self.camera.start_preview()
    threading.Thread(target=time.sleep, args=(2,)).start()

  def takePic(self):
    self.camera.capture(self.output, 'bgr')
    frame = self.output.array
    self.output.truncate(0)
    self.count += 1
    data = pickle.dumps(frame)
    # send to Laptop via HTTP POST
    r = requests.post("http://192.168.6.18:8123", data=data) 
    print("Image", self.count, "sent")

if __name__ == '__main__':
      obj = sendImg()
      obj.takePic()
