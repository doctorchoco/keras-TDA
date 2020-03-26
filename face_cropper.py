import os
import numpy as np
import cv2
from collections import Counter
from flask import Flask, request, jsonify, make_response
from flask.views import MethodView
import time
'''
Face Cropper total
'''
class FaceCropperAll():
    def __init__(self, marginRatio=1.3, type='mtcnn', resizeFactor=0.5):
        '''
        initializer of face cropper all
        :param marginRatio: ratio of margin
        :param type: type of face cropper:('mtcnn', 'haar' )
        '''
        if(type == 'haar'):
            self.detector = HaarCascadeFaceDetector()
        else:
            assert False, 'Wrong face cropper type...'
    def detectMulti(self, inputList):
        return self.detector.detectMulti(inputList)
'''
Basic face cropper
'''
class FaceCropper():
    def __init__(self, marginRatio=1.3, resizeFactor=1.0):
        '''
        FaceCropper Basic Class
        :param marginRatio: margin of face(default: 1.3)
        '''
        self.marginRatio=1.3
        self.prevX = 0
        self.prevY = 0
        self.prevW = 0
        self.prevH = 0
        self.resizeFactor = resizeFactor
    def cropFace(self, input, x,y,w,h):
        '''
        Crop Face with given bbox
        :param input: input image
        :param x: X
        :param y: Y
        :param w: W
        :param h: H
        :return: cropped image
        '''
        x_n = int(x - (self.marginRatio - 1) / 2.0 * w)
        y_n = int(y - (self.marginRatio - 1) / 2.0 * h)
        w_n = int(w * self.marginRatio)
        h_n = int(h * self.marginRatio)
        return input[y_n:y_n + h_n, x_n:x_n + w_n],x,y,w,h
    def detect(self, input):
        '''
        Face detect with single input
        :param input: single image (W x H x C)
        :return: bbox information(x,y,w,h)
        '''
        pass
    def detectMulti(self, inputList):
        '''
        Face detect with multiple inputs
        :param inputList: multi images (N x W x H x C)
        :return: face cropped image list (N x W' x H' x C)
        '''
        pass
'''
Haar face cropper
'''
class HaarCascadeFaceDetector(FaceCropper):
    def __init__(self, marginRatio=1.3, resizeFactor=1.0):
        super().__init__(marginRatio, resizeFactor)
        # load xml file
        self.faceCascade = cv2.CascadeClassifier('weights/haarcascade_frontalface_default.xml')
        #self.faceCascade = cv2.CascadeClassifier('weights/haarcascade_frontalface_alt_tree.xml')
    def detect(self, img):
        '''
        Face detect with single input
        :param img: single image (W x H x C)
        :return: face cropped image(W' x H' x C)
        '''
        imgResize = cv2.resize(img, dsize=(0,0),fx=self.resizeFactor, fy=self.resizeFactor)
        img8U = np.uint8(imgResize)
        gray = cv2.cvtColor(img8U, cv2.COLOR_BGR2GRAY)
        res = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)
        if(len(res) == 0):
            return self.prevX, self.prevY, self.prevW, self.prevH, False
        max = -999999
        for f in res:
            (x,y,w,h) = [v for v in f]
            size = w * h
            if(size > max):
                max = size
                (maxX, maxY, maxW, maxH) = (x,y,w,h)
        self.prevX = maxX
        self.prevY = maxY
        self.prevW = maxW
        self.prevH = maxH
        return maxX, maxY, maxW, maxH, True
    def detectMulti(self, inputList):
        '''
        Face detect with multiple inputs
        :param inputList: multi images(N x W x H x C)
        :return: face cropped image list(N x W' x H' x C)
                cnt: number of successing face detect
        '''
        resList = []
        cnt=0
        for eachInput in inputList:
            bbox = self.detect(eachInput)
            x, y, w, h, isDetected = bbox
            if(isDetected == True):
                cnt+=1
            res = self.cropFace(eachInput, x, y, w, h)
            resList.append(res[0])
        return cnt

detectorType = 'haar'
#detectorType = 'mtcnn'
detector = FaceCropperAll(type=detectorType, resizeFactor=1.0)

zeroList = []
for i in range(32):
    zeroList.append(np.zeros((224,224,3)))

if(detectorType != 'haar'):
    detector.detectMulti(zeroList)

def testInput(parentPath):
    '''
    test for sample input(M x N x W x H x C)
    M: number of intruders
    N: samples per one intruder(one second)(default: 30(30 fps))
    W: width of image
    H: height of image
    C: number of channel in image(3)
    :return: M x N x W x H x C inputs
    '''
    frameSeq = os.listdir(parentPath)
    frameSeq.sort()
    wholeInputList = []
    intruderInputList = []
    for i in range(len(frameSeq)):
        frameSeqAbsPath = os.path.join(parentPath, frameSeq[i])
        img = cv2.imread(frameSeqAbsPath)
        intruderInputList.append(img)

    wholeInputList.append(intruderInputList)
    return np.array(wholeInputList)

def testFaceCropper(inputList):
    global detector
    print('TEST FACE CROPPER')
    cnt = detector.detectMulti(inputList)
    return cnt


class API(MethodView):
    def __init__(self):
        pass
    def post(self, imagePath):
        a = time.time()
        dir_name = imagePath.replace("=","/")
        k = testInput(dir_name)
        numTotalImages = np.shape(k)[1]
        print(numTotalImages)
        numSuccessFaceDetect = 0
        # for online stream processing~!~!


        for i in range(len(k)):
            cnt = testFaceCropper(k[i])
            numSuccessFaceDetect += cnt

        # In lack of detected faces, return None
        if (numSuccessFaceDetect < int(0.5 * numTotalImages)):
            print('Number of detected faces: ' + str(numSuccessFaceDetect))
            print('Not enough face detected...')
            ImageDetected = 0
        else:
            ImageDetected = 1


        return str(ImageDetected)

def test():
    a = time.time()
    dir_name = './sample/'
    k = testInput(dir_name)
    numTotalImages = np.shape(k)[1]
    print(numTotalImages)
    numSuccessFaceDetect = 0

    # for online stream processing~!~!
    for i in range(len(k)):
        cnt = testFaceCropper(k[i])
        numSuccessFaceDetect += cnt

    # In lack of detected faces, return None
    if (numSuccessFaceDetect < int(0.5 * numTotalImages)):
        print('Number of detected faces: ' + str(numSuccessFaceDetect))
        print('Not enough face detected...')
        ImageDetected = 0
    else:
        ImageDetected = 1
    print((time.time()-a))

    return str(ImageDetected)

class Server:
    def __init__(self):
        app = Flask(__name__)
        api = API()
        app.add_url_rule('/<imagePath>', view_func=api.as_view('wrapper'))
        # Run
        app.run(host='0.0.0.0', port=8002, threaded=True, debug=False)

if __name__ == '__main__':
    api = test()