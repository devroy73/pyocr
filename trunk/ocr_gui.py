import wx #GUI library
import neural #Neural Network
import numpy
import pickle

OCR_TRAIN_SET = ('1a.bmp', '1b.bmp', '1c.bmp', '2a.bmp', '2b.bmp', '2c.bmp')

IMAGES_DIR = "C:\\Users\\Yan\\Programming\\ocr\\images"
IMAGE_WIDTH = 10
IMAGE_HEIGHT = 10

WHITE = '\xff\xff\xff'
BLACK = '\x00\x00\x00'

TRUE = 0.9
FALSE = 0.1

TRAINING_ITERATIONS = 1000
DIGITS = 4

def load_bitmap(filename):
    bmp = wx.Image(filename, wx.BITMAP_TYPE_BMP)
    rgb_data = bmp.GetData()
    bin_data = []
    for i in xrange(len(rgb_data)/3):
        if WHITE == rgb_data[i*3:i*3+3]:
            pixel = FALSE
        else:
            pixel = TRUE
        bin_data.append(pixel)
    assert(len(bin_data) == IMAGE_WIDTH * IMAGE_HEIGHT)
    del rgb_data
    return bin_data

app = wx.App()
data = [numpy.array(load_bitmap(IMAGES_DIR + "\\" + image)) for image in OCR_TRAIN_SET]
del app
del wx

def TrainOCR(network):
    digits = [int(image[0]) for image in OCR_TRAIN_SET]
    expected_results = []
    for i in xrange(len(OCR_TRAIN_SET)):
        expected_results.append([FALSE]*DIGITS)
        expected_results[i][digits[i]] = TRUE

    for i in xrange(TRAINING_ITERATIONS):
        error = 0.0
        for input_element, expected_result in zip(data, expected_results):
            error += network.train(input_element, expected_result)
            #print expected_result
            #print network.output_layer.outputs
            #print network.output_layer.errors
        if (error < neural.ERROR_SATISFACTION):
            break

def main():
    for datum in data:
        print datum
    #nnw = neural.BP_NeuralNetwork(IMAGE_WIDTH * IMAGE_HEIGHT, 50, DIGITS)
    #TrainOCR(nnw)
    
    #ocrnet_file = open('ocr.net', 'wb')
    #pickle.dump(nnw, ocrnet_file)
    ocrnet_file = open('ocr.net', 'rb')
    nnw = pickle.load(ocrnet_file)
    ocrnet_file.close()
    
    nnw.activate(data[0])
    print nnw.output_layer.outputs
    nnw.activate(data[1])
    print nnw.output_layer.outputs
    nnw.activate(data[3])
    print nnw.output_layer.outputs
    
if __name__ == '__main__':
    main()
