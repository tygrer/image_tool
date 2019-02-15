import tensorflow as tf
import numpy as np
import os
import urllib.request
import argparse
import sys
import cv2
import tensorflow as tf
import numpy as np
import caffe_classes
def data_import(file_name):

class net():
    # we usually don't do convolution and pooling on batch and channel
    def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding="SAME"):
        """max-pooling"""
        return tf.nn.max_pool(x, ksize=[1, kHeight, kWidth, 1],
                              strides=[1, strideX, strideY, 1], padding=padding, name=name)

    def dropout(x, keepPro, name=None):
        """dropout"""
        return tf.nn.dropout(x, keepPro, name)

    def LRN(x, R, alpha, beta, name=None, bias=1.0):
        """LRN"""
        return tf.nn.local_response_normalization(x, depth_radius=R, alpha=alpha,
                                                  beta=beta, bias=bias, name=name)

    def fcLayer(x, inputD, outputD, reluFlag, name):
        """fully-connect"""
        with tf.variable_scope(name) as scope:
            w = tf.get_variable("w", shape=[inputD, outputD], dtype="float")
            b = tf.get_variable("b", [outputD], dtype="float")
            out = tf.nn.xw_plus_b(x, w, b, name=scope.name)
            if reluFlag:
                return tf.nn.relu(out)
            else:
                return out

    def convLayer(x, kHeight, kWidth, strideX, strideY,
                  featureNum, name, padding="SAME", groups=1):
        """convolution"""
        channel = int(x.get_shape()[-1])
        conv = lambda a, b: tf.nn.conv2d(a, b, strides=[1, strideY, strideX, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            w = tf.get_variable("w", shape=[kHeight, kWidth, channel / groups, featureNum])
            b = tf.get_variable("b", shape=[featureNum])

            xNew = tf.split(value=x, num_or_size_splits=groups, axis=3)
            wNew = tf.split(value=w, num_or_size_splits=groups, axis=3)

            featureMap = [conv(t1, t2) for t1, t2 in zip(xNew, wNew)]
            mergeFeatureMap = tf.concat(axis=3, values=featureMap)
            # print mergeFeatureMap.shape
            out = tf.nn.bias_add(mergeFeatureMap, b)
            return tf.nn.relu(tf.reshape(out, mergeFeatureMap.get_shape().as_list()), name=scope.name)

class alexNet(object):
    """alexNet model"""

    def __init__(self, x, keepPro, classNum, skip, modelPath="bvlc_alexnet.npy"):
        self.X = x
        self.KEEPPRO = keepPro
        self.CLASSNUM = classNum
        self.SKIP = skip
        self.MODELPATH = modelPath
        # build CNN
        self.buildCNN()

    def buildCNN(self):
        """build model"""
        alexnet = net()
        conv1 = alexnet.convLayer(self.X, 11, 11, 4, 4, 96, "conv1", "VALID")
        lrn1 = alexnet.LRN(conv1, 2, 2e-05, 0.75, "norm1")
        pool1 = alexnet.maxPoolLayer(lrn1, 3, 3, 2, 2, "pool1", "VALID")

        conv2 = alexnet.convLayer(pool1, 5, 5, 1, 1, 256, "conv2", groups=2)
        lrn2 = alexnet.LRN(conv2, 2, 2e-05, 0.75, "lrn2")
        pool2 = alexnet.maxPoolLayer(lrn2, 3, 3, 2, 2, "pool2", "VALID")

        conv3 = alexnet.convLayer(pool2, 3, 3, 1, 1, 384, "conv3")

        conv4 = alexnet.convLayer(conv3, 3, 3, 1, 1, 384, "conv4", groups=2)

        conv5 = alexnet.convLayer(conv4, 3, 3, 1, 1, 256, "conv5", groups=2)
        pool5 = alexnet.maxPoolLayer(conv5, 3, 3, 2, 2, "pool5", "VALID")

        fcIn = tf.reshape(pool5, [-1, 256 * 6 * 6])
        fc1 = alexnet.fcLayer(fcIn, 256 * 6 * 6, 4096, True, "fc6")
        dropout1 = alexnet.dropout(fc1, self.KEEPPRO)

        fc2 = alexnet.fcLayer(dropout1, 4096, 4096, True, "fc7")
        dropout2 = alexnet.dropout(fc2, self.KEEPPRO)

        self.fc3 = alexnet.fcLayer(dropout2, 4096, self.CLASSNUM, True, "fc8")

    def loadModel(self, sess):
        """load model"""
        wDict = np.load(self.MODELPATH, encoding="bytes").item()
        # for layers in model
        for name in wDict:
            if name not in self.SKIP:
                with tf.variable_scope(name, reuse=True):
                    for p in wDict[name]:
                        if len(p.shape) == 1:
                            # bias
                            sess.run(tf.get_variable('b', trainable=False).assign(p))
                        else:
                            # weights
                            sess.run(tf.get_variable('w', trainable=False).assign(p))

def config():
    pass

def main():
    parser = argparse.ArgumentParser(description='Classify some images.')
    parser.add_argument('-m', '--mode', choices=['folder', 'url'], default='folder')
    parser.add_argument('-p', '--path', help='Specify a path [e.g. testModel]', default='testModel')
    args = parser.parse_args(sys.argv[1:])

    if args.mode == 'folder':
        # get testImage
        withPath = lambda f: '{}/{}'.format(args.path, f)
        testImg = dict((f, cv2.imread(withPath(f))) for f in os.listdir(args.path) if os.path.isfile(withPath(f)))
    elif args.mode == 'url':
        def url2img(url):
            '''url to image'''
            resp = urllib.request.urlopen(url)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            return image

        testImg = {args.path: url2img(args.path)}

    # noinspection PyUnboundLocalVariable
    if testImg.values():
        # some params
        dropoutPro = 1
        classNum = 1000
        skip = []

        imgMean = np.array([104, 117, 124], np.float)
        x = tf.placeholder("float", [1, 227, 227, 3])

        model = alexNet(x, dropoutPro, classNum, skip)
        score = model.fc3
        softmax = tf.nn.softmax(score)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model.loadModel(sess)

            for key, img in testImg.items():
                # img preprocess
                resized = cv2.resize(img.astype(np.float), (227, 227)) - imgMean
                maxx = np.argmax(sess.run(softmax, feed_dict={x: resized.reshape((1, 227, 227, 3))}))
                res = caffe_classes.class_names[maxx]

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, res, (int(img.shape[0] / 3), int(img.shape[1] / 3)), font, 1, (0, 255, 0), 2)
                print("{}: {}\n----".format(key, res))
                cv2.imshow("demo", img)
                cv2.waitKey(0)