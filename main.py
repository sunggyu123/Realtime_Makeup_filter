# -*- coding: utf-8 -*-

import os
import time
import argparse

import cv2
import dlib
import numpy as np
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--makeup', type=str, default=os.path.join(os.getcwd(), 'imgs', 'makeup', '-4.jpg'), help='path to the makeup image')
args = parser.parse_args()

detector = dlib.get_frontal_face_detector()
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor(os.path.join(os.getcwd(), 'models', 'shape_predictor_68_face_landmarks', 'shape_predictor_68_face_landmarks.dat'))

face_proto = './models/opencv_face_detector.pbtxt'
face_model = './models/opencv_face_detector_uint8.pb'
face_net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)

def bitOperation(hpos, vpos, hpos2, vpos2, img1, img2):
    print(hpos, vpos, hpos2, vpos2)
    img2 = cv2.resize(img2, (hpos2 - hpos, vpos2 - vpos))
    roi = img1[vpos:vpos2, hpos:hpos2]
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2_gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(roi, img2, mask=mask_inv)
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    dst = cv2.add(img1_bg, img2_fg)
    img1[vpos:vpos2, hpos:hpos2] = dst

    cv2.imshow("result", img1)

    return img1

def preprocess(img):
    return img.astype(np.float32) / 127.5 - 1.

def deprocess(img):
    return ((img + 1.) * 127.5).astype(np.uint8)

def align_face(img):
    dets = detector(img, 1)
    objs = dlib.full_object_detections()
    for detection in dets:
        s = predictor(img, detection)
        objs.append(s)

    faces = dlib.get_face_chips(img, objs, size=256, padding=0.35)

    return faces

# def face_filter(img, rect):
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     objs = dlib.full_object_detections()
#     (x, y, w, h) = rect.astype('int')
#     s = predictor(img, dlib.rectangle(x, y, w, h))
#     objs.append(s)
#
#     faces = dlib.get_face_chips(img, objs, size=256, padding=0.35)
#
#     X_img = preprocess(faces[0])
#     X_img = np.expand_dims(X_img, axis=0)
#     with tf.device('/gpu:0'):
#         Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
#     Xs_ = deprocess(Xs_[0])
#     # output = np.uint8(np.squeeze(Xs_))
#     output = cv2.cvtColor(Xs_, cv2.COLOR_BGR2RGB)
#
#     return output

# def faces_filter(number, img, net):
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     objs = [dlib.full_object_detections() for _ in range(number)]
#     for cnt in range(len(objs)):
#         for i in range(0, net.shape[2]):
#             box = net[0, 0, i, 3:7] * np.array([256, 256, 256, 256])
#             (x, y, w, h) = box.astype('int')
#             s = predictor(img[cnt], dlib.rectangle(x, y, w, h))
#             objs[cnt].append(s)
#
#     # faces = [dlib.get_face_chips(img[cnt], objs[cnt], size=256, padding=0.35) for cnt in range(len(objs))]
#     # print("faces : ", str(len(faces)))
#
#     outputs = []
#     for cnt in range(len(objs)):
#         X_img = preprocess(img[cnt])
#         X_img = np.expand_dims(X_img, axis=0)
#         with tf.device('/gpu:0'):
#             Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
#         Xs_ = deprocess(Xs_[0])
#         # output = np.uint8(np.squeeze(Xs_))
#         output = cv2.cvtColor(Xs_, cv2.COLOR_BGR2RGB)
#         outputs.append(output)
#     # print("outputs:", str(len(outputs)))
#
#     final_output = cv2.hconcat(outputs)
#
#     return final_output

def faces_filter(number, img, rects):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    objs = [dlib.full_object_detections() for _ in range(number)]
    for cnt in range(len(objs)):
        rect = rects[cnt]
        (x, y, w, h) = rect.astype('int')
        s = predictor(img[cnt], dlib.rectangle(x, y, w, h))
        objs[cnt].append(s)

    outputs = []
    for cnt in range(len(objs)):
        X_img = preprocess(img[cnt])
        X_img = np.expand_dims(X_img, axis=0)
        with tf.device('/gpu:0'):
            Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
        Xs_ = deprocess(Xs_[0])
        # output = np.uint8(np.squeeze(Xs_))
        output = cv2.cvtColor(Xs_, cv2.COLOR_BGR2RGB)
        outputs.append(output)
    # print("outputs:", str(len(outputs)))

    return outputs

batch_size = 1
img_size = 256
minimum_confidence = 0.5

makeups = args.makeup
makeup = dlib.load_rgb_image(makeups)
makeup = align_face(makeup)
makeup = preprocess(makeup[0])
Y_img = np.expand_dims(makeup, axis=0)

tf.reset_default_graph()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())

saver = tf.train.import_meta_graph(os.path.join(os.getcwd(), 'models', 'model.meta'))
saver.restore(sess, tf.train.latest_checkpoint('models'))

graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')
print('model load')

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc('W', 'M', 'V', '1')
videoWriter = cv2.VideoWriter(os.path.join(os.getcwd(), 'output', 'video1.wmv'), fourcc, 10.0, (640, 480))
num = 0
while True:
    # Get frame from video
    # get success : ret = True / fail : ret= False
    ret, frame = cap.read()
    (H, W) = frame.shape[:2]
    frame = cv2.flip(frame, 1)
    face_rec = frame.copy()
    face_test = frame.copy()
    face_test = cv2.cvtColor(face_test, cv2.COLOR_BGR2RGB)

    blob = cv2.dnn.blobFromImage(face_rec, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    dets = face_net.forward()
    number = 0
    transfered_image = None
    transfered_images = None
    test = None
    rect = None
    rects = []
    face_imgs = []
    face_mark = []
    for i in range(0, dets.shape[2]):
        confidence = dets[0, 0, i, 2]

        # 얼굴 인식 확률이 최소 확률보다 큰 경우
        if confidence > minimum_confidence:
            # bounding box 위치 계산
            box = dets[0, 0, i, 3:7] * np.array([W, H, W, H])
            rect = dets[0, 0, i, 3:7] * np.array([img_size, img_size, img_size, img_size])
            rects.append(rect)
            (startX, startY, endX, endY) = box.astype("int")

            # bounding box 가 전체 좌표 내에 있는지 확인
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(W - 1, endX), min(H - 1, endY))

            print(startX, startY, endX, endY)

            face_mark.append((startX, startY, endX, endY))
            test = face_test[startY:endY, startX:endX]
            # print(endY-startY, endX-startX)
            test = cv2.resize(test, (img_size, img_size), cv2.INTER_CUBIC)
            face_imgs.append(test)

            cv2.putText(face_rec, "Face[{}]".format(number + 1), (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)  # 얼굴 번호 출력
            cv2.rectangle(face_rec, (startX, startY), (endX, endY), (0, 255, 0), 2)  # bounding box 출력

            number = number + 1  # 얼굴 번호 증가
            # print(number)

    transfered_images = faces_filter(number, face_imgs, rects)


    sigma = 3
    gaus = cv2.GaussianBlur(face_rec, (0, 0), sigma)

    img1 = None
    for i in range(len(transfered_images)):
        x, y, w, h = face_mark[i]
        img1 = bitOperation(x, y, w, h, face_rec, transfered_images[i])

    videoWriter.write(face_rec)

    # wait for keyboard input
    key = cv2.waitKey(1)
    # if esc
    if key == 27:
        break

cap.release()
videoWriter.release()
cv2.destroyAllWindows()