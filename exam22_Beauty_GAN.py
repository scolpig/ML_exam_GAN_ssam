import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

detector = dlib.get_frontal_face_detector()
shape = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')
#
# img = dlib.load_rgb_image('./imgs/09.jpg')
# plt.figure(figsize=(16, 10))
# plt.imshow(img)
# plt.show()
#
# img_result = img.copy()
# dets = detector(img, 1)
#
# if len(dets) == 0:
#     print('Not find faces')
#
# else:
#     fig, ax = plt.subplots(1, figsize=(10, 16))
#     for det in dets:
#         x, y, w, h = det.left(), det.top(), det.width(), det.height()
#         rect = patches.Rectangle((x, y), w, h,
#                  linewidth=2, edgecolor='b', facecolor='None')
#         ax.add_patch(rect)
# ax.imshow(img_result)
# plt.show()
#
# fig, ax = plt.subplots(1, figsize=(16, 10))
# obj = dlib.full_object_detections()
#
# for detection in dets:
#     s = shape(img, detection)
#     obj.append(s)
#
#     for point in s.parts():
#         circle = patches.Circle((point.x, point.y),
#                     radius=3, edgecolor='b', facecolor='b')
#         ax.add_patch(circle)
#     ax.imshow(img_result)
# plt.show()

def align_faces(img):
    dets = detector(img, 1)
    objs = dlib.full_object_detections()
    for detection in dets:
        s = shape(img, detection)
        objs.append(s)
    faces = dlib.get_face_chips(img, objs,
                size=256, padding=0.35)
    return faces
test_img = dlib.load_rgb_image('./imgs/02.jpg')
test_faces = align_faces(test_img)
fig, axes = plt.subplots(1, len(test_faces)+1, figsize=(10, 8))
axes[0].imshow(test_img)
for i, face in enumerate(test_faces):
    axes[i + 1].imshow(face)
plt.show()









