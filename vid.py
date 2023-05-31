from ultralytics import YOLO
import cv2
import random
import numpy as np
import pyshine as ps


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)



# Load a model
model = YOLO("yolov8m.pt")
class_names = model.names
print('Class Names: ', class_names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]
cap = cv2.VideoCapture(2)

while True:
    success, img = cap.read()
    if not success:
        break

    results = model.predict(img)
    # print('results: ', results)

    for result in results:
        bboxs = result.boxes.xyxy
        conf = result.boxes.conf
        cls = result.boxes.cls
        for bbox, cnf, cs in zip(bboxs, conf, cls):
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2])
            ymax = int(bbox[3])

            # print(xmin, ymin, xmax, ymax, float(cnf), int(cs))
            plot_one_box(
                [xmin, ymin, xmax, ymax], img,
                colors[int(cs)], f'{class_names[int(cs)]} {float(cnf):.3}',
                3
            )
    k = cv2.waitKey(1)
    cv2.imshow('img', img)
    if k == ord('q'):
        cv2.destroyAllWindows()
        break