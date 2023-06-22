import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader


class Predict:
    def __init__(self):
        with open('data.yaml', mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yaml['names']

        self.yolo = cv2.dnn.readNetFromONNX('best.onnx')
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self, img_path):
        #print(img_path)
        img = cv2.imread(img_path)
        image = img.copy()
        row, col, d = image.shape

        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image

        input_width_height = 640
        blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (input_width_height, input_width_height), swapRB=True, crop=False) #Binary Large OBject
        self.yolo.setInput(blob)
        predictions = self.yolo.forward()

        detections = predictions[0]
        boxes = []
        confidences = []
        classes = []

        img_width, img_height = input_image.shape[:2]
        x_factor = img_width / input_width_height
        y_factor = img_height / input_width_height

        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]
            if confidence > 0.4:
                class_score = row[5:].max()
                class_id = row[5:].argmax()

                if class_score > 0.25:
                    cx, cy, w, h = row[0:4]
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    box = np.array([left, top, width, height])

                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()

        index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.4)

        for ind in index:
            x, y, w, h = boxes_np[ind]
            box_confidence = confidences_np[ind]
            classes_id = classes[ind]
            class_name = self.labels[classes_id]

            text = f'{class_name}: {round(box_confidence, 2)}%'
            #print(text)

            cv2.rectangle(image, (x, y-10), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(image, (x, y - 20), (x + w, y), (255, 255, 255), -1)

            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 0.7, (200, 2, 200), 1)

        image_bordered = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
        cv2.imshow('predictions', image_bordered)
        cv2.waitKey(0)



