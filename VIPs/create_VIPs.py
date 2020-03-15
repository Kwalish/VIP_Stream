from sqlalchemy import create_engine
from models import VIP
from glob import glob
from keras_vggface import VGGFace
from mtcnn import MTCNN
import cv2
import os
import numpy as np
import uuid
from sqlalchemy.orm import sessionmaker

mtcnn = MTCNN()
embedder = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

db_string = "postgresql://postgres:142336@localhost:5432/VIPJETSON"
db = create_engine(db_string)

Session = sessionmaker(db)
session = Session()

def crop(image, box, padding=None):
    height, width = image.shape[:2]
    x0, y0, w, h = box
    x1, y1 = x0 + w, y0 + h
    if padding:
        size = max(x1 - x0, y1 - y0)
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        x0, y0 = xm - size / 2 - 5, ym - size / 2 - 5
        x1, y1 = xm + size / 2 + 5, ym + size / 2 + 5
    x0, y0, x1, y1 = list(map(int, [x0, y0, x1, y1]))
    return image[max(0, y0): min(y1, height), max(0, x0): min(x1, width)]


def detect_faces(image, threshold=.95):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = mtcnn.detect_faces(image)
    candidates = filter(lambda d: d['confidence'] > threshold, detections)
    boxes = map(lambda c: c['box'], candidates)
    return list(boxes)

def get_vips():
    images = glob('./*.jpg')
    vips = []
    names = []
    for i, path in enumerate(images):
        vip_name, _ = os.path.splitext(os.path.basename(path))
        names.append(vip_name)
        img = cv2.imread(path)

        boxes = mtcnn.detect_faces(img)
        assert len(boxes) == 1
        box = boxes[0]['box']
        face_image = crop(img, box, padding=True)
        face_image = cv2.resize(face_image, (224, 224))
        vips.append(face_image)
    vips = np.array(vips)
    embeddings = embedder.predict(vips)
    for name, embed in zip(names, embeddings):
        vip_record = VIP(name=name, id=str(uuid.uuid1()), embed=embed.tolist())
        session.add(vip_record)
        session.commit()




print(get_vips())



