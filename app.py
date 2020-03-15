import boto3
import cv2
import imagezmq
from keras_vggface import VGGFace
from mtcnn import MTCNN
from glob import glob
import numpy as np
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import VIP, Frame
import uuid
import io

from minio import Minio

minioClient = Minio('15.188.86.39:9001',
                    access_key='gradients',
                    secret_key='1qMf6TVF46ONg22V',
                    secure=False)

db_string = "postgresql://postgres:142336@localhost:5432/VIPJETSON"

db = create_engine(db_string)

Session = sessionmaker(db)
session = Session()

vip_records = session.query(VIP)

vip_names = []
vips = []
for vip in vip_records:
    vip_names.append(vip.name)
    vips.append(vip.embed)


    # print(vip.name)

vips = np.array(vips)

print(vip_names)
print(vips.shape)


def match_faces(src, target, threshold=.90):
    def cos(a, b):
        x = np.sqrt(np.sum(np.square(a), axis=1, keepdims=True))
        y = np.sqrt(np.sum(np.square(b), axis=1, keepdims=True))
        return np.dot(a, b.T) / np.dot(x, y.T)

    sims = cos(src, target)
    print(sims)
    candidates = np.argmax(sims, axis=1)
    scores = np.max(sims, axis=1)
    filter = scores > threshold
    indices = np.where(filter, candidates, -1)
    return indices


image_hub = imagezmq.ImageHub()

print('[debug] Starting Server')

while True:
    data, image = image_hub.recv_image()
    print(data)
    image_string = cv2.imencode('.jpg', image)[1].tostring()
    frame_id = str(uuid.uuid1())
    url_key = "{}_{}.jpg".format(data["camera_id"], frame_id)
    url = "http://15.188.86.39:9001/assprototype/{}".format(url_key)
    image_stream = io.BytesIO(image_string)
    minioClient.put_object(bucket_name="assprototype", object_name=url_key, data=image_stream, length=len(image_string))


    # temporary
    embeds = data["embeds"]
    bboxes = []
    names = []
    if len(embeds):
        embeds = np.array(embeds)
        print(embeds.shape)
        matches = match_faces(embeds, vips, threshold=.55)

        for i, index in enumerate(matches):
            bboxes.append(data["bbox"][i])
            if index == -1:
                names.append("unknown")
            else:
                names.append(vip_names[index])

    frame_record = Frame(camera_id=data["camera_id"], frame_url=url, bboxes=bboxes, identities=names,
                         timestamp=data["timestamp"],
                         id=frame_id)
    session.add(frame_record)
    session.commit()

    image_hub.send_reply(b'OK')
#
