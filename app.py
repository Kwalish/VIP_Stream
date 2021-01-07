import cv2
import imagezmq
from datetime import datetime
import numpy as np
import uuid
import io
from dotenv import load_dotenv
from keras_vggface import VGGFace
from multiprocessing import Process

from utils import match_faces, get_VIPs, Minio, Connection, crop_faces

load_dotenv()

# instanciate db
Conn = Connection()
Embedding, Frame, Face, VIP = Conn.get_tables()
session = Conn.get_session()

# retrieve VIPs embeddings
vip_ids, vip_embeds = get_VIPs(session, Embedding, VIP)

def create_image_info(camera_id):
    frame_id = str(uuid.uuid1())
    url_key = "camera_{}/{}.jpg".format(camera_id, frame_id)
    url = "http://15.188.86.39:9001/organization1/{}".format(url_key)
    return frame_id, url, url_key

def send_image_minio(frame, url_key):
    tick = datetime.now()
    minioClient = Minio()
    frame_string = cv2.imencode('.jpg', frame)[1].tostring()
    image_stream = io.BytesIO(frame_string)
    minioClient.save_frame("organization1", url_key, image_stream, len(frame_string))
    print("[debug] saving frame takes {}".format(datetime.now() - tick))



print('[debug] Starting Server')
image_hub = imagezmq.ImageHub(open_port='tcp://192.168.1.31:5555', REQ_REP=False)

embedder = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

count = 0

while True:
    tick = datetime.now()
    # receive the frame
    data, frame = image_hub.recv_image()

    # prepare frame for storage
    frame_id, url, url_key = create_image_info(data["camera_id"])
    p = Process(target=send_image_minio, args=(frame, url_key,))
    p.start()
    # p.join()
    # extract embeds from data

    # save the frame information in the database
    session.execute(Frame.insert().values(
        id=frame_id,
        camera_id=data["camera_id"],
        frame_url=url,
        timestamp=data["timestamp"]
    ))

    boxes = data["bbox"]


    if len(boxes) > 0:
        tack = datetime.now()
        # load embedder
        faces = crop_faces(frame, boxes)
        embeds = embedder.predict(faces).tolist()
        print("[debug] facial feature extraction takes {}".format(datetime.now() - tack))
        if len(embeds):
            embeds = np.array(embeds)
            matches = match_faces(embeds, vip_embeds, threshold=.55)
            print("---MATCHES---")
            print(len(matches))
            print(matches)
            print("---BBOX---")
            print(len(data["bbox"]))
            print(data["bbox"])
            for i, index in enumerate(matches):
                bounding_box = data["bbox"][i]
                VIP_id = 16
                if not index == -1:
                    VIP_id = vip_ids[index]

                # save the face information in the database
                session.execute(Face.insert().values(
                    frame_id=frame_id,
                    VIP_id=VIP_id,
                    bounding_box=bounding_box
                ))


    # commit db changes
    session.commit()
    print("[debug] total process takes {}".format(datetime.now() - tick))
    count += 1
    print("[debug] received {} images".format(count))
    # image_hub.send_reply(b'OK')

