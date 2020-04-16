import cv2
import imagezmq
from datetime import datetime
import numpy as np
import uuid
import io
from dotenv import load_dotenv
from keras_vggface import VGGFace

from utils import match_faces, get_VIPs, Minio, Connection, crop_faces

load_dotenv()

# instanciate db and Minio
minioClient = Minio()
Conn = Connection()
Embedding, Frame, Face, VIP = Conn.get_tables()
session = Conn.get_session()

# retrieve VIPs embeddings
vip_ids, vip_embeds = get_VIPs(session, Embedding, VIP)




print('[debug] Starting Server')
image_hub = imagezmq.ImageHub(open_port='tcp://192.168.1.27:5556', REQ_REP=False)

while True:

    # receive the frame
    data, frame = image_hub.recv_image()

    # prepare frame for storage
    tick = datetime.now()
    frame_string = cv2.imencode('.jpg', frame)[1].tostring()
    frame_id = str(uuid.uuid1())
    url_key = "camera_{}/{}.jpg".format(data["camera_id"], frame_id)
    url = "http://15.188.86.39:9001/organization1/{}".format(url_key)
    image_stream = io.BytesIO(frame_string)
    minioClient.save_frame("organization1", url_key, image_stream, len(frame_string))
    print("[debug] saving frame takes {}".format(datetime.now() - tick))

    # extract embeds from data
    boxes = data["bbox"]

    tack = datetime.now()
    if len(boxes):
        # load embedder
        embedder = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        faces = crop_faces(frame, boxes)
        embeds = embedder.predict(faces).tolist()
    print("[debug] facial feature extraction takes {}".format(datetime.now() - tack))

    # save the frame information in the database
    session.execute(Frame.insert().values(
        id=frame_id,
        camera_id=data["camera_id"],
        frame_url=url,
        timestamp=data["timestamp"]
    ))

    # if len(embeds):
    #     embeds = np.array(embeds)
    #     print(embeds.shape)
    #     print(vip_embeds.shape)
    #     matches = match_faces(embeds, vip_embeds, threshold=.55)
    #
    #     for i, index in enumerate(matches):
    #         bounding_box = data["bbox"][i]
    #         VIP_id = 16
    #         if not index == -1:
    #             VIP_id = vip_ids[index]
    #
    #         # save the face information in the database
    #         session.execute(Face.insert().values(
    #             frame_id=frame_id,
    #             VIP_id=VIP_id,
    #             bounding_box=bounding_box
    #         ))

    # commit db changes
    session.commit()


    # image_hub.send_reply(b'OK')
