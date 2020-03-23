import cv2
import imagezmq
import numpy as np
import uuid
import io
from dotenv import load_dotenv

from utils import match_faces, get_VIPs, Minio, Connection

load_dotenv()

# instanciate db and Minio
minioClient = Minio()
Conn = Connection()
Embedding, Frame, Face, VIP = Conn.get_tables()
session = Conn.get_session()

# retrieve VIPs embeddings
vip_ids, vip_embeds = get_VIPs(session, Embedding, VIP)

print('[debug] Starting Server')
image_hub = imagezmq.ImageHub()

while True:

    # receive the frame
    data, frame = image_hub.recv_image()

    # prepare frame for storage
    frame_string = cv2.imencode('.jpg', frame)[1].tostring()
    frame_id = str(uuid.uuid1())
    url_key = "camera_{}/{}.jpg".format(data["camera_id"], frame_id)
    url = "http://15.188.86.39:9001/organization1/{}".format(url_key)
    image_stream = io.BytesIO(frame_string)
    minioClient.save_frame("organization1", url_key, image_stream, len(frame_string))

    # extract embeds from data
    embeds = data["embeds"]

    # save the frame information in the database
    session.execute(Frame.insert().values(
        id=frame_id,
        camera_id=data["camera_id"],
        frame_url=url,
        timestamp=data["timestamp"]
    ))

    if len(embeds):
        embeds = np.array(embeds)
        matches = match_faces(embeds, vip_embeds, threshold=.55)

        for i, index in enumerate(matches):
            bounding_box = data["bbox"][i]
            VIP_id = 1
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

    image_hub.send_reply(b'OK')
