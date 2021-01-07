import numpy as np
import cv2


def match_faces(src, target, threshold=.90):
    def cos(a, b):
        x = np.sqrt(np.sum(np.square(a), axis=1, keepdims=True))
        y = np.sqrt(np.sum(np.square(b), axis=1, keepdims=True))
        return np.dot(a, b.T) / np.dot(x, y.T)

    sims = cos(src, target)
    candidates = np.argmax(sims, axis=1)
    scores = np.max(sims, axis=1)
    filter = scores > threshold
    indices = np.where(filter, candidates, -1)
    return indices


def get_VIPs(session, embedding_model, vip_model):
    vip_records = session.query(embedding_model, vip_model).join(vip_model)
    vip_ids = []
    vip_embeds = []
    for vip in vip_records:
        vip_ids.append(vip.id)
        vip_embeds.append(np.array(vip.embedding).flatten())

    print(vip_ids)
    vip_embeds = np.array(vip_embeds)
    return vip_ids, vip_embeds

def crop(image, box, padding=5):
    height, width = image.shape[:2]
    x0, y0, w, h = box
    x1, y1 = x0 + w, y0 + h
    if padding:
        size = max(x1 - x0, y1 - y0)
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        x0, y0 = xm - size / 2 - padding, ym - size / 2 - padding
        x1, y1 = xm + size / 2 + padding, ym + size / 2 + padding
    x0, y0, x1, y1 = list(map(int, [x0, y0, x1, y1]))
    return image[max(0, y0): min(y1, height), max(0, x0): min(x1, width)]

def crop_faces(image, boxes, target_size=(224, 224)):
    faces = np.zeros(shape=[len(boxes), *target_size, image.shape[-1]])
    for i, box in enumerate(boxes):
        face = crop(image, box, padding=True)
        faces[i] = cv2.resize(face, target_size)
    return faces
