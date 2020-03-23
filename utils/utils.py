import numpy as np


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
        vip_embeds.append(vip.embedding)

    vip_embeds = np.array(vip_embeds)
    return vip_ids, vip_embeds
