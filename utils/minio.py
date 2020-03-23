from minio import Minio
import os

minioClient = Minio(os.environ['MINIO_IP'],
                    access_key=os.environ['MINIO_ACCESS_KEY'],
                    secret_key=os.environ['MINIO_SECRET_KEY'],
                    secure=False)


class Minio:
    def __init__(self):
        self.client = minioClient

    def save_frame(self, bucket_name, object_name, data, length):
        self.client.put_object(bucket_name=bucket_name,
                               object_name=object_name,
                               data=data,
                               length=length)
