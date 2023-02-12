import os
import boto3
import tarfile

s3 = boto3.client('s3')

tf = tarfile.open('data/utterance_corpuses/train-other-500.tar.gz', 'r:gz')
for path in tf:
    try:
        file = tf.extractfile(path)
        if file:
            filename = path.name.split('./')[1]
            print(filename + ' uploaded')
            s3.upload_fileobj(
                file, 
                os.environ['BUCKET_NAME'],
                filename
            )
    except e:
        print(e)
        continue

