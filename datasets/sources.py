import io
import os
import boto3

class Source:
    def load(self, path):
        # get io.BytesIO stream for single file in directory
        # path is in form /some/dir/here
        pass
    
    def member_paths(self, path='/'): 
        # get all paths in directory
        # these can be files or directories
        pass

class AWSCloudSource(Source):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.s3 = boto3.client('s3')

    def load(self, path):
        obj = self.s3.get_object(Bucket=os.environ['BUCKET_NAME'], Key=os.path.join(self.root_dir, path)[1:])
        return io.BytesIO(obj['Body'].read())
        
    def member_paths(self, path='/'):
        paginator = self.s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=os.environ['BUCKET_NAME'], Prefix=os.path.join(self.root_dir, path)[1:])
        return [ f['Key'] for page in pages for f in page['Contents']]

class LocalDirectorySource(Source):
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def load(self, path):
        file = open(path, 'rb')
        out = io.BytesIO(file.read())
        file.close()
        return out

    def member_paths(self, path='/'):
        member_dir = self.root_dir+path
        data = []
        for root, _, files in os.walk(member_dir):
            for file in files:
                data.append(os.path.join(root, file))
        return data

