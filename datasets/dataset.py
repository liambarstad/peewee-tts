from torch.utils.data import Dataset as DS
from .sources import AWSCloudSource, LocalDirectorySource

class Dataset(DS):
    def __init__(self,
                 source: str,
                 root_dir='/'
                 ):
        self.root_dir = root_dir
        if source == 'aws_cloud':
            self.source = AWSCloudSource(self.root_dir)
        elif source == 'local_directory':
            self.source = LocalDirectorySource(self.root_dir)
