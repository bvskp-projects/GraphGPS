import fsspec
import io
import os.path as osp
import pickle
import shutil
import torch

from torch_geometric.data import InMemoryDataset, Data

def torch_save(data, path) -> None:
    buffer = io.BytesIO()
    torch.save(data, buffer)
    with fsspec.open(path, 'wb') as f:
        f.write(buffer.getvalue())

def torch_load(path):
    with fsspec.open(path, 'rb') as f:
        return torch.load(f)

class Mirror(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root)
        path = osp.join(self.processed_dir, self.processed_file_names[0])
        self.load(path)

    def download(self):
        src = "/nobackup/vbalivada/GraphGPS/cs762/mirror.pkl"
        dst = osp.join(self.raw_dir, "mirror.pkl")
        shutil.copy(src, dst)

    @property
    def raw_file_names(self):
        return ['r2.pkl']

    @property
    def processed_file_names(self):
        return ['p2.pt']

    def process(self):
        with open(osp.join(self.raw_dir, "mirror.pkl"), "rb") as f:
            graphs = pickle.load(f)
        self.save(self.__class__, graphs, osp.join(self.processed_dir, self.processed_file_names[0]))

    @staticmethod
    def save(cls, data_list, path):
        r"""Saves a list of data objects to the file path :obj:`path`."""
        data, slices = cls.collate(data_list)
        torch_save((data.to_dict(), slices), path)
    
    def load(self, path):
        r"""Loads the dataset from the file path :obj:`path`."""
        data, self.slices = torch_load(path)
        if isinstance(data, dict):  # Backward compatibility.
            data = Data.from_dict(data)
        self.data = data
