import torch
import json
import logging
import os 
import random
import numpy as np

import MinkowskiEngine as ME


from torch.utils.data import Dataset, DataLoader

# import open3d
from utils.data import random_rotate_z, normalize_pc, augment_pc

class ObjaverseLVIS(Dataset):
    def __init__(self, config):
        self.split = json.load(open(config.objaverse_lvis.split, "r"))
        self.y_up = config.objaverse_lvis.y_up
        self.num_points = config.objaverse_lvis.num_points
        self.use_color = config.objaverse_lvis.use_color
        self.normalize = config.objaverse_lvis.normalize
        self.categories = sorted(np.unique([data['category'] for data in self.split]))
        if config.clip_embed_version == "OpenCLIP":
            self.clip_cat_feat = np.load(config.objaverse_lvis.clip_feat_path, allow_pickle=True)
            # self.clip_cat_feat = np.concatenate(self.clip_cat_feat, axis=0)
            self.category2idx = {self.categories[i]: i for i in range(len(self.categories))}
        else:
            clip_feat = np.load(config.objaverse_lvis.clip_feat_path, allow_pickle=True).item()
            self.category2idx = {}
            self.clip_cat_feat = []
            for i, category in enumerate(self.categories):
                self.category2idx[category] = i
                self.clip_cat_feat.append(clip_feat[category]["prompt_avg"])
            self.clip_cat_feat = np.concatenate(self.clip_cat_feat, axis=0)

        logging.info("ObjaverseLVIS: %d samples" % (len(self.split)))
        logging.info("----clip feature shape: %s" % str(self.clip_cat_feat.shape))

    def __getitem__(self, index: int):
        data_path = self.split[index]['data_path']
        data_path = data_path.replace("/mnt/data", 'data')
        data = np.load(data_path, allow_pickle=True).item()
        n = data['xyz'].shape[0]
        # if n != self.num_points:
        # idx = random.sample(range(n), self.num_points)
        xyz = data['xyz'][: self.num_points]
        rgb = data['rgb'][: self.num_points]

        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.use_color:
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz

        assert not np.isnan(xyz).any()

        # idx = np.random.randint(data['image_feat'].shape[0])
        # img_feat = data["image_feat"][idx]

        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "group": self.split[index]['group'],
            "name": self.split[index]['uid'],
            "category": self.category2idx[self.split[index]["category"]],
            # "image_feat": torch.from_numpy(img_feat).type(torch.float32)
        }

    def __len__(self):
        return len(self.split)


def minkowski_objaverse_lvis_collate_fn(list_data):
    return {
        "xyz": ME.utils.batched_coordinates([data["xyz"] for data in list_data], dtype=torch.float32),
        "features": torch.cat([data["features"] for data in list_data], dim=0),
        "xyz_dense": torch.stack([data["xyz"] for data in list_data]).float(),
        "features_dense": torch.stack([data["features"] for data in list_data]),
        "group": [data["group"] for data in list_data],
        "name": [data["name"] for data in list_data],
        "category": torch.tensor([data["category"] for data in list_data], dtype=torch.int32),
        # "image_feat": torch.stack([data['image_feat'] for data in list_data])
    }


def make_objaverse_lvis(config):
    dataset = ObjaverseLVIS(config)
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    return DataLoader(
        ObjaverseLVIS(config), \
        num_workers=config.objaverse_lvis.num_workers, \
        collate_fn=minkowski_objaverse_lvis_collate_fn, \
        batch_size=config.objaverse_lvis.batch_size, \
        pin_memory=True, \
        shuffle=False, 
        
        # sampler=sampler
    )