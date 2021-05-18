from torch.nn.functional import embedding
from data import dataloader
import torch

import torchvision.ops as ops
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import os
import yaml
import shutil
import pickle
import subprocess
import numpy as np
from PIL import Image
import Res2Net.res2net as res2net
from models.heads import TwoLayerLinearHead


id_list = ["6045d6aecbb67023b0e62922",
"604081897b6123519b1e7ec1",
"6007f1751d6abc10d25364dc",
"6007ef768e3ed72df90308c7",
"5ff53f9e279d05bcbf241594",
"5fe86f7f82320da589e77095",
"5fc397e9e731f5bc070de252",
"5fa7715fb61f981d38da27ec",
"5fd9952fcca9e11024cf9860",
"5f905449dd5ee0a18c909a17",
"5f8d83a6c3e03fb94a40676b"]

def download_dataset(id_):
    param = {"prj_id":id_,
            "prj_name":"patrick",
            "prj_type":"bbox",
            "phase":"inhouse_prod",
            "dataset":"/",
            "output":"dataset/"}

    download_image = ["eimmo-inhouse",
            "download-files",
            "-p",
            param["phase"],
            "-pid",
            param["prj_id"],
            "--path",
            param["dataset"],
            "-o",
            param["output"],
            "-k",
            "down_quality"]
    subprocess.run(download_image)

class Backbone(torch.nn.Module):
    def __init__(self, name, proj_head_kwargs, scrl_kwargs, scrl_enabled=False):
        super(Backbone, self).__init__()
        
        if name == 'res2net':
            network = res2net.res2net50(pretrained=True)
        else:
            network = eval(f"models.{name}")()
        self.encoder = torch.nn.Sequential(*list(network.children())[:-1])

        roi_out_size = (scrl_kwargs['pool_size'], ) * 2
        self.roi_align = ops.RoIAlign(output_size=roi_out_size,
            sampling_ratio=scrl_kwargs['sampling_ratio'],
            spatial_scale=scrl_kwargs['spatial_scale'],
            aligned=scrl_kwargs['detectron_aligned'])
        self.projector = TwoLayerLinearHead(**proj_head_kwargs)
        self.scrl_enabled = scrl_enabled
    def forward(self, x, boxes=None):
        for n, layer in enumerate(self.encoder):
            x = layer(x)
            if n == len(self.encoder) - 2:
                h_pre_gap = x
        h = x.squeeze()
        if self.scrl_enabled:
            assert boxes is not None
            roi_h = self.roi_align(h_pre_gap, boxes).squeeze()
            roi_p = self.projector(roi_h)
            return roi_p, h
        else:
            p = self.projector(h)
            return p, h

class CustomDataset(Dataset):
    def __init__(self, data_root: str):
        self.data_root = data_root
        self.transform = transforms.Compose([transforms.Resize((224,224)),
        transforms.ToTensor()])
        self.image_list = list()

        for path,subdirs,files in os.walk(self.data_root):
            for name in files:
                if ('.jpg' in name) or ('.png' in name):
                    self.image_list.append(os.path.join(path,name))

    def __len__(self):
        return len(self.image_list)
    def __getitem__(self,idx):
        img = Image.open(self.image_list[idx])
        self.img = self.transform(img)
        return self.img, self.image_list[idx]
def main(id_):
    config = yaml.load(open('config/scrl_200ep.yaml'),Loader=yaml.FullLoader)
    scrl_kwargs = config['network']['scrl']
    proj_head_kwargs = config['network']['proj_head']
    model = Backbone(name='resnet50',proj_head_kwargs=proj_head_kwargs,scrl_kwargs=scrl_kwargs)
    try:
        state_dict = torch.load(f'runs/{id_}/checkpoint_30.pth')
    except:
        state_dict = torch.load(f'runs/{id_}/checkpoint_20.pth')
    model.load_state_dict(state_dict['online_network_state_dict'])
    model.eval()
    model.cuda()

    download_dataset(id_)
    dataset = CustomDataset('dataset')
    dataloader = DataLoader(dataset=dataset, batch_size=32)

    embedding = np.empty((1,256))
    file_names = []
    for i,batch in enumerate(dataloader):
        data = batch[0].cuda()
        file_name = batch[1]
        y = model(data)
        embedding = np.append(embedding,y[0].detach().cpu().numpy(),axis=0)
        file_names+=file_name
    embedding = np.delete(embedding,[0]*256,axis=0)
    pickle.dump((embedding,file_names),open(f'embeddings/{id_}_embedding.txt','wb'))
    shutil.rmtree('dataset')

if __name__ == '__main__':
    for id_ in id_list:
        main(id_)
