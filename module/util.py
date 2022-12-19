''' Modified from https://github.com/alinlab/LfF/blob/master/module/util.py '''

import torch.nn as nn
from module.mlp import *
from torchvision.models import resnet18

def get_model(model_tag, num_classes):

    if model_tag == "ResNet18":
        if num_classes==6:
            print('bringing pretrained resnet18 for bar ...')
            model = resnet18(pretrained=True)
        else:
            print('bringing no pretrained resnet18 ...')
            model = resnet18(pretrained=False)
        model.fc = nn.Linear(512, num_classes)
        return model
    elif model_tag == "MLP":
        return MLP(num_classes=num_classes)
    elif model_tag == "mlp_DISENTANGLE":
        return MLP_DISENTANGLE(num_classes=num_classes)
    elif model_tag == 'resnet_DISENTANGLE':
        if num_classes==6:
            print('bringing pretrained resnet18 disentangle ...')
            model = resnet18(pretrained=True)
        else:
            print('bringing no pretrained resnet18 disentangle...')
            model = resnet18(pretrained=False)
        model.fc = nn.Linear(1024, num_classes)
        return model
    else:
        raise NotImplementedError

def get_backbone(model_key, num_classes, pretrained=False, first_stage=False, args=None):
    if model_key == 'MLP':
        model = MLP(num_classes=num_classes)
    elif model_key == 'ResNet18':
        print(f'Resnet18 pretrained {pretrained} loaded...')
        model = resnet18(pretrained=pretrained)
        feature_dim = 512
        if args.train_disent_be and first_stage == False:
            feature_dim = 1024
    if 'ResNet' in model_key:
        model.fc = nn.Linear(feature_dim, num_classes)
    return model