#%%
import random
import numpy as np

import torch
import torchvision.models as models
import torch.nn as nn

from fc_fp_dataset import FCFPDataset

num_folds = 5
use_cpu = False

num_epochs = 10000
batch_size = 16
num_workers = 0
num_synth = 0

use_gpu = torch.cuda.is_available() and not use_cpu
device = 'cuda' if use_gpu else 'cpu'

base_path = '/home/finperprint-analysis/datasets/fc'

criterion = nn.CrossEntropyLoss()

fp_ids = list(range(2000))
random.shuffle(fp_ids)
splits = np.array(np.array_split(fp_ids, num_folds))

for fold in range(num_folds):
    # weights = models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1
    # model = models.vit_l_16(weights=weights)
    # model.heads.head = nn.Linear(in_features=1024, out_features=5, bias=True)
    # for param in model.encoder.parameters():
    #     param.requires_grad = False

    model = models.resnet18(weights='IMAGENET1K_V1')
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                            padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)

    # model = models.vgg11_bn(weights=models.VGG11_BN_Weights.DEFAULT)
    # first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)]
    # first_conv_layer.extend(list(model.features))
    # model.features = nn.Sequential(*first_conv_layer)
    # classifier_wo_last_layer = list(model.classifier.children())[0:-1]
    # new_classifier = classifier_wo_last_layer + [nn.Linear(4096, 5)]
    # model.classifier = nn.Sequential(*new_classifier)

    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    fp_ids_val = splits[fold]
    fc_fp_ds_val = FCFPDataset(base_path, fp_ids_val)
    # fc_fp_ds_val.set_resize()
    fc_fp_dl_val = torch.utils.data.DataLoader(fc_fp_ds_val,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=use_gpu)

    fp_ids_tra = np.append(splits[:fold], splits[fold+1:])
    fc_fp_ds_tra = FCFPDataset(base_path, fp_ids_tra)
    # fc_fp_ds_tra.set_hflip()
    fc_fp_ds_tra.set_rotate()
    # fc_fp_ds_tra.set_resize()
    fc_fp_dl_tra = torch.utils.data.DataLoader(fc_fp_ds_tra,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=use_gpu)
    for e in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        for xi, yi, fp_class_2, gender, index in fc_fp_dl_tra:
            x = xi.to(device)
            y = yi.to(device)
            optimizer.zero_grad()
            y_out = model(x)
            loss = criterion(y_out, y)
            _, labels = torch.max(y, 1)
            _, preds = torch.max(y_out, 1)
            total_correct += torch.sum(preds == labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(e, total_loss / len(fc_fp_dl_tra),
              total_correct/len(fc_fp_ds_tra))

        # Perform validation every 10 epochs
        if e % 10 == 0:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            with torch.no_grad():
                for xi, yi, fp_class_2, gender, index in fc_fp_dl_val:
                    x = xi.to(device)
                    y = yi.to(device)
                    y_out = model(x)
                    loss = criterion(y_out, y)
                    _, labels = torch.max(y, 1)
                    _, preds = torch.max(y_out, 1)
                    val_correct += torch.sum(preds == labels)
                    val_loss += loss.item()
            print(f"Validation - Epoch {e}: Loss {val_loss / len(fc_fp_dl_val)}, Accuracy {val_correct / len(fc_fp_ds_val)}")

# %%
