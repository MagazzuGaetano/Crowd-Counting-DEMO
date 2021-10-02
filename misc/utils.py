import torch
from torch import nn
from models.CrowdCounter import CrowdCounter
import torchvision.transforms as standard_transforms
from torch.autograd import Variable

from PIL import Image
import numpy as np
import cv2

def initialize_weights(models):
    for model in models:
        real_init_weights(model)

def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, nn.Conv2d):    
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m,nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print(m)

def weights_normal_init(*models):
    for model in models:
        dev=0.01
        if isinstance(model, list):
            for m in model:
                weights_normal_init(m, dev)
        else:
            for m in model.modules():            
                if isinstance(m, nn.Conv2d):        
                    m.weight.data.normal_(0.0, dev)
                    if m.bias is not None:
                        m.bias.data.fill_(0.0)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, dev)


def remove_transparency(im):
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        im = np.array(im)
        im = im[:,:,:3]
        image_without_alpha = Image.fromarray(np.uint8(im))
        return image_without_alpha
    else:
        return im

def save_map(pred_map):
    pred_map = cv2.blur(pred_map,(15,15))

    im = Image.fromarray(np.uint8(pred_map * 255) , 'L')
    im.save("static/prediction.jpg", cmap='L')

    im = cv2.imread('static/prediction.jpg', cv2.IMREAD_GRAYSCALE)
    im_color = cv2.applyColorMap(im, cv2.COLORMAP_JET)
    cv2.imwrite('static/map.jpg', im_color)

def get_model_filename(model_name):
    model_path = None

    if model_name == 'AlexNet'.lower():
        model_path = '1_alexnet.pth'
    elif model_name == 'VGG'.lower():
        model_path = '4_vgg.pth'
    elif model_name == 'VGG_DECODER'.lower():
        model_path = '1_vgg_decoder.pth'
    elif model_name == 'MCNN'.lower():
        model_path = '4_mcnn.pth'
    elif model_name == 'CSRNet'.lower():
        model_path = '4_csrnet.pth'
    elif model_name == 'SANET'.lower():
        model_path = '1_sanet.pth'
    elif model_name == 'ResNet50'.lower():
        model_path = '1_resnet50_all_ep_135_mae_8.6_mse_14.1.pth'
    elif model_name == 'ResNet101'.lower():
        model_path = '1_resnet101.pth'
    elif model_name == 'SFCN+'.lower():
        model_path = '4_sfcn+.pth'
    elif model_name == 'SCAR'.lower():
        model_path = '4_scar.pth'
    else:
        raise Exception("the selected model_name is not valid!")

    return model_path

def predict(dataset, image, model):
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True

    img = Image.open('./' + image)
    net = CrowdCounter(model)
    model_path = 'static/models/' + get_model_filename(model)

    checkpoint = torch.load(model_path)
    mean_std = checkpoint['mean_std']
    net.load_state_dict(checkpoint['state_dict'], strict=False)
    net.cuda()
    net.eval()

    img_transform = standard_transforms.Compose(
        [standard_transforms.ToTensor(), standard_transforms.Normalize(*mean_std)]
    )
    img = remove_transparency(img) 
    if img.mode == 'L':
        img = img.convert('RGB')
    img = img_transform(img)

    with torch.no_grad():
        img = Variable(img[None,:,:,:]).cuda()
        if not(torch.cuda.is_available()):
            device = "cpu"
        else:
            device = None
        pred_map = net.test_forward(img, device)

    pred_map = pred_map.cpu().data.numpy()[0, 0, :, :]
    pred = np.sum(pred_map) / 100.0
    pred_map = pred_map / np.max(pred_map + 1e-20)

    return pred_map, pred
