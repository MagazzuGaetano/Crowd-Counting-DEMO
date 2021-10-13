import torch
import torch.nn as nn
import torch.nn.functional as F

class CrowdCounter(nn.Module):
    def __init__(self, model_name):
        super(CrowdCounter, self).__init__()        

        if model_name == 'AlexNet'.lower():
            from .counters.AlexNet import AlexNet as net        
        elif model_name == 'VGG'.lower():
            from .counters.VGG import VGG as net
        elif model_name == 'VGG_DECODER'.lower():
            from .counters.VGG_decoder import VGG_decoder as net
        elif model_name == 'MCNN'.lower():
            from .counters.MCNN import MCNN as net
        elif model_name == 'CSRNet'.lower():
            from .counters.CSRNet import CSRNet as net
        elif model_name == 'SCAR'.lower():
            from .counters.SCAR import SCAR as net
        elif model_name == 'ResNet50'.lower():
            from .counters.Res50 import Res50 as net
        elif model_name == 'ResNet101'.lower():
            from .counters.Res101 import Res101 as net            
        elif model_name == 'SFCN+'.lower():
            from .counters.Res101_SFCN import Res101_SFCN as net
        elif model_name == "SANet".lower():
            from .counters.SANet import SANet as net

        self.CCN = net()

    def test_forward(self, img):               
        density_map = self.CCN(img)                    
        return density_map
