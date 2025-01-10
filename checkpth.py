
import torch
if __name__ == '__main__':
    model_pth = r"G:\DeepLearning\Pytorch\SRResnet_all\SRResnet_DT\epochsX4\best.pth"
    net = torch.load(model_pth, map_location=torch.device('cuda'))
    print(net.keys())