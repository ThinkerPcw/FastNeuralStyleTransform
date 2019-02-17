import torch
import torchvision


class LossNet(torch.nn.Module):
    def __init__(self):
        super(LossNet, self).__init__()
        self.module_list = list(torchvision.models.vgg19(pretrained=True).cuda().features)
        self.need_layer = [3, 8, 17, 26, 35]

    def forward(self, inputs):
        result = []
        x = self.module_list[0](inputs)
        for i in range(1, len(self.module_list)):
            x = self.module_list[i](x)
            if i in self.need_layer:
                result.append(x)
        return result
