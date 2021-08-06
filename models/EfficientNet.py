from efficientnet_pytorch import EfficientNet
from torch import nn
import torch


class EfficientNet_b0(torch.nn.Module):
    def __init__(self, num_class=1000, in_channels=3, pretrained=False, freeze_bn=False):
        super(EfficientNet_b0, self).__init__()
        self.model = EfficientNet.from_pretrained(model_name='efficientnet-b0', in_channels=in_channels)
        self.model._fc = nn.Linear(in_features=self.model._fc.in_features, out_features=num_class, bias=True)

    def forward(self, x):
        return self.model(x)


class EfficientNet_b1(torch.nn.Module):
    def __init__(self, num_class=1000, in_channels=3, pretrained=False, freeze_bn=False):
        super(EfficientNet_b1, self).__init__()
        self.model = EfficientNet.from_pretrained(model_name='efficientnet-b1', in_channels=in_channels)
        self.model._fc = nn.Linear(in_features=self.model._fc.in_features, out_features=num_class, bias=True)

    def forward(self, x):
        return self.model(x)


class EfficientNet_b2(torch.nn.Module):
    def __init__(self, num_class=1000, in_channels=3, pretrained=False, freeze_bn=False):
        super(EfficientNet_b2, self).__init__()
        self.model = EfficientNet.from_pretrained(model_name='efficientnet-b2', in_channels=in_channels)
        self.model._fc = nn.Linear(in_features=self.model._fc.in_features, out_features=num_class, bias=True)

    def forward(self, x):
        return self.model(x)


class EfficientNet_b3(torch.nn.Module):
    def __init__(self, num_class=1000, in_channels=3, pretrained=False, freeze_bn=False):
        super(EfficientNet_b3, self).__init__()
        self.model = EfficientNet.from_pretrained(model_name='efficientnet-b3', in_channels=in_channels)
        self.model._fc = nn.Linear(in_features=self.model._fc.in_features, out_features=num_class, bias=True)

    def forward(self, x):
        return self.model(x)


class EfficientNet_b4(torch.nn.Module):
    def __init__(self, num_class=1000, in_channels=3, pretrained=False, freeze_bn=False):
        super(EfficientNet_b4, self).__init__()
        self.model = EfficientNet.from_pretrained(model_name='efficientnet-b4', in_channels=in_channels)
        self.model._fc = nn.Linear(in_features=self.model._fc.in_features, out_features=num_class, bias=True)

    def forward(self, x):
        return self.model(x)


class EfficientNet_b5(torch.nn.Module):
    def __init__(self, num_class=1000, in_channels=3, pretrained=False, freeze_bn=False):
        super(EfficientNet_b5, self).__init__()
        self.model = EfficientNet.from_pretrained(model_name='efficientnet-b5', in_channels=in_channels)
        self.model._fc = nn.Linear(in_features=self.model._fc.in_features, out_features=num_class, bias=True)

    def forward(self, x):
        return self.model(x)


class EfficientNet_b6(torch.nn.Module):
    def __init__(self, num_class=1000, in_channels=3, pretrained=False, freeze_bn=False):
        super(EfficientNet_b6, self).__init__()
        self.model = EfficientNet.from_pretrained(model_name='efficientnet-b6', in_channels=in_channels)
        self.model._fc = nn.Linear(in_features=self.model._fc.in_features, out_features=num_class, bias=True)

    def forward(self, x):
        return self.model(x)


class EfficientNet_b7(torch.nn.Module):
    def __init__(self, num_class=1000, in_channels=3, pretrained=False, freeze_bn=False):
        super(EfficientNet_b7, self).__init__()
        self.model = EfficientNet.from_pretrained(model_name='efficientnet-b7', in_channels=in_channels)
        self.model._fc = nn.Linear(in_features=self.model._fc.in_features, out_features=num_class, bias=True)

    def forward(self, x):
        return self.model(x)


class EfficientNet_b8(torch.nn.Module):
    def __init__(self, num_class=1000, in_channels=3, pretrained=False, freeze_bn=False):
        super(EfficientNet_b8, self).__init__()
        self.model = EfficientNet.from_pretrained(model_name='efficientnet-b8', in_channels=in_channels)
        self.model._fc = nn.Linear(in_features=self.model._fc.in_features, out_features=num_class, bias=True)

    def forward(self, x):
        return self.model(x)
