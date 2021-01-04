from torch import nn
import torchvision
import torch
class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        self.encoder_trans = nn.Linear(2048, 512) 
        
        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        ####
        vgg=torchvision.models.vgg16(pretrained=True)
        modules_vgg=list(vgg.features.children())[:-1]
        self.vgg=nn.Sequential(*modules_vgg)
        ####
        
    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        with torch.no_grad():
            out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
            out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
            out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
            batch_size=out.size(0)
            out = out.view(batch_size, -1, 2048)
            out = self.encoder_trans(out)
            
            
            #### VGG ####
            out_vgg=self.vgg(images)
            out_vgg = self.adaptive_pool(out_vgg)
            out_vgg=out_vgg.permute(0,2,3,1)
            batch_size=out_vgg.size(0)
            out_vgg=out_vgg.view(batch_size,-1,512)
            object_mean=torch.mean(torch.stack([out,out_vgg],dim=0).float(),dim=0)
        return object_mean

