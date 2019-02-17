from model.transformer_net import TransformerNet
from utils import *

net = TransformerNet()
net = restore_network("storage", 12000, net)

m_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

content_img = get_image("./images/test image.jpg", m_transform).cuda()

output_image = net(content_img)
show_image(output_image.cpu().data)
save_image(output_image.cpu().data, "images/output image.jpg")
