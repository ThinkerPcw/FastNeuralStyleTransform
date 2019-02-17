from model.transformer_net import TransformerNet
from model.loss_net import LossNet
from torch import optim
from torchvision import datasets
from utils import *
from torch.utils import data
from torchvision import transforms

LR = 0.001
BATCH_SIZE = 4
IMAGE_SIZE = 224
STYLE_WEIGHTS = [i * 2 for i in [1e2, 1e4, 1e4, 5e3, 1e4]]
DATASET = "/media/cvai/新加卷1/pcw/MS COCO 2017/"

m_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
train_dataset = datasets.ImageFolder(DATASET, m_transform)
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

style_img = get_image("./images/style image.jpg", m_transform).cuda()

# define network
transformerNet = TransformerNet().cuda()
lossNet = LossNet().cuda()

# define loss
mse = nn.MSELoss()

# Set the transformerNet to trainable and lossNet to untrainable
transformerNet.train()
lossNet.eval()

# define optimizer
optimizer = optim.Adam(transformerNet.parameters(), LR)

style_feature = lossNet(style_img.repeat(BATCH_SIZE, 1, 1, 1))
style_target = [GramMatrix(f).detach() for f in style_feature]

step = 0
for i in range(2):
    for contents_imgs, _ in train_loader:
        contents_imgs = contents_imgs.cuda()
        optimizer.zero_grad()
        # Use transformerNet generate images
        generate_imgs = transformerNet(contents_imgs)

        # Use lossNet calculate the style of generated images
        generate_features = lossNet(generate_imgs)
        style_generate = [GramMatrix(f) for f in generate_features]

        content_generate = generate_features[1]

        # use lossNet calculate the content of content images
        content_features = lossNet(contents_imgs)
        content_target = content_features[1].detach()

        content_loss = mse(content_generate, content_target)

        style_loss = 0
        for j in range(5):
            style_loss += STYLE_WEIGHTS[j] * mse(style_generate[j], style_target[j])

        loss = content_loss + style_loss
        loss.backward()

        if step % 100 == 0:
            print(step, "  content loss:", content_loss.data, "    style loss:", style_loss)

        if step % 600 == 0:
            show_image(contents_imgs.cpu().data, is_show=False)
            show_image(generate_imgs.cpu().data)

        if step > 8000 and step % 1000 == 0:
            save_network("storage", transformerNet, step)

        optimizer.step()

        step += 1
        if step > 13000:
            break
