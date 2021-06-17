# matplotlib inline
import time
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image
import d2lzh_pytorch as d2l
import os


if torch.cuda.is_available():
    print("yes")
else:
    print('no')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

d2l.set_figsize()
#content_img = Image.open('../../data/rainier.jpg')
content_img = Image.open('./data/content_image/David.jpg')
#d2l.plt.imshow(content_img);

d2l.set_figsize()
style_img = Image.open('./data/style_image/face.jpg')
#d2l.plt.imshow(style_img);

rgb_mean = np.array([0.485, 0.456, 0.406])   #预训练模型的参数，将图片做同样的标准化，保持一致
rgb_std = np.array([0.229, 0.224, 0.225])


def preprocess(PIL_img, image_shape):
    process = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),  #更改输入图像的尺寸
        torchvision.transforms.ToTensor(),        #将PIL图像转成网络能接受的tensor
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])    #在rgb三个通道分别做标准化

    return process(PIL_img).unsqueeze(dim=0)  # (batch_size, 3, H, W)


def postprocess(img_tensor):     #将图像中的像素值转变成标准化之前的值
    inv_normalize = torchvision.transforms.Normalize(    #反标准化
        mean=-rgb_mean / rgb_std,
        std=1 / rgb_std)
    to_PIL_image = torchvision.transforms.ToPILImage()   #将tensor转变成PIL
    return to_PIL_image(inv_normalize(img_tensor[0].cpu()).clamp(0, 1))


pretrained_net = torchvision.models.vgg19(pretrained=True, progress=True)

style_layers, content_layers = [0, 5, 10, 19, 28], [25]   #需要保留的样式层和内容层

net_list = []
for i in range(max(content_layers + style_layers) + 1):
    net_list.append(pretrained_net.features[i])
net = torch.nn.Sequential(*net_list)


def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles


def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y


def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y


def content_loss(Y_hat, Y):
    return F.mse_loss(Y_hat, Y)


def gram(X):
    num_channels, n = X.shape[1], X.shape[2] * X.shape[3]
    X = X.view(num_channels, n)
    return torch.matmul(X, X.t()) / (num_channels * n)


def style_loss(Y_hat, gram_Y):
    return F.mse_loss(gram(Y_hat), gram_Y)


def tv_loss(Y_hat):
    return 0.5 * (F.l1_loss(Y_hat[:, :, 1:, :], Y_hat[:, :, :-1, :]) +
                  F.l1_loss(Y_hat[:, :, :, 1:], Y_hat[:, :, :, :-1]))

content_weight, style_weight, tv_weight = 1, 1e3, 10

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # 分别计算内容损失、样式损失和总变差损失
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # 对所有损失求和
    l = sum(styles_l) + sum(contents_l) + tv_l
    return contents_l, styles_l, tv_l, l


class GeneratedImage(torch.nn.Module):   #合成图像是唯一需要更新的变量，将合成图像视为参数来训练
    def __init__(self, img_shape):
        super(GeneratedImage, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(*img_shape))    #将不可训练类型tensor转换成parameter，在
        # 参数优化时可进行优化，随机生成一个和image_shape一样大小的[0,1)之间的噪声

    def forward(self):
        return self.weight


def get_inits(X, device, lr, styles_Y):     #计算合成图像
    gen_img = GeneratedImage(X.shape).to(device)
    gen_img.weight.data = X.data
    optimizer = torch.optim.Adam(gen_img.parameters(), lr=lr)  #优化合成图像
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, optimizer


def train(X, contents_Y, styles_Y, device, lr, max_epochs, lr_decay_epoch):
    print("training on ", device)
    X, styles_Y_gram, optimizer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_epoch, gamma=0.1)   #optimizer指的是需要优化的参数
    #StepLR等间隔调整学习率
    for i in range(max_epochs):
        start = time.time()

        contents_Y_hat, styles_Y_hat = extract_features(
                X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
                X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)

        optimizer.zero_grad()   #把梯度置为0
        l.backward(retain_graph = True)   #反向传播
        optimizer.step()    #更新参数
        scheduler.step()    #更新参数

        if i % 50 == 0 and i != 0:
            print('epoch %3d, content loss %.2f, style loss %.2f, '
                  'TV loss %.2f, %.2f sec'
                  % (i, sum(contents_l).item(), sum(styles_l).item(), tv_l.item(),
                     time.time() - start))
    return X.detach()

image_shape =  (150, 225)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
style_X, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.01, 500, 200)

#d2l.plt.imshow(postprocess(output));


image_shape = (300, 450)
_, content_Y = get_contents(image_shape, device)
_, style_Y = get_styles(image_shape, device)
X = preprocess(postprocess(output), image_shape).to(device)
big_output = train(X, content_Y, style_Y, device, 0.01, 500, 200)


d2l.set_figsize((7, 5))
img = postprocess(big_output)
#d2l.plt.imshow(img);
save_filename = 'result'
save_path = './results'
save_path = os.path.join(save_path,save_filename)
torch.save(img,save_path)
