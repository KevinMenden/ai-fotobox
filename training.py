import utils
from vgg import Vgg16
import os
import time
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx
from MobileStyleNet import MobileStyleNet
from transformer_net import TransformerNet
torch.autograd.set_detect_anomaly(True)


def train():
    device = torch.device("cuda")

    batch_size = 8
    content_weight = 1e5
    style_weight = 1e10
    image_size = 256
    model_name = "model_mobile_5e"

    # Generate the training dataset
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder("/home/kevin/food_rec/train/", transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Set up the model for training
    transformer = MobileStyleNet().to(device)

    if os.path.exists(model_name):
        transformer.load_state_dict(torch.load(model_name))

    optimizer = Adam(transformer.parameters(), 0.001)
    mse_loss = torch.nn.MSELoss()

    # Set up VGG16
    vgg = Vgg16(requires_grad=False).to(device)

    # Set up the style immage
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image("/home/kevin/fotobox/mosaic.jpg", size=image_size)
    style = style_transform(style)
    style = style.repeat(batch_size, 1, 1, 1).to(device)

    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(5):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss = style_loss + mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss = style_loss * style_weight

            total_loss = content_loss + style_loss

            total_loss.backward()
            optimizer.step()

            agg_content_loss = agg_content_loss + content_loss.item()
            agg_style_loss = agg_style_loss + style_loss.item()

            if (batch_id + 1) % 10 == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)
                transformer.eval().cpu()
                torch.save(transformer.state_dict(), model_name)
                transformer.to(device).train()

    print("\nDone, trained model saved at")


train()
