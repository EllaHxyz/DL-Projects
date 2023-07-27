import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch
import torchvision
import matplotlib
from data_utils import *
from image_utils import *
from captum_utils import *
import numpy as np
from visualizers import GradCam
from image_utils import preprocess, deprocess
from data_utils import load_imagenet_val


def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled


def guided_bp(X, y):
    gc = GradCam()
    X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0).requires_grad_(True)
    y_tensor = torch.tensor(y)
    gc_model = torchvision.models.squeezenet1_1(pretrained=True)
    gbp_result = gc.guided_backprop(X_tensor, y_tensor, gc_model)
    return gbp_result


def gradcam(X, y):
    gc = GradCam()
    gc_model = torchvision.models.squeezenet1_1(pretrained=True)
    for param in gc_model.parameters():
        param.requires_grad = True
    X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0).requires_grad_(True)
    y_tensor = torch.LongTensor(y)
    gradcam_result = gc.grad_cam(X_tensor, y_tensor, gc_model)
    return gc_model, gradcam_result


def guided_gc(X, y):
    visualizations = {'Hay/Stingray': X[0]}
    gc = GradCam()
    gbp_img = guided_bp(X, y)
    visualizations['Guided Backprop'] = gbp_img[0]
    gc_model, gc_result = gradcam(X, y)
    visualizations['GradCAM'] = X[0] + (matplotlib.cm.jet(gc_result[0])[:, :, :3] * 255)
    visualizations['GradCAM'] = visualizations['GradCAM'] / np.max(visualizations['GradCAM'])
    gc_result.resize((1,224,224,1))
    img = gc_result * gbp_img
    visualizations['Guided GradCAM'] = img[0]
    return visualizations


try: 
    with Image.open("./visualization/fooling_image_2.png") as f:
        f = np.asarray(f, np.uint8)
        X = [f]
        y = [6]

        visualizations = guided_gc(X, y)
        plt.figure(figsize=(18, 4.2))
        for i, tup in enumerate(zip(visualizations.keys(), visualizations)):
            plt.subplot(1, 5, i + 1)
            img = visualizations[tup[0]]
            img = rescale(img)
            plt.imshow(img)
            plt.title(tup[1])
            plt.axis('off')
        plt.gcf().tight_layout()
        plt.savefig('./visualization/fooling_images_attr.png', bbox_inches='tight')
        plt.show()
except FileNotFoundError as e:
    print("All steps need to be completed to produce the required image.")
    print(e)
