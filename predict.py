
import torch
from torch import nn
from torchvision import models
from collections import OrderedDict
from get_input_args import predict_args
from PIL import Image
import numpy as np
import json

in_args = predict_args()



with open(in_args.category_names, 'r') as f:
    cat_to_name = json.load(f)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    img =Image.open(image)
    img.thumbnail((500, 256))
    left = (img.width-224)/2
    bottom = (img.height-224)/2
    right = left + 224
    top= bottom + 224
    img = img.crop((left, bottom, right,top))
    np_image = np.array(img)
    np_image = np_image/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    img = np_image.transpose((2, 0, 1))
    return torch.from_numpy(img).type(torch.FloatTensor)

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

archs = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}


    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = archs[checkpoint['model']]
    for param in model.parameters():
        param.requires_grad = False
#     classifier = nn.Sequential(OrderedDict([
#         ('fc1', nn.Linear(25088, 512)),
#         ('relu', nn.ReLU()),
#         ('fc2', nn.Linear(512, 102)),
#         ('output', nn.LogSoftmax(dim=1))
#     ]))
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers'])),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(checkpoint['hidden_layers'], checkpoint['output_size'])),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])

    return model, checkpoint["class_to_idx"]

device= torch.device("cuda" if in_args.gpu == 'cuda' and torch.cuda.is_available() else "cpu")

def predict(image_path, model):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file

    inputs = process_image(image_path)
    inputs = inputs.to(device)
    class_to_idx = model[1]
    model = model[0].to(device)
    model.eval()
    inputs.unsqueeze_(0)
    logps = model.forward(inputs)
    ps = torch.exp(logps)
    index_to_class = {v: k for k, v in class_to_idx.items()}
    top_p, top_index = ps.topk(in_args.top_k, dim=1)
    top_probs = top_p.detach().numpy().tolist()[0]
    top_class = top_index.detach().numpy().tolist()[0]
    top_class = [index_to_class[x] for x in top_class]
    flowers = [cat_to_name[lab] for lab in top_class]

    return flowers, top_probs,


flower_name, class_probability = predict(in_args.path_to_image,load_checkpoint(in_args.checkpoint))
print(flower_name, class_probability)