
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from get_input_args import train_args
from collections import OrderedDict

in_arg = train_args()

data_dir = in_arg.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

val_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
data_transforms = [train_transforms,val_transforms,test_transforms]

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
val_data = datasets.ImageFolder(valid_dir, transform=val_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
image_datasets = [train_data,val_data,test_data]

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(val_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
dataloaders = [trainloader,valloader,testloader]

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

archs = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16} #

model = archs[in_arg.arch]
in_features = model.classifier[0].in_features if (in_arg.arch == 'resnet') else(model.classifier[1].in_features if (in_arg.arch == 'alexnet')
                                                                               else model.fc.in_features) 
for param in model.parameters():
    param.requires_grad = False


# create model
classifier = nn.Sequential(OrderedDict([
('fc1', nn.Linear(in_features, in_arg.hidden_units)),
('relu', nn.ReLU()),
('fc2', nn.Linear(in_arg.hidden_units, 102)),
('output', nn.LogSoftmax(dim=1))
]))
model.classifier = classifier

# train model
# Only train the classifier parameters, feature parameters are frozen
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=in_arg.learning_rate)
device= torch.device("cuda" if in_arg.gpu == 'cuda' and torch.cuda.is_available() else "cpu")

model.to(device)


for epoch in range(in_arg.epochs):
    running_loss = 0
    for inputs, labels in trainloader:
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    val_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            val_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Epoch {epoch+1}/{in_arg.epochs}.. "
          f"Train loss: {running_loss/len(trainloader):.3f}.. "
          f"Validation loss: {val_loss/len(valloader):.3f}.. "
          f"Validation accuracy: {accuracy/len(valloader):.3f}")
    model.train()

print("here1")
# test model
model.eval()
val_loss = 0
accuracy = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)

        val_loss += batch_loss.item()

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print(f"Test loss: {val_loss / len(testloader):.3f}.. "
      f"Test accuracy: {accuracy / len(testloader):.3f}")

# Save the checkpoint
model.class_to_idx = image_datasets[0].class_to_idx
checkpoint = {'input_size': in_features,
              'output_size': 102,
              'hidden_layers': in_arg.hidden_units,
              'optimizer_dict': optimizer.state_dict(),
              'state_dict': model.state_dict(),
             "class_to_idx":model.class_to_idx,
             "model":in_arg.arch}
torch.save(checkpoint, 'checkpoint.pth')

