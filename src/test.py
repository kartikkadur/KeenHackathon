from __future__ import print_function, division
import torch
import torch.nn as nn
import logging

from PIL import Image
from torch.utils.data import DataLoader
from network import KeenModel, keen_model
from dataloader import KeenDataloader
from tqdm import tqdm


test_config = {'batch_size': 16,
               'num_workers': 0,
               'pin_memory': True, 
               'drop_last': True}

def test_step(img_dir, ckpt_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if ckpt_path is not None:
        model = KeenModel(2, 256)
        # load checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model = keen_model(True)
    model = model.to(device)
    # create criterion
    criterion = nn.CrossEntropyLoss().to(device)
    # create testset
    testset = KeenDataloader(img_dir, labels=label, is_training=False)
    testloader = DataLoader(testset, **test_config)
    model.eval()
    test_loss = 0.0
    iteration = 0
    correct = 0
    total = 0
    for data in tqdm(testloader):
        iteration+=1
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data['image'].to(device), data['label'].to(device)

        # forward + backward + optimize
        outputs = model(inputs)
        outputs = outputs.squeeze()
        loss = criterion(outputs, labels)
        total += labels.size(0)
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()

        test_loss += loss.item()
    print('[%5d] Test loss: %.3f, Test Accuracy: %.3f' %
                    (iteration, test_loss/(iteration+1e-5), correct/(total+1e-5)))
    logging.info(f'Test completed')
    return test_loss/(iteration+1e-5), correct/(total+1e-5)

def predict(image_path, ckpt_path=None):
    label = {0 : "Fluten", 1: "Normalzustand"}
    # create model
    if ckpt_path is not None:
        model = KeenModel(2, 256)
        # load checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model = keen_model(True)
    # basic transformations
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                                    transforms.Normalize((0.5021, 0.4781, 0.4724), (0.3514, 0.3439, 0.3409))])
    # open image
    image = Image.open(image_path)
    image = transform(image)
    image = torch.unsqueeze(image, dim=0)
    model = model.eval() 
    outputs = model(image)
    _, prediction = torch.max(outputs, dim=1)
    probablities = torch.nn.functional.softmax(outputs.squeeze(), dim=0)
    print(f"Fluten : {probablities[0]*100} %, Normalzustand : {probablities[1]*100} %")
    return label[int(prediction)]

if __name__ == "__main__":
    loss, acc = test_step("data/ValidationData")
    print(loss, acc)