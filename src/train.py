from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import logging
import pickle
import torchvision.models as models

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from network import KeenModel
from dataloader import KeenDataloader
from tqdm import tqdm
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
    'train_path' : "C:\\Users\\Karthik\\Documents\\KEEN_DATA\\Training",
    'val_path' : "C:\\Users\\Karthik\\Documents\\KEEN_DATA\\Validation",
    'epochs' : 20,
    'lr' : 0.0001,
    'wd' : 0.0001,
    'batch_size' : 64,
    'val_batch_size' : 64,
    'num_workers' : 4,
    'save_root' : "C:\\Users\\Karthik\\Desktop\\experiments\\AVGPOOL",
    'checkpoint' : "C:\\Users\\Karthik\\Desktop\\experiments\\AVGPOOL\\checkpoint",
    'logs_root' : "C:\\Users\\Karthik\\Desktop\\experiments\\AVGPOOL\\logs",
    'resume' : None,
    'print_freq' : 100,
    'save_freq' : 1,
    'val_freq' : 2,
    'initial_eval' : False,
    'is_training' : True,
    'lr_factor' : 0.1,
    'lr_step_size' : 7,
    'start_epoch' : 0,
}

def load_model(model, optimizer, config):
    if config["resume"] is not None : 
        checkpoint = torch.load(config['resume'], map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        #if 'optimizer' in checkpoint:
        #    optimizer.load_state_dict(checkpoint['optimizer'])

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def create_dataloader(config):
    train_set = KeenDataloader(config['train_path'], is_training=True)
    val_set = KeenDataloader(config['val_path'], is_training=False)
    tkwargs = {'batch_size': config['batch_size'],
               'num_workers': config['num_workers'],
               'pin_memory': True, 'drop_last': True}
    trainloader = DataLoader(train_set, **tkwargs)
    tkwargs = {'batch_size': config['val_batch_size'],
               'num_workers': config['num_workers'],
               'pin_memory': True, 'drop_last': True}
    testloader = DataLoader(val_set, **tkwargs)
    return trainloader, testloader

def create_model_and_optimizer():
    model = KeenModel(2, 256)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    load_model(model, optimizer, config)

    return model, criterion, optimizer

def train_step(model, criterion, optimizer, trainloader, epoch):
    running_loss = 0.0
    iteration = 0
    correct = 0
    total = 0
    for data in tqdm(trainloader):
        iteration+=1
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data['image'].to(device), data['label'].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        outputs = outputs.squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # calculate acc
        total += labels.size(0)
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()

        # print statistics
        running_loss += loss.item()
        if iteration % config['print_freq'] == 0:
            logging.info('[%d, %5d] loss: %.3f, accuracy: %.3f' %
                (epoch + 1, iteration, running_loss/(iteration+1e-5), correct/(total+1e-5)))
    logging.info('[%d, %5d] Epoch loss: %.3f, Accuracy: %.3f' %
                    (epoch + 1, iteration, running_loss/(iteration+1e-5), correct/(total+1e-5)))
    logging.info(f'Epoch {epoch} completed')
    return running_loss/(iteration+1e-5), correct/(total+1e-5)

def val_step(model, criterion, valloader):
    model.eval()
    val_loss = 0.0
    iteration = 0
    correct = 0
    total = 0
    for data in tqdm(valloader):
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
        val_loss += loss.item()
    logging.info('[%5d] Validation loss: %.3f, Validation Accuracy: %.3f' %
                    (iteration, val_loss/(iteration+1e-5), correct/(total+1e-5)))
    logging.info(f'Validation completed')
    return val_loss/(iteration+1e-5), correct/(total+1e-5)

def train(epochs, model, criterion, optimizer, trainloader, valloader):
    metrics = {'train_loss' : [], 'train_acc' : [], 'val_loss' : [], 'val_acc' : []}
    model.train()
    # le scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['lr_step_size'])
    for epoch in range(config['start_epoch'], epochs):
        logging.info(f"Current learning rate : {scheduler.get_last_lr()}")
        # train a single epoch
        train_loss, train_acc = train_step(model, criterion, optimizer, trainloader, epoch)
        # validation
        if epoch % config['val_freq'] == 0:
            val_loss, val_accuracy = val_step(model, criterion, valloader)
            metrics['val_loss'].append(val_loss)
            metrics['val_acc'].append(val_accuracy)
        # lr schedule
        scheduler.step()
        # update metrics
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        # save model
        if epoch % config['save_freq'] == 0:
            save_model(model, optimizer, epoch, config)
            logging.info(f"Model saved under : {os.path.join(config['save_root'], f'ckpt_epoch_{epoch}.pth')}")
    # save at the end of epoch
    save_model(model, optimizer, epoch, config)
    logging.info(f"Model saved under : {os.path.join(config['save_root'], f'ckpt_epoch_{epoch}.pth')}")
    return metrics

def save_model(model, optimizer, epoch, config):
    # save checkpoint
    model_optim_state = {'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         }
    model_name = os.path.join(
        config['checkpoint'], 'ckpt_epoch_%03d_.pth' % (
            epoch))
    torch.save(model_optim_state, model_name)
    logging.info('saved model {}'.format(model_name))

def predict(image_path, ckpt_path):
    label = {0 : "Fluten", 1: "Normalzustand"}
    # create model
    model = KeenModel(2, 256)
    # load checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    # basic transformations
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),])
    # open image
    image = Image.open(image_path)
    image = transform(image)
    image = torch.unsqueeze(image, dim=0)
    model = model.eval() 
    outputs = model(image)
    _, prediction = torch.max(outputs, dim=1)
    return label[int(prediction)]

if __name__ == '__main__':
    if config['is_training']:
        os.makedirs(config['save_root'], exist_ok=True)
        os.makedirs(config['logs_root'], exist_ok=True)
        os.makedirs(config['checkpoint'], exist_ok=True)

        logging.basicConfig(level=logging.INFO,
                        filename=os.path.join(config['logs_root'], 'stdout.log'),
                        format='%(asctime)s %(message)s')
        logging.info('Run configurations:')
        for item in config:
            logging.info(f"{item} : {config[item]}")
        logging.info("Creating dataloaders")
        trainloader, valloader = create_dataloader(config)
        logging.info("Creating model, optimizer and criterion functions")
        model, criterion, optimizer = create_model_and_optimizer()
    
        if config["initial_eval"]:
            val_loss = val_step(model, criterion, valloader)
        logging.info(f"Training model on {device}:")
        metrics = train(config['epochs'], model, criterion, optimizer, trainloader, valloader)
        with open(os.path.join(config['save_root'], 'metrics.pkl'), 'wb') as pkl:
            pickle.dump(metrics, pkl, pickle.HIGHEST_PROTOCOL)
        logging.info('Finished Training')
    else:
        #logging.basicConfig(level=logging.INFO,
        #                filename=os.path.join(config['save_root'], 'val.log'),
        #                format='%(asctime)s %(message)s')
        #trainloader, valloader = create_dataloader(config)
        ##model, criterion, optimizer = create_model_and_optimizer()
        #val_loss, val_accuracy = val_step(model, criterion, valloader)
        #print(val_loss, val_accuracy)
        
        img_dir = "C:\\Users\\Karthik\\Documents\\KEEN_DATA\\Validation\\Fluten"
        count = 0
        for img in sorted(os.listdir(img_dir)):
            if count == 20:
                break
            img_path = os.path.join(img_dir, img)
            output = predict(img_path, "C:\\Users\\Karthik\\Desktop\\experiments\\Exp6\\checkpoint\\ckpt_epoch_000_.pth")
            print(output)
            count += 1

