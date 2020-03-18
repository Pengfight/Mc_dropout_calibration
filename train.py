import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm_notebook
from model import Net_MCDO
from temperature_scaling import ModelWithTemperature
import os

CE = nn.CrossEntropyLoss()


def update_target(target, original, update_rate):
        for target_param, param in zip(target.parameters(), original.parameters()):
            target_param.data.copy_((1.0 - update_rate) * target_param.data + update_rate*param.data)
def train(epoch, net, net_test, optimizer, log_freq):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = CE(outputs, labels)
        loss.backward()
        optimizer.step()
        update_target(net_test, net, 0.001)

        # print statistics
        running_loss += loss.item()
        if (i+1) % log_freq == 0:    # print every 2000 mini-batches
            print('[Epoch : %d, Iter: %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / log_freq))
    return running_loss / log_freq

def test(net, is_MCDO=False):
    print('Start test')
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            output = 0
            if is_MCDO:
                for i in range(10):
                    output += net(inputs)/10.
                output = torch.log(output)
                # print(output)
            else:
                output = net(inputs)
            _, predicted = torch.max(output, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    test_score = np.mean([100 * class_correct[i] / class_total[i] for i in range(10)])
    print(test_score)
    return test_score


# SAVE
def save(name, net):
    net_path = './model/'+name+'1.pkl'
    net = net.cpu()
    torch.save(net.state_dict(), net_path)
    # Place it to GPU back
    net.to(device)
    return net

def load(name, net):
    net_path = './model/'+name+'.pkl'
    net.to(device)
    # LOAD
    net.load_state_dict(torch.load(net_path))
    # Place it to GPU
    
    return net
""" def load(name, net, net_test):
    net_path = './model/'+name+'.pkl'
    net_test_path = './model/'+name+'_test.pkl'
    net.to(device)
    net_test.to(device)
    # LOAD
    net.load_state_dict(torch.load(net_path))
    net_test.load_state_dict(torch.load(net_test_path))
    # Place it to GPU
    
    return net, net_test """

def save_model(log_freq):
    #lenets = [Net, Net_DO, Net_MCDO]
    log_freq = log_freq
    for lenet in lenets:
        print(lenet.__name__)
        net = lenet()
        net_test = lenet()
        if torch.cuda.device_count() > 1:
            print("Let's use",torch.cuda.device_count(),"GPUs!")
            net = nn.DataParallel(net)
            net_test = nn.DataParallel(net_test)
        net.to(device)
        net_test.to(device)
        net_test.load_state_dict(net.state_dict())
        
        optimizer = optim.Adam(net.parameters(), lr=5e-4, weight_decay=0.0005, amsgrad=True)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)
        
        for i in tqdm_notebook(range(epoch_num)):
            scheduler.step()
            if lenet.__name__ == 'Net_DO':
                net.train()
                net_test.train()
            loss_avg = train(epoch=i, net=net, net_test=net_test, optimizer=optimizer,log_freq = log_freq)
            losses.append(loss_avg)
            if (i+1) % test_freq == 0:
                if lenet.__name__ == 'Net_DO':
                    print('NET_DO TEST')
                    net.eval()
                    net_test.eval()
                if lenet.__name__ == 'Net_MCDO':
                    print('NET_MCDO TEST')
                    print('Train net test')
                    net_score = test(net,is_MCDO=True)
                    net_scores.append(net_score)
                    print('Test net test')
                    test_score = test(net_test,is_MCDO=True)
                    test_scores.append(test_score)
                else:
                    print('Train net test')
                    net_score = test(net)
                    net_scores.append(net_score)
                    print('Test net test')
                    test_score = test(net_test)
                    test_scores.append(test_score)
                
        save(lenet.__name__, net)

def data_load():
    batch_size = 1000

    train_transform = transforms.Compose(
        [
    #   transforms.RandomCrop(32, padding=4),
    #   transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=4)
    trainset, valset = torch.utils.data.random_split(trainset, [40000, 10000])
    print('train:', len(trainset), 'validation:', len(valset))
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=4)

    valid_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=True, num_workers=4)
    print('train set size: {}'.format(len(trainset)))
    print('validation set size: {}'.format(len(valset)))
    log_freq = len(trainset)//batch_size
    print('log freq: {}'.format(log_freq))
    print('test set size: {}'.format(len(testset)))

    return trainloader,valid_loader,testloader,log_freq

if __name__ == "__main__":

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    trainloader, valid_loader, testloader, log_freq = data_load()
    lenets = [Net_MCDO]
    epoch_num = 400
    test_freq = 10
    losses = list()
    net_scores = list()
    test_scores = list()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # train and save model to directory './model'
    save_model(log_freq)

    """ # load saved model
    net = Net_MCDO()
    net = nn.DataParallel(net)
    net = load('Net_MCDO',net)
    # Now we're going to wrap the model with a decorator that adds temperature scaling
    model = ModelWithTemperature(net)

    # Tune the model temperature, and save the results
    model.set_temperature(valid_loader)
    #model = nn.DataParallel(model)
    #model_filename = os.path.join('./model', 'model_with_temperature.pkl')
    save('model_with_temperature',model)
    #torch.save(model.state_dict(), model_filename)
    print('Temperature scaled model sved')
    print('Done!') """
    """ net = Net_MCDO()
    net = nn.DataParallel(net)
    net = load('Net_MCDO',net)
    cali_net = torch.load('./model/model_with_temperature.pkl') """
    """ cali_net = Net_MCDO()
    cali_net = nn.DataParallel(cali_net)
    cali_net = load('model_with_temperature',cali_net) """
    """ with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            output = 0
            for i in range(10):
                #output += net_test(inputs)/10.
                output_org = net(inputs)/10.
                output_cali = cali_net(inputs)/10.
                #output = torch.log(output)
                print('11:',output_org,'22:',output_cali) """



