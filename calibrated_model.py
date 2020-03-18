import torch
import os
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from model import Net_MCDO
from temperature_scaling import ModelWithTemperature
from train import load,save,data_load
import numpy as np

def save(name, net):
    net_path = './model/'+name+'.pkl'
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

def test(net, is_MCDO,testloader):
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
def get_calibrated_model():
    trainloader, valid_loader, testloader, log_freq = data_load()
    # load saved model
    net = Net_MCDO()
    net = nn.DataParallel(net)
    net = load('Net_MCDO',net)

    # Now we're going to wrap the model with a decorator that adds temperature scaling
    cali_model = ModelWithTemperature(net)

    # Tune the model temperature, and save the results
    cali_model.set_temperature(valid_loader)
    #model = nn.DataParallel(model)
    #model_filename = os.path.join('./model', 'model_with_temperature.pkl')
    save('model_with_temperature',cali_model)
    #torch.save(model.state_dict(), model_filename)
    print('Temperature scaled model sved')
    print('Done!')
    """ net = net.eval()
    cali_model = cali_model.eval() """
    test(net,True,testloader)
    test(cali_model,True,testloader)

if __name__ == "__main__":
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    get_calibrated_model()
    pass