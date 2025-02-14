import torch, copy, os
import torch.nn.quantized
from torch import nn
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt

DATA_ROOT = '../dataset/'

# PIL image -> normalized tensors
transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Get MNIST Dataloader
def get_mnist_data(batch_size, shuffle: bool = True) -> tuple:
    trainset_mnist = torchvision.datasets.MNIST(
        root = DATA_ROOT,
        train = True,
        download = True,
        transform = transform_mnist
    )
    testset_mnist = torchvision.datasets.MNIST(
        root = DATA_ROOT,
        train = False,
        download = True,
        transform = transform_mnist
    )
    trainloader = torch.utils.data.DataLoader(trainset_mnist, batch_size = batch_size, shuffle = shuffle, num_workers = 0)
    testloader = torch.utils.data.DataLoader(testset_mnist, batch_size = batch_size, shuffle = shuffle, num_workers = 0)
    return trainloader, testloader


@torch.no_grad()
def evaluate_accuracy(net: nn.Module, testdata, device):
    net.eval()
    net = net.to(device)
    
    correct_cnt = 0
    total_cnt = 0
    for inputs, targets in testdata:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = net(inputs)                                           # (batch, channel)
        
        max_vaues, max_indices = torch.max(outputs, dim = 1)            # (batch,)
        correct_cnt += torch.eq(max_indices, targets).sum().item()
        total_cnt += targets.size(dim = 0)

    return correct_cnt / total_cnt


def train(net: nn.Module, loss_func, optimizer, data, epoch_num, device, scheduler = None, disp_freq = 100, *, save_model=False, save_curve=False, **save_info):
    train_data, test_data = data
    plt_y = []
    
    net_snapshot = copy.deepcopy(net)
    accuracy_snapshot = 0.

    net = net.to(device)
    net.train()
    for epoch in range(epoch_num):

        print(f'----------epoch{epoch+1}----------')
        running_loss = 0.
        for i, (inputs, targets) in enumerate(train_data):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # inference
            outputs = net(inputs)
            loss = loss_func(outputs, targets)
            
            # optimize
            optimizer.zero_grad() # clear grad buffer
            loss.backward() # back propogate
            optimizer.step() # gradient descend 

            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item()

            if i % disp_freq == disp_freq - 1:
                
                accuracy = evaluate_accuracy(net, test_data, device)
                plt_y.append(accuracy)
                print(f'{i + 1:5d}th bathch, average_loss: {running_loss / 100:.3f}, accuracy: {accuracy*100:.2f}% ', end = '')

                if accuracy > accuracy_snapshot:
                    print('(new best record)')

                    state_dict = net.state_dict()
                    net_snapshot.load_state_dict(state_dict)
                    accuracy_snapshot = accuracy
                else:
                    print()
                
                running_loss = 0.0
                net.train()
    
    net.load_state_dict(net_snapshot.state_dict())
    print(f'done traning, best accuracy: {accuracy_snapshot*100:.2f}%')
    
    # save the trained model
    if save_model:
        torch.save(net_snapshot.state_dict(), save_info['save_path_net'])
    
    # save training curve
    if save_curve:
        plt.plot(plt_y)
        plt.xlabel(f'batch/{disp_freq}')
        plt.ylabel('accuracy')
        plt.title(save_info['module_name'] + ' Inference Accuracy')
        plt.savefig(save_info['save_path_plt'])

    return plt_y
