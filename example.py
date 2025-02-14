import torch
from torch import nn
import torch.quantization
from torch.quantization import (
    QuantStub, DeQuantStub,
    QConfig, MinMaxObserver, HistogramObserver
)
import collections, os
from data_utils import train, get_mnist_data, evaluate_accuracy

import myquantlib

SAVE_PATH = './trained_net.pth'
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCH = 2

class ExampleNet(nn.Module):
    fuse_map = [['conv1', 'bn1'], ['conv2', 'bn2'], ['fc1', 'bn3'], ['fc2', 'bn4']]

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d((2,2))
        
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d((2,2))

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.relu4 = nn.ReLU()
        
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        out = self.maxpool1(self.relu1(self.bn1(self.conv1(input))))
        out = self.maxpool2(self.relu2(self.bn2(self.conv2(out))))

        out = self.flatten(out)
        
        out = self.relu3(self.bn3(self.fc1(out)))
        out = self.relu4(self.bn4(self.fc2(out)))

        return self.fc3(out)

def insert_stub(model: nn.Module) -> nn.Module:
    return nn.Sequential(
        collections.OrderedDict([
            ("quant", QuantStub()),
            ("model", model),
            ("dequant", DeQuantStub()),
        ])
    )

# Step 0: Prepare the pre-trained model

model = ExampleNet()
data = traindata, testdata = get_mnist_data(batch_size = BATCH_SIZE)

if os.path.exists(SAVE_PATH):
    model.load_state_dict(torch.load(SAVE_PATH, weights_only = True))
else:
    loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    train(model, loss, optim, data, EPOCH, 'cpu', save_model = True, save_path_net = SAVE_PATH)

model.eval()

# Step 1: Fuse Model
model_quantized = torch.quantization.fuse_modules(model, modules_to_fuse = model.fuse_map)

# Step 2: Insert Stubs
model_quantized = insert_stub(model_quantized)

# Step 3: Add QConfig
model_quantized.qconfig = QConfig(
    activation = HistogramObserver.with_args(dtype = torch.qint8, qscheme = torch.per_tensor_affine),
    weight = MinMaxObserver.with_args(dtype = torch.qint8, qscheme = torch.per_tensor_symmetric)
)

# Step 4: Insert Observers
torch.quantization.prepare(model_quantized, inplace = True)

# Step 5: Calibrate
evaluate_accuracy(model_quantized, testdata, 'cpu')

# ** Step 6: Convert to Quantized Model
torch.quantization.convert(
    model_quantized,
    mapping = myquantlib.MY_QUANT_MODULE_MAPPINGS,  # use maping provided by myquantlib to convert the modules into MyQuantizedModule
    inplace = True
)

# ** Step 7: Dry-Run: The converted model must perform a dry-run (_forward_qparams) to compute the qparams used in the actual pass
model_quantized(None)

# Final Step: Evaluation
original_accuracy = evaluate_accuracy(model, testdata, 'cpu')
quantized_accuracy = evaluate_accuracy(model_quantized, testdata, 'cpu')

print("Accuracy of original floating-point model:", original_accuracy)
print("Accuracy of quantized int8 model:", quantized_accuracy)
