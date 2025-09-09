import torch
from torch import nn
import torch.quantization
from torch.quantization import (
    QuantStub, DeQuantStub,
    QConfig, MovingAverageMinMaxObserver
)
from torch.ao.quantization.fake_quantize import FakeQuantize, default_weight_fake_quant
import collections, os
from data_utils import train, get_mnist_data, evaluate_accuracy

import myquantlib

SAVE_PATH = './trained_net.pth'
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCH = 10
BACKEND = "qnnpack" if "qnnpack" in torch.backends.quantized.supported_engines else "fbgemm"

torch.backends.quantized.engine = BACKEND

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


# Step 0-3: Similar to step 0-3 in `example.py`
model = ExampleNet()
data = traindata, testdata = get_mnist_data(batch_size = BATCH_SIZE)

if os.path.exists(SAVE_PATH):
    print("Found pretrained model.")
    model.load_state_dict(torch.load(SAVE_PATH, weights_only = True))
else:
    print("No pretrained model found. Pretraing model...")
    loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    train(model, loss, optim, data, EPOCH, 'cpu', save_model = True, save_path_net = SAVE_PATH)

model.eval()
model_quantized = torch.quantization.fuse_modules(model, modules_to_fuse = model.fuse_map)
model_quantized = insert_stub(model_quantized)

model_quantized.qconfig = QConfig(
    activation = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver.with_args(dtype=torch.qint8),
        dtype=torch.qint8
    ),
    weight = default_weight_fake_quant
)

# Step 4: Insert Fake Quantizer
torch.quantization.prepare_qat(model_quantized, inplace=True)

# ** Step 5: Quantization-Aware Training
train(
    net = model_quantized,
    loss_func = nn.CrossEntropyLoss(),
    optimizer = torch.optim.SGD(model_quantized.parameters(), lr=0.01),
    data = data,
    epoch_num = 1,
    device = 'cpu'
)

# Step 5: Convert to Actual Quantized Model
model_quantized.eval()
torch.quantization.convert(model_quantized, inplace=True, mapping = myquantlib.MY_QUANT_MODULE_MAPPINGS)

# Step 7: Dry-Run
model_quantized(None)

# Final Step: Evaluation
original_accuracy = evaluate_accuracy(model, testdata, 'cpu')
quantized_accuracy = evaluate_accuracy(model_quantized, testdata, 'cpu')

print("The accuracy of the original floating-point model is", original_accuracy)
print("The accuracy of the quantization-aware trained model is", quantized_accuracy)
