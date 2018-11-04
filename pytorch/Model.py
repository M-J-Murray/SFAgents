import torch.nn as nn 

# Creates a 2D convolutional layer with xavier initialisation
def initConv2D(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    nn.init.xavier_uniform_(conv.weight)
    return conv

# Creates the sequential layer steps required for a convolutional layer
def initConvLayer(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
    return nn.Sequential(
        initConv2D(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.ReLU(),
        nn.MaxPool2d(2, 2))

# The class used by PyTorch to represent the convolutional neural network
class Model(nn.Module):

    def __init__(self, frames, move_classes, attack_classes):
        super(Model, self).__init__()
        self.move_classes = move_classes
        self.attack_classes = attack_classes
        self.conv1 = initConvLayer(frames, 12)
        self.conv2 = initConvLayer(12, 24)
        self.conv3 = initConvLayer(24, 36)
        self.conv4 = initConvLayer(36, 36)
        self.conv5 = initConvLayer(36, 24)
        self.fc1 = nn.Linear(24*1*3, move_classes+attack_classes) # The fully connected tail
        nn.init.xavier_uniform_(self.fc1.weight)

    # Applies forward propagation to the inputs 
    def forward(self, frameStack):
        out = self.conv1(frameStack)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.fc1(out.view(out.size(0), -1))
        moveOut = out[:,0:self.move_classes]
        attackOut = out[:,self.move_classes:self.move_classes+self.attack_classes]
        return moveOut, attackOut