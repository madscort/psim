import torch
import torch.nn as nn

class SequenceNetGlobalAvg(nn.Module):
    def __init__(self):
        super(SequenceNetGlobalAvg, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = torch.mean(x, dim=2)  # Global average pooling
        x = self.fc(x)
        return x.squeeze(-1)


class SequenceNetGlobalKernel(nn.Module):
    def __init__(self, conv_dropout_rate=0.1, fc_dropout_rate=0.5, kernel_size=(3, 5)):
        super(SequenceNetGlobalKernel, self).__init__()

        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv1 = nn.Conv1d(in_channels=5, out_channels=16,
                               kernel_size=kernel_size[0], stride=1, padding=padding[0])
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32,
                               kernel_size=kernel_size[1], stride=1, padding=padding[1])
        self.fc = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout_conv = nn.Dropout(conv_dropout_rate)
        self.dropout = nn.Dropout(fc_dropout_rate)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)


class BasicCNN(nn.Module):
    def __init__(self, alt_dropout_rate: float=0.1, fc_dropout_rate: float=0.5, activation_fn: str='ReLU', batchnorm: bool=False, fc_num: int=1, conv_num: int=3, kernel_size: tuple=(3, 3, 3)):
        super(BasicCNN, self).__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)

        self.conv1 = nn.Conv1d(in_channels=5, out_channels=16,
                               kernel_size=kernel_size[0], stride=1, padding=padding[0])
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32,
                               kernel_size=kernel_size[1], stride=1, padding=padding[1])
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32,
                               kernel_size=kernel_size[2], stride=1, padding=padding[2])

        self.fc_pre = nn.Linear(32, 32)
        self.fc_opt1 = nn.Linear(32, 16)
        self.fc_opt2 = nn.Linear(16, 1)
        self.fc = nn.Linear(32, 1)
        self.activation_fn = getattr(nn, activation_fn)()
        self.dropout_conv = nn.Dropout(alt_dropout_rate)
        self.dropout = nn.Dropout(fc_dropout_rate)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.batchnorm = batchnorm
        self.fc_num = fc_num
        self.conv_num = conv_num

    def forward(self, x):
        # Conv1
        x = self.conv1(x)
        x = self.activation_fn(x)
        if self.batchnorm:
            x = self.bn1(x)
        x = self.dropout_conv(x)
        # Conv2
        x = self.conv2(x)
        x = self.activation_fn(x)
        if self.batchnorm:
            x = self.bn2(x)
        x = self.dropout_conv(x)

        # 1. optional Conv
        if self.conv_num > 2:
            x = self.conv3(x)
            x = self.activation_fn(x)
            if self.batchnorm:
                x = self.bn2(x)
            x = self.dropout_conv(x)

        # Global avg pooling
        x = torch.mean(x, dim=2)
        
        # 1. optional FC
        if self.fc_num > 1:
            x = self.fc_pre(x)
            x = self.activation_fn(x)
            x = self.dropout(x)
        # 2. optional FC
        if self.fc_num == 3:
            x = self.fc_opt1(x)
            x = self.activation_fn(x)
            x = self.dropout(x)
            x = self.fc_opt2(x)
            x = self.dropout(x)
        else:
            x = self.fc(x)
            x = self.dropout(x)
        return x.squeeze(-1)

class SequenceNetGlobalDropOut(nn.Module):
    def __init__(self, conv_dropout_rate=0.1, fc_dropout_rate=0.5):
        super(SequenceNetGlobalDropOut, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32, 1)  # Since global average pooling will be applied
        self.relu = nn.ReLU()
        self.dropout_conv = nn.Dropout(conv_dropout_rate)
        self.dropout = nn.Dropout(fc_dropout_rate)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout_conv(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout_conv(x)
        x = torch.mean(x, dim=2)  # Global average pooling
        x = self.fc(x)
        x = self.dropout(x)
        return x.squeeze(-1)
    
class SequenceNetGlobalDropOutBatchNorm(nn.Module):
    def __init__(self, conv_dropout_rate=0.1, fc_dropout_rate=0.5):
        super(SequenceNetGlobalDropOutBatchNorm, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32, 1)  # Since global average pooling will be applied
        self.relu = nn.ReLU()
        self.dropout_conv = nn.Dropout(conv_dropout_rate)
        self.dropout = nn.Dropout(fc_dropout_rate)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.dropout_conv(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.dropout_conv(x)
        x = torch.mean(x, dim=2)  # Global average pooling
        x = self.fc(x)
        x = self.dropout(x)
        return x.squeeze(-1)

class SequenceNetGlobalKernel(nn.Module):
    def __init__(self, conv_dropout_rate=0.1, fc_dropout_rate=0.5, kernel_size=(3, 5)):
        super(SequenceNetGlobalKernel, self).__init__()

        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv1 = nn.Conv1d(in_channels=5, out_channels=16,
                               kernel_size=kernel_size[0], stride=1, padding=padding[0])
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32,
                               kernel_size=kernel_size[1], stride=1, padding=padding[1])
        self.fc = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout_conv = nn.Dropout(conv_dropout_rate)
        self.dropout = nn.Dropout(fc_dropout_rate)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.dropout_conv(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.dropout_conv(x)
        x = torch.mean(x, dim=2)  # Global average pooling
        x = self.fc(x)
        x = self.dropout(x)
        return x.squeeze(-1)

class SequenceNetFlat(nn.Module):
    def __init__(self, conv_dropout_rate=0.1, fc_dropout_rate=0.5):
        super(SequenceNetFlat, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32*25000, 1)  # No global pooling, so input size is 32*25000
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten before fully connected layer
        x = self.fc(x)
        return x.squeeze(-1)

class SequenceNetFlatDropOut(nn.Module):
    def __init__(self, conv_dropout_rate=0.1, fc_dropout_rate=0.5):
        super(SequenceNetFlatDropOut, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32*25000, 1)
        self.relu = nn.ReLU()
        self.dropout_conv = nn.Dropout(conv_dropout_rate)
        self.dropout = nn.Dropout(fc_dropout_rate)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout_conv(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dropout(x)
        return x.squeeze(-1)
    
class SequenceNetFlatDropOutBatchNorm(nn.Module):
    def __init__(self, conv_dropout_rate=0.1, fc_dropout_rate=0.5):
        super(SequenceNetFlatDropOutBatchNorm, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc = nn.Linear(32*25000, 1)
        self.relu = nn.ReLU()
        self.dropout_conv = nn.Dropout(conv_dropout_rate)
        self.dropout = nn.Dropout(fc_dropout_rate)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.dropout_conv(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.dropout_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dropout(x)
        return x.squeeze(-1)

class SequenceNetFlatCustomDepth(nn.Module):
    def __init__(self, conv_layers=[(5, 16), (16, 32)],
                 fc_layers=[32*25000, 1], dropout_rate=0.5):
        super(SequenceNetFlatCustomDepth, self).__init__()
        self.conv_layers = nn.ModuleList([nn.Conv1d(in_channels=in_c,
                                                    out_channels=out_c, kernel_size=3,
                                                    stride=1,
                                                    padding=1) for in_c, out_c in conv_layers])
        self.fc_layers = nn.ModuleList([nn.Linear(in_features=in_f,
                                                  out_features=out_f) for in_f, out_f in zip(fc_layers[:-1], fc_layers[1:])])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)
            x = self.relu(x)
            x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        
        for fc in self.fc_layers:
            x = fc(x)
            x = self.relu(x)
        
        return x.squeeze(-1)

class SequenceNetGlobalAvgPool(nn.Module):
    def __init__(self):
        super(SequenceNetGlobalAvgPool, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64*6250, 1)  # After 2 pooling layers, sequence length is 6250
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten before fully connected layer
        x = self.fc(x)
        return x.squeeze(-1)


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )

        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )
        
        self.branch3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)

        outputs = [branch1_out, branch2_out, branch3_out]
        return torch.cat(outputs, 1)


class SequenceNetGlobalInception(nn.Module):
    def __init__(self, conv_dropout_rate=0.1, fc_dropout_rate=0.5):
        super(SequenceNetGlobalInception, self).__init__()
        
        self.inception1 = InceptionModule(in_channels=5, out_channels=16)

        self.fc = nn.Linear(48, 1)  # Adjust the input dimension as per the inception module output
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(fc_dropout_rate)
    
    def forward(self, x):
        x = self.inception1(x)

        x = torch.mean(x, dim=2)  # Global average pooling
        x = self.fc(x)
        x = self.dropout(x)
        return x.squeeze(-1)

class SequenceNetGlobalInceptionV2(nn.Module):
    def __init__(self, conv_dropout_rate=0.1, fc_dropout_rate=0.5):
        super(SequenceNetGlobalInceptionV2, self).__init__()
        
        self.inception1 = InceptionModule(in_channels=5, out_channels=16)
        self.inception1 = InceptionModule(in_channels=5, out_channels=16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # Adding pooling layer
        self.inception2 = InceptionModule(in_channels=48, out_channels=32) 
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # Adding another pooling layer
        
        self.fc = nn.Linear(96, 1)  # Adjust the input dimension as per the inception module output and pooling layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(fc_dropout_rate)
    
    def forward(self, x):
        x = self.inception1(x)
        x = self.pool1(x)
        x = self.inception2(x)
        x = self.pool2(x)
        x = torch.mean(x, dim=2)  # Global average pooling
        x = self.fc(x)
        x = self.dropout(x)
        return x.squeeze(-1)




class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)  # for channel matching
        self.skip_bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.skip(identity)
        identity = self.skip_bn(identity)

        out += identity
        out = self.relu(out)

        return out


class SequenceNetWithResBlock(nn.Module):
    def __init__(self):
        super(SequenceNetWithResBlock, self).__init__()
        self.conv_initial = nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.resblock = ResBlock(16, 32)
        self.fc = nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.conv_initial(x)
        x = self.resblock(x)
        x = torch.mean(x, dim=2)  # Global average pooling
        x = self.fc(x)
        return x.squeeze(-1)


# class SequenceEfficientNet(nn.Module):
#     def __init__(self, model_name='efficientnet_b0'):
#         super(SequenceEfficientNet, self).__init__()

#         # Load the pre-trained model with timm
#         self.base_model = timm.create_model(model_name, pretrained=False)

#         # Fetch the number of in_features from the original first convolutional layer
#         in_features = self.base_model.conv_stem.out_channels

#         # Replace the first convolutional layer to accept 5 channels
#         self.base_model.conv_stem = nn.Conv2d(5, in_features, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), bias=False)

#         # Adjust the final layer to your specific problem (binary classification)
#         self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, 1)

#     def forward(self, x):
#         # Reshape the input to [batch_size, channels, height, width]
#         x = x.unsqueeze(2)  # adds the height dimension
#         return self.base_model(x).squeeze(-1)

# class SequenceTimm(nn.Module):
#     def __init__(self, model_name='mobilenetv2_100'):
#         super(SequenceTimm, self).__init__()

#         # Load the pre-trained model with timm
#         self.base_model = timm.create_model(model_name, pretrained=False)

#         # Adjust the first convolutional layer to accept 5 channels
#         if hasattr(self.base_model, 'conv_stem'):
#             in_features_stem = self.base_model.conv_stem.out_channels
#             self.base_model.conv_stem = nn.Conv2d(5, in_features_stem, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), bias=False)
#         elif hasattr(self.base_model, 'conv1'):
#             in_features_conv1 = self.base_model.conv1.out_channels
#             self.base_model.conv1 = nn.Conv2d(5, in_features_conv1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), bias=False)

#         # Fetch the number of in_features from the last fully connected layer and adjust it
#         last_layer_name = 'fc' if hasattr(self.base_model, 'fc') else 'classifier'
#         num_features = getattr(self.base_model, last_layer_name).in_features
#         setattr(self.base_model, last_layer_name, nn.Linear(num_features, 1))

#     def forward(self, x):
#         # Reshape the input to [batch_size, channels, height, width]
#         x = x.unsqueeze(2)  # adds the height dimension
#         return self.base_model(x).squeeze(-1)


# class InceptionModule(nn.Module):
#     def __init__(self, in_channels, f1, f2):
#         super(InceptionModule, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels, f1, 1)
#         self.conv3 = nn.Conv1d(in_channels, f2, 3, padding=1)
        
#     def forward(self, x):
#         x1 = self.conv1(x)
#         x3 = self.conv3(x)
#         return torch.cat((x1, x3), dim=1)

# class SequenceInception(nn.Module):
#     def __init__(self):
#         super(SequenceInception, self).__init__()
        
#         self.avgpool_initial = nn.AvgPool1d(50)  # Adjusting stride to go from 25000 to 500
        
#         # Feature Extractor
#         self.conv1 = nn.Conv1d(5, 32, 3)  # Adjusting input channels to 5
#         self.conv2 = nn.Conv1d(32, 16, 3)
#         self.incept1 = InceptionModule(16, 16, 16)
#         self.incept2 = InceptionModule(32, 16, 16)
#         self.incept3 = InceptionModule(32, 8, 8)
#         self.incept4 = InceptionModule(16, 8, 8)
        
#         # Classifier
#         self.flatten = nn.Flatten()
#         self.dropout1 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(32, 32)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(32, 16)
#         self.dropout3 = nn.Dropout(0.5)
#         self.fc3 = nn.Linear(16, 1)
        
#     def forward(self, x):
#         x = self.avgpool_initial(x)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = self.avgpool_initial(x)  
#         x = self.incept1(x)
#         x = self.incept2(x)
#         x = F.max_pool1d(x, 2)
#         x = self.incept3(x)
#         x = self.incept4(x)
#         x = F.max_pool1d(x, 2)
#         x = self.flatten(x)
#         x = self.dropout1(x)
#         x = F.relu(self.fc1(x))
#         x = self.dropout2(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout3(x)
#         x = torch.sigmoid(self.fc3(x))
#         return x.squeeze(1)
