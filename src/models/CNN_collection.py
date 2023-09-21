import torch
import torch.nn as nn

class BasicCNN(nn.Module):
    def __init__(self,
                alt_dropout_rate: float=0.1,
                fc_dropout_rate: float=0.5,
                activation_fn: str='ReLU',
                batchnorm: bool=True,
                fc_num: int=1,
                kernel_size: tuple=(3,3,3),
                num_inception_layers: int = 5,
                out_channels: int = 16,
                kernel_size_b1: int = 3,
                kernel_size_b2: int = 5,
                keep_b3 = True,
                keep_b4 = True,
                input_size=25000,
                hidden_size_lstm=64,
                num_layers_lstm=1,
                num_classes=1,
                pad_pack: bool=False,
                embedding_dim=None,
                vocab_size=5):
        super(BasicCNN, self).__init__()
        
        padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
        
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=16,
                               kernel_size=kernel_size[0],
                               stride=1,
                               padding=padding[0])
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32,
                               kernel_size=kernel_size[1], stride=1, padding=padding[1])
        if kernel_size[2] > 0:
            self.conv3 = nn.Conv1d(in_channels=32, out_channels=32,
                                kernel_size=kernel_size[2], stride=1, padding=padding[2])

        self.fc_pre = nn.Linear(32, 32)
        self.fc_opt1 = nn.Linear(32, 16)
        self.fc_opt2 = nn.Linear(16, 2)
        self.fc = nn.Linear(32, 2)
        self.activation_fn = getattr(nn, activation_fn)()
        self.dropout_conv = nn.Dropout(alt_dropout_rate)
        self.dropout = nn.Dropout(fc_dropout_rate)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.batchnorm = batchnorm
        self.fc_num = fc_num

        self.feature_maps = {}
        self._register_hooks()

        # placeholder for the gradients
        self.gradients = None

    def _register_hooks(self):
        for name, layer in self.named_children():
            if isinstance(layer, nn.Conv1d):  # Adjust as necessary to match the layers you're interested in
                layer.register_forward_hook(self._hook_fn(name))

    def _hook_fn(self, name):
        def hook(module, input, output):
            self.feature_maps[name] = output
        return hook
    
    # GRAD CAM - hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        # Conv1
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    # --

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
        
        # GRAD CAM - Register hook (if conv2 is last)
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        
        if self.batchnorm:
            x = self.bn2(x)
        x = self.dropout_conv(x)

        # 1. optional Conv
        if self.kernel_size[2] > 0:
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
        x = torch.softmax(x, dim=1)
        return x


class BasicInception(nn.Module):
    def __init__(self,
                alt_dropout_rate: float=0.1,
                fc_dropout_rate: float=0.5,
                activation_fn: str='ReLU',
                batchnorm: bool=True,
                fc_num: int=1,
                kernel_size: tuple=(3,3,3),
                num_inception_layers: int = 5,
                out_channels: int = 16,
                kernel_size_b1: int = 3,
                kernel_size_b2: int = 5,
                keep_b3 = True,
                keep_b4 = True,
                input_size=25000,
                hidden_size_lstm=64,
                num_layers_lstm=1,
                num_classes=1,
                pad_pack: bool=False,
                embedding_dim=None,
                vocab_size=5):
        super(BasicInception, self).__init__()
        
        self.activation_fn = getattr(nn, activation_fn)()
        self.num_inception_layers = num_inception_layers
        self.inception1 = InceptionModule(in_channels=5, out_channels=out_channels, kernel_size_b1=kernel_size_b1, kernel_size_b2=kernel_size_b2, keep_b3=keep_b3, keep_b4=keep_b3)

        inception_out_ch = (2+int(keep_b3)*2)*out_channels
        self.inception_extra = InceptionModule(in_channels=inception_out_ch, out_channels=out_channels, kernel_size_b1=kernel_size_b1, kernel_size_b2=kernel_size_b2, keep_b3=keep_b3, keep_b4=keep_b3)

        self.fc_pre = nn.Linear(inception_out_ch, inception_out_ch)
        self.fc_opt1 = nn.Linear(inception_out_ch, 16)
        self.fc_opt2 = nn.Linear(16, 1)
        self.fc = nn.Linear(inception_out_ch, 1)
        
        self.dropout_conv = nn.Dropout(alt_dropout_rate)
        self.dropout = nn.Dropout(fc_dropout_rate)
        self.batchnorm = batchnorm
        self.fc_num = fc_num

    def forward(self, x):
        x = self.inception1(x)
        if self.num_inception_layers > 1:
            for i in range(self.num_inception_layers - 1):
                x = self.inception_extra(x)

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
        x = torch.softmax(x, dim=1)
        return x



class InceptionModule(nn.Module):
    def __init__(self, in_channels: int = 5, out_channels: int = 16, kernel_size_b1: int = 3, kernel_size_b2: int = 5, keep_b3 = True, keep_b4 = True):
        super(InceptionModule, self).__init__()

        self.keep_b3 = keep_b3
        self.keep_b4 = keep_b4

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

        self.branch4 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        outputs = [branch1_out, branch2_out]
        if self.keep_b3:
            branch3_out = self.branch3(x)
            outputs.append(branch3_out)
        if self.keep_b4:
            branch4_out = self.branch4(x)
            outputs.append(branch4_out)
        return torch.cat(outputs, 1)
