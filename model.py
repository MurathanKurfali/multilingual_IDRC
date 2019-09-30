import torch.nn as nn
import torch.nn.functional as F
import torch

class relation_classifier(nn.Module):

    def __init__(self, input_size, output_size=2):
        super(relation_classifier, self).__init__()

        self.layer1 = nn.Linear(input_size, 100)
        self.drop_layer = nn.Dropout(0.3)
        self.layer2 = nn.Linear(100, output_size)
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, arg1, arg2):
        x = torch.cat((arg1, arg2, arg1 - arg2, arg1 * arg2, (arg1 + arg2) / 2), 1)

        x = F.relu(self.layer1(x))
        x = self.drop_layer(x)
        x = F.softmax(self.layer2(x), dim=1)
        return x
