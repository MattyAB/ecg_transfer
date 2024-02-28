import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init


## EarlyReturn is an exception type that is called when we want to return from our model if we aren't calling the whole depth of the model.
class EarlyReturnException(Exception):
    pass

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, kernel_stride=1, mp_size=5, mp_stride=2, dropout=0.1):
        super().__init__()

        self.conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, kernel_stride)
        self.mp_layer = nn.MaxPool1d(mp_size, mp_stride)
        self.dropout_layer = nn.Dropout1d(dropout)

    def forward(self, x):
        x = self.conv_layer(x)
        x = F.relu(x)
        x = self.mp_layer(x)
        x = self.dropout_layer(x)

        return x

class SingleLeadModel(nn.Module):
    def __init__(self, lstm_hidden_size=16, output_size=4):
        super().__init__()

        self.lstm_hidden_size = lstm_hidden_size

        self.conv1 = ConvLayer(1, 8, 20, 1)
        self.conv2 = ConvLayer(8, 16, 12, 1)
        self.conv3 = ConvLayer(16, 32, 8, 1)
        self.conv4 = ConvLayer(32, 32, 4, 1)

        self.lstm1 = nn.LSTM(input_size=32, hidden_size=self.lstm_hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=2*self.lstm_hidden_size, hidden_size=self.lstm_hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_dropout = nn.Dropout1d(0.2)

        self.fc_dropout = nn.Dropout(0.2) 

        self.fullyconnected_layer = nn.Linear(2*self.lstm_hidden_size, output_size)

    def forward(self, x, request=[-1]):
        def savelayer(x):
            nonlocal i

            if i in request:
                output_list.append(x)

                if -1 not in request and i == max(request):
                    raise EarlyReturnException()

            i += 1

        i = 0
        output_list = []

        try:
            x = x.unsqueeze(1)

            ## Convolution Layers

            x = self.conv1.forward(x)
            savelayer(x)
            x = self.conv2.forward(x)
            savelayer(x)
            x = self.conv3.forward(x)
            savelayer(x)
            x = self.conv4.forward(x)
            savelayer(x)


            ## LSTM Layer

            x = x.contiguous().view(x.shape[0], x.shape[2], x.shape[1])
            x, _ = self.lstm1(x)
            savelayer(x)

            x = x.transpose(1, 2)
            x = self.lstm_dropout(x)
            x = x.transpose(1, 2)
                
            x, _ = self.lstm2(x)

            forward_out = x[:, -1, :self.lstm_hidden_size]
            backward_out = x[:, 0, self.lstm_hidden_size:]
            final_out = torch.cat([forward_out, backward_out], dim=1)
            savelayer(final_out)
            

            ## Fully Connected Layer

            x = self.fc_dropout(final_out)
            x = self.fullyconnected_layer(x)

            if -1 in request:
                output_list.append(x)
            return output_list
        
        ## If we have had an EarlyReturn.
        except EarlyReturnException:
            return output_list
        

### Transfer Modelling

class TransferModel(nn.Module):
    def __init__(self, base=None, allow_finetune=False, output_size=4):
        super().__init__()

        ## Save baseline model as a reference point
        self.base_model = SingleLeadModel()
        if base:
            base.seek(0)
            self.base_model.load_state_dict(torch.load(base))
        for param in self.base_model.parameters():
            param.requires_grad = False

        ## Duplication for our constitutent models.
        self.constituent_models = nn.ModuleList()

        for _ in range(12):
            model = SingleLeadModel()
            if base:
                base.seek(0)
                model.load_state_dict(torch.load(base))
            
            if not allow_finetune:
                for param in model.parameters():
                    param.requires_grad = False
                     
            self.constituent_models.append(model)


        self.lstm_hidden_size = self.base_model.lstm_hidden_size

    def forward(self, x, return_request):
        outputs = []

        for i in range(x.shape[2]):
            outputs.append(self.constituent_models[i].forward(x[:,:,i], return_request)[0])

        return outputs
    
    def get_l1_weightdiff(self):
        diff = torch.tensor(0, device=next(self.base_model.parameters()).device, dtype=torch.float32)

        for i in range(12):
            for base_param, const_param in zip(self.base_model.parameters(), self.constituent_models[i].parameters()):
                
                weight_diff = base_param - const_param
                
                diff += weight_diff.abs().sum()

        return diff
    
                
class TransferFCModel(TransferModel):
    def __init__(self, base=None, allow_finetune=True, output_size=4):
        super().__init__(base, allow_finetune, output_size)

        self.fc_layer = nn.Linear(2 * 12 * self.lstm_hidden_size, output_size)
        self.fc_dropout = nn.Dropout(0.5)

    def forward(self, x):
        return_request = [5]
        outputs = super().forward(x, return_request)

        x = torch.cat(outputs, dim=1)
        
        x = self.fc_dropout(x)
        x = self.fc_layer(x)

        return [x]
                
class Transfer1LSTMModel(TransferModel):
    def __init__(self, base=None, allow_finetune=True, output_size=4):
        super().__init__(base, allow_finetune, output_size)

        self.lstm_dropout = nn.Dropout1d(0.5)
        self.lstm_2 = nn.LSTM(input_size=2*12*self.lstm_hidden_size, hidden_size=self.lstm_hidden_size, num_layers=1, batch_first=True, bidirectional=True)

        self.fc_layer = nn.Linear(2 * self.lstm_hidden_size, output_size)
        self.fc_dropout = nn.Dropout(0.5)

    def forward(self, x):
        return_request = [4]
        outputs = super().forward(x, return_request)

        x = torch.cat(outputs, dim=2)
        
        x = self.lstm_dropout(x)
        x, _ = self.lstm_2(x)

        forward_out = x[:, -1, :self.lstm_hidden_size]
        backward_out = x[:, 0, self.lstm_hidden_size:]
        final_out = torch.cat([forward_out, backward_out], dim=1)

        x = self.fc_dropout(final_out)
        x = self.fc_layer(x)

        return [x]
                
### Allowing interaction terms between first and second LSTM layer by adding 'adapters'
class TransferAdaptersLSTMModel(TransferModel):
    def __init__(self, base=None, allow_finetune=True, output_size=4):
        super().__init__(base, allow_finetune, output_size)

        self.lstm_2_list = nn.ModuleList()

        for model in self.constituent_models:
            self.lstm_2_list.append(model.lstm2)

        self.intermediate_fc = nn.Linear(12*32, 12*32)

        init.constant_(self.intermediate_fc.weight, 0)
        init.constant_(self.intermediate_fc.bias, 0)        

        identity_matrix = torch.eye(384)
        self.intermediate_fc.weight.data.copy_(identity_matrix)

        self.fc_layer = nn.Linear(2 * self.lstm_hidden_size * 12, output_size)
        self.fc_dropout = nn.Dropout(0.5)

    def forward(self, x):
        return_request = [4]
        outputs = super().forward(x, return_request)

        x = torch.stack(outputs, dim=3)

        x_flattened = x.view(x.size(0), x.size(1), -1)
        x_transformed = self.intermediate_fc(x_flattened)
        x_transformed = F.relu(x_transformed)
        
        x = x_transformed.view((x_transformed.shape[0], x_transformed.shape[1], 32, 12))

        output_list = []

        for i in range(12):
            lstm_out = self.lstm_2_list[i](x[:,:,:,i])[0]

            forward_out = lstm_out[:, -1, :self.lstm_hidden_size]
            backward_out = lstm_out[:, 0, self.lstm_hidden_size:]
            final_out = torch.cat([forward_out, backward_out], dim=1)

            output_list.append(final_out)

        x = torch.cat(output_list, dim=1)

        x = self.fc_dropout(x)
        x = self.fc_layer(x)

        return [x]
    
    def get_l1_weightdiff(self, weight_factor=30):
        cost = super().get_l1_weightdiff()

        cost += weight_factor * (self.intermediate_fc.weight - torch.eye(384, device=next(self.base_model.parameters()).device)).abs().sum()
        cost += weight_factor * (self.intermediate_fc.bias - torch.zeros(384, device=next(self.base_model.parameters()).device)).abs().sum()

        return cost