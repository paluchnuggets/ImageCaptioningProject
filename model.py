import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        # store important shapes.sizes
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # embedding layer which converts vectors to embed_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer which takes embedded vector as input
        # and outputs hidden state of size hidden_size
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        
        # Fully-connected layer maps LSTM output into vocab_size 
        self.hidden2vocab = nn.Linear(hidden_size, vocab_size)
   
    
    def forward(self, features, captions):
        # first we have to pre-process features and captions in order to concatenate it into one tensor
        
        # from (batch_size, caption_len_in_current_batch) -> (batch_size, cap_len_cur_btch-1, embed_size)
        # last column is ommited because we don't want our model to predict next word when <end> is input
        captions = self.embed(captions[:,:-1]) 
        # from (batch_size, embed_size) -> (batch_size, 1, embed_size)
        features = features.unsqueeze(dim=1)
        
        # now we can concat those tensors in order to obtain input tensor
        # dim=1 means that we concatenate along horizontal axis
        # shape: (batch_size, cap_len_cur_btch -1 + 1, embed_size)
        inputs = torch.cat((features, captions), dim=1)
        
        # having inputs concatenated we process them through our network
        # from inputs.shape -> (batch_size, cap_len_cur_btch, hidden_size)
        lstm_output, _ = self.lstm(inputs)
        
        # then fully-connected one
        # from lstm_output.shape -> (batch_size, cap_len_cur_btch, vocab_size)
        outputs = self.hidden2vocab(lstm_output)
        
        return outputs
    

    def sample(self, inputs, states=None, max_len=20):
        """
        Accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len).
        
         ------------------------------------------
        Parameters:
        inputs: torch tensor of size (1, 1, embed_size) representing embedded single image
        max_len: lenght of the caption generated for image provided in inputs
        """
        
        # Initialize hidden state
        hidden = (torch.zeros(self.num_layers, 1, self.hidden_size).to(inputs.device),
                  torch.zeros(self.num_layers, 1, self.hidden_size).to(inputs.device))
        
        caption = []
        
        for i in range(max_len):
            lstm_output, hidden = self.lstm(inputs, hidden) # inputs.shape = (1, 1, embed) -> (1, 1, hidden_size)
            outputs = self.hidden2vocab(lstm_output.squeeze(1)) # (1, 1, hiden_size) -> (1, 1, vocab_size)
            
            # first from (1, 1, vocab_size) -> (1, vocab_size) then return index of the most probable token in vocab
            #outputs = outputs.squeeze(1)
            word = outputs.argmax(dim=1)
            # after that we append prediction to the caption list
            # .item() convert torch tensor with that maximum index to integer value
            caption.append(word.item())
            
            # prepare output from timestep t to be an input for timestep t+1
            # from (1, 1) to (1, 1, embed) again
            inputs = self.embed(word.unsqueeze(0))
        
        return caption