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
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embed_size)
        # Important: batch_first=True, since we deliver the input in batches 
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # reshape the features
        features = features.view(len(features), 1, -1)

        # embed the captions
        embeddings = self.embed(captions[:, :-1])

        # concatenate the image features with the embeddings
        inputs = torch.cat((features, embeddings), 1)
        
        # pass through the lstm
        out, hidden = self.lstm(inputs)
        
        # fully connnected layer
        out = self.fc(out)

        return out

    def sample(self, features, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        # In the first step the input consists only of the image features.
        # Later the input is "extended" by the words that the LSTM predicts.
        inputs = features
        
        for i in range(max_len):
            
            # pass through the lstm
            out, hidden = self.lstm(inputs)
            
            # pass through the fully connected layer
            out = self.fc(out)
            
            # get the captions with the highest value
            captions = out.argmax(2)
            
            # now prepare the input for the next iteration
            # embed the captions
            embeddings = self.embed(captions)
            
            # input for the next iteration: concatenate the image features with the embeddings
            inputs = torch.cat((features, embeddings), 1)
          
        return captions.tolist()[0]