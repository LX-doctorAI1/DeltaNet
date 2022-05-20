import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence

from Models.SubLayers import MultiHeadAttention


def init_weight(f):
    init.kaiming_uniform_(f.weight, mode='fan_in')
    f.bias.data.fill_(0)
    return f


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class EncoderCNN(nn.Module):
    """Encoder image"""

    def __init__(self, embed_size, hidden_size, N):
        super(EncoderCNN, self).__init__()
        self.N = N
        self.d_model = hidden_size

        # ResNet-152 backend
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-2]  # delete the last fc layer and avg pool.
        resnet_conv = nn.Sequential(*modules)  # last conv feature

        self.resnet_conv = resnet_conv
        self.avgpool = nn.AvgPool2d(7)
        self.affine_as = get_clones(nn.Linear(2048, hidden_size), N + 1)

        # Dropout before affine transformation
        self.dropout = nn.Dropout(0.5)

        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        for i in range(self.N + 1):
            self.affine_as[i] = init_weight(self.affine_as[i])

    def forward(self, images):
        '''
        Input: images
        Output: V=[v_1, ..., v_n]
        '''
        bs, n, c, h, w = images.size()
        V = torch.zeros((bs, self.N + 1, 49, self.d_model))
        if torch.cuda.is_available():
            V = V.cuda()

        for i in range(self.N + 1):
            # Last conv layer feature map
            # previous image feature: bs x feature_size x 7 x 7
            A = self.resnet_conv(images[:, i])
            # bs x 49 x feature_size
            feat = A.view(bs, A.size(1), -1).transpose(1, 2)
            # bs x 49 x d_model
            feat = F.relu(self.affine_as[i](self.dropout(feat)))
            # bs x (N+1) x 49 x d_model
            V[:, i] = feat

        return V


# Caption Decoder
class Decoder(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, N=2):
        super(Decoder, self).__init__()

        self.affine_va = nn.Linear(hidden_size, embed_size)

        # word embedding
        self.caption_embed = nn.Embedding(vocab_size, embed_size)

        # LSTM decoder
        self.LSTM = nn.LSTM(embed_size * 2, hidden_size, 1, batch_first=True)

        # Save hidden_size for hidden variable
        self.hidden_size = hidden_size

        # Attention Block
        self.attention = MultiHeadAttention(heads=8, d_model=hidden_size)

        # Final Caption generator
        self.mlp = nn.Linear(hidden_size * 2, vocab_size)

        self.dropout = nn.Dropout(0.5)
        self.init_weights()

    def init_weights(self):
        init.kaiming_uniform_(self.affine_va.weight, mode='fan_in')
        self.affine_va.bias.data.fill_(0)
        init.kaiming_normal_(self.mlp.weight, mode='fan_in')
        self.mlp.bias.data.fill_(0)

    def forward(self, V, T, captions, basic_model, states=None):
        # Word Embedding, bs x len x d_model
        embeddings = self.caption_embed(captions)

        # bs x 49 x d_model
        _, curr_vf = V[:, 0], V[:, -1]
        # bs x 1 x d_model
        v_a = torch.mean(curr_vf, dim=1, keepdim=True)
        # bs x len x d_model*2
        x = torch.cat((embeddings, v_a.expand_as(embeddings)), dim=2)
        # Recurrent Block
        hiddens, states = self.LSTM(x, states)

        ctx = self.attention(hiddens, curr_vf, curr_vf)
        output = torch.cat((hiddens, ctx), dim=2)
        # Final score along vocabulary
        # bs x len x vocab_size
        scores = self.mlp(self.dropout(output))

        # Return states for Caption Sampling purpose
        return scores


class EncoderTXT(nn.Module):
    """encode conditional report"""

    def __init__(self, vocab_size, embed_size, hidden_size, N):
        super(EncoderTXT, self).__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.BiLSTM = nn.LSTM(embed_size, hidden_size // 2, num_layers=2, bidirectional=True)

    def forward(self, input):
        # encode
        # embedding: bs x N x tlen -> bs x N x tlen x d_model
        embed_report = self.embed(input)

        bs, N, tlen, d_model = embed_report.size()
        align_report = torch.zeros((bs, self.N, tlen, d_model))
        if torch.cuda.is_available():
            align_report = align_report.cuda()

        for i in range(self.N):
            #  bs x tlen x d_model
            align_report_, _ = self.BiLSTM(embed_report[:, i])
            align_report[:, i] = align_report_

        return align_report


# Whole Architecture with Image Encoder and Caption decoder
class Encoder2Decoder(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, N):
        super(Encoder2Decoder, self).__init__()

        # Image CNN encoder
        self.encoder_image = EncoderCNN(embed_size, hidden_size, N)

        # Concept encoder
        self.encoder_concept = EncoderTXT(vocab_size, embed_size, hidden_size, N)

        # Caption Decoder
        self.decoder = Decoder(embed_size, vocab_size, hidden_size, N)

        assert embed_size == hidden_size, "The values of embed_size and hidden_size should be equal."

    def forward(self, images, captions, image_concepts, lengths, basic_model):
        # imag -> V : bs x C x H x W -> bs x 2 x 49 x d_model
        # concept -> T: bs x tlen -> bs x tlen x d_model
        V = self.encoder_image(images)
        T = None

        # Language Modeling on word prediction
        # bs x len x vocab_size
        scores = self.decoder(V, T, captions, basic_model)

        # Pack it to make criterion calculation more efficient
        packed_scores = pack_padded_sequence(scores, lengths, batch_first=True)

        return packed_scores