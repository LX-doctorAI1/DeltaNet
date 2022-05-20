''' Define the Layers '''
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.SubLayers import FeedForward, Norm, MultiHeadAttention


class ConditionLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff=2048, dropout=0.1):
        super(ConditionLayer, self).__init__()
        self.condi_att = MultiHeadAttention(n_head, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff)

        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, e_outputs):
        x2 = self.norm_1(x)
        x = self.dropout_1(self.condi_att(x2, e_outputs, e_outputs))
        x2 = self.norm_2(x)
        x = self.dropout_2(self.ff(x2))

        return x


# Attention Block for C_t calculation
class AttentionBlock(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionBlock, self).__init__()

        self.affine_x = nn.Linear(hidden_size, 49, bias=False)  # W_x
        self.affine_h = nn.Linear(hidden_size, 49, bias=False)  # W_g
        self.affine_alpha = nn.Linear(49, 1, bias=False)  # w_h

        self.dropout = nn.Dropout(0.5)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        nn.init.xavier_uniform_(self.affine_x.weight)
        nn.init.xavier_uniform_(self.affine_h.weight)
        nn.init.xavier_uniform_(self.affine_alpha.weight)

    def forward(self, X, h_t):
        '''
        Input: X=[x_1, x_2, ... x_k], h_t from LSTM
        Output: c_t, attentive feature map
        '''

        # W_x * X + W_g * h_t * 1^T
        content_x = self.affine_x(self.dropout(X)).unsqueeze(1) \
                    + self.affine_h(self.dropout(h_t)).unsqueeze(2)

        # alpha_t = softmax(W_h * tanh( content_x ))
        z_t = self.affine_alpha(self.dropout(torch.tanh(content_x))).squeeze(3)
        alpha_t = F.softmax(z_t.view(-1, z_t.size(2)), dim=-1).view(z_t.size(0), z_t.size(1), -1)

        # Construct attention_context_t: B x seq x hidden_size
        attention_context_t = torch.bmm(alpha_t, X).squeeze(2)

        return attention_context_t


class GateLayer(nn.Module):
    def __init__(self, hidden_size):
        super(GateLayer, self).__init__()
        self.txt_att = MultiHeadAttention(heads=8, d_model=hidden_size)
        self.vis_att = MultiHeadAttention(heads=8, d_model=hidden_size)
        self.mlp = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size),
                                  nn.Tanh(),
                                  nn.Linear(hidden_size, 1))

    def forward(self, I, I_, T_):
        bs, fs, d_model = I.size()
        # bs x 49 x d_model
        ve_ = self.vis_att(I, I_, I_)
        te_ = self.txt_att(I, T_, T_)
        # bs x 49 x (3*d_model)
        g_cat = torch.cat((te_, ve_, (I - I_)), dim=2)
        # bs x (3*d_model)
        g_ = torch.mean(g_cat, dim=1)
        # bs x 1
        g = self.mlp(g_)
        g = g + torch.bmm(te_.view(bs, 1, -1), ve_.view(bs, -1, 1)).squeeze(2)

        return g
