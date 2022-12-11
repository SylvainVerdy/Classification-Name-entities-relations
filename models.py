# coding my network
import torch.optim as optim
import torch
import math 
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def ScaledDotProductAttention(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn = torch.matmul(q , k.transpose(-2, -1)) #matmul
    attn = attn / math.sqrt(d_k) # scale
    if mask is not None:
        print(attn.shape)
        attn = attn.masked_fill(mask == 0, -9e15)

    attention = F.softmax(attn, dim=-1) # softmax
    output = torch.matmul(attention, v) #matmul
    return output, attention


class PositionalEncoding(torch.nn.Module):
  """
  Notes
  -----
  Not from myself here.
  Attributes
  ----------

  """
  def __init__(self, d_model, dropout=0.1, max_len=5000):
      """
      d_model: size output 
      """
      super().__init__()
      self.dropout = torch.nn.Dropout(p=dropout)

      position = torch.arange(max_len).unsqueeze(1)
      div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
      pe = torch.zeros(max_len, 1, d_model)
      pe[:, 0, 0::2] = torch.sin(position * div_term)
      pe[:, 0, 1::2] = torch.cos(position * div_term)
      self.register_buffer('pe', pe)

  def forward(self, x):
      x = x + self.pe[:x.size(0)]
      return self.dropout(x)


class PositionWiseFeedForward(torch.nn.Module):
  """
  Notes
  -----

  Attributes
  ----------

  fully connected layers
  dim_ffn: hidden nodes
  input_dim: e
  """
  def __init__(self, input_dim, dim_ffn, dropout = 0.1):
    super().__init__()

    self.fc1 = torch.nn.Linear(input_dim, dim_ffn) 
    self.fc2 = torch.nn.Linear(dim_ffn, input_dim)
    self.relu = torch.nn.ReLU(inplace = True)
    self.dropout = torch.nn.Dropout(dropout)
    self.layer_norm = torch.nn.LayerNorm(input_dim, eps=1e-6)

  def forward(self, x):
    residual_connection = x
    x = self.relu(self.fc1(x))
    x = self.fc2(x)
    x = self.dropout(x)
    x += residual_connection # add 
    x = self.layer_norm(x) # norm

    return x


class MultiHeadAttention(torch.nn.Module):
  """
  Notes
  -----

  Attributes
  ----------
  n_head: Number of attention's head
  d_model: dimension input embedding
  d_k: dimension for the key
  d_v: dimension for the value
  return_attention: get the output of the attention matrix
  """
  def __init__(self, n_heads, d_model, d_k, d_v, dropout=0.1, return_attention=None):
    super().__init__()
    
    self.d_model = d_model
    self.n_heads = n_heads
    self.d_k = d_k
    self.d_v = d_v
    self.head_dim = self.d_model // n_heads
    assert self.head_dim * n_heads == self.d_model, "d_model dim must be divisible by num_heads"

    self.return_attention = return_attention

    self.w_qs = torch.nn.Linear(d_model, n_heads * d_k, bias=False)
    self.w_ks = torch.nn.Linear(d_model, n_heads * d_k, bias=False)
    self.w_vs = torch.nn.Linear(d_model, n_heads * d_v, bias=False)


    self.fc = torch.nn.Linear(n_heads * d_v, d_model, bias=False)
    self.dropout = torch.nn.Dropout(dropout)
    self.layer_norm = torch.nn.LayerNorm(d_model, eps=1e-6)


    self._reset_parameters()

  def _reset_parameters(self):
      torch.nn.init.xavier_uniform_(self.w_qs.weight)
      torch.nn.init.xavier_uniform_(self.w_ks.weight)
      torch.nn.init.xavier_uniform_(self.w_vs.weight)


  def forward(self,  q, k, v, mask=None):

    residual_connection = q
    batch, seq_length, dims = q.size()
    d_q, d_k, d_v = q.size(-1), k.size(-1), v.size(-1)
    self.batch, self.seq_q, self.seq_k, self.seq_v = q.size(0), q.size(1), k.size(1), v.size(1)
    q = self.w_qs(q).view(batch, self.seq_q, self.n_heads, d_q)
    k = self.w_ks(k).view(batch, self.seq_k, self.n_heads, d_k)
    v = self.w_vs(v).view(batch, self.seq_v, self.n_heads, d_v) # [batch, seq, head, dim_v]

    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # [batch, head, seq, dim_v]
    
    output, attention = ScaledDotProductAttention(q,k,v, mask=mask)
    output = output.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims == k,v dim]
    output = output.reshape(batch, seq_length, -1)# [Batch, SeqLen, Head, Dims ==> Batch , Seq ,  head * dim]
    output = self.dropout(self.fc(output))
    output += residual_connection # add 
    output = self.layer_norm(output) # norm
    if self.return_attention:
      return output, attention
    else:
      return output


class TransformerEncoderLayer(torch.nn.Module):
  """
  Notes
  -----

  Attributes
  ----------

  """

  def __init__(self, input_dim, dim_ffn, nheads, k_dim, v_dim, dropout = 0.0):
    super().__init__()

    self.mha = MultiHeadAttention(n_heads=nheads, d_model=input_dim, d_k=k_dim, d_v=v_dim, dropout=dropout, return_attention=True)
    self.pwffn = PositionWiseFeedForward(input_dim=input_dim, dim_ffn=dim_ffn, dropout=dropout)
  
  def forward(self, x, mask_attention=None):

    output_attention, attentions_map = self.mha(x,x,x, mask_attention)
    output = self.pwffn(output_attention)
    return output, attentions_map 


class TransformerEncoder(torch.nn.Module):
  """
  Notes
  -----

  Attributes
  ----------

  """
  def __init__(self, d_model=512, d_ffn=2048, n_heads=1, n_layers=1,  k_dim=512, v_dim=512, dropout=0.1):
    super().__init__()
    self.layers_encoder_tf = torch.nn.ModuleList([
        TransformerEncoderLayer(input_dim=d_model, dim_ffn=d_ffn, nheads=n_heads, k_dim=k_dim, v_dim=v_dim, dropout=dropout) for layer in range(n_layers)])

  def forward(self, x, mask=None):
        hidden_states = []
        for l in self.layers_encoder_tf:
            x, attn = l(x, mask)
            hidden_states.append(x)
        return x, hidden_states


class Model(torch.nn.Module):
  """
  Notes
  -----

  Attributes
  ----------

  """
  def __init__(self, n_classes=19, layers_encoder=4, positional_encoding = False):
    super().__init__()
    self.positional_encoding  = positional_encoding
    self.layers = layers_encoder
    self.dim = 512
    self.pe = PositionalEncoding(d_model=512, dropout=0.1)
    self.emb = torch.nn.Embedding(num_embeddings=30522, embedding_dim=512) # tokenizer.get_vocab_size(), embedding_dim=512)
    encoder_layer = torch.nn.TransformerEncoderLayer(d_model=512, nhead=2)
    #self.encoder = TransformerEncoder(d_model=512, d_ffn=2048, n_heads=2, n_layers=self.layers, dropout=0.1)
    self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=4, norm=torch.nn.LayerNorm(512, eps=1e-6))
    
    self.layer_norm = torch.nn.LayerNorm(self.dim, eps=1e-6)
    
    self.dense = torch.nn.Linear(self.dim* self.layers, self.dim)
    self.activation = torch.nn.Tanh()
    
    self.fc = torch.nn.Linear(self.dim , n_classes)
    
  def forward(self, x, mask=None):
    tx = x
    x = self.emb(x)
    if self.positional_encoding:
      x = self.pe(x)
    output = self.encoder(x)
    output = self.fc(output)
    output = output[:, 0, :] # torch.mean(output, 1)
    return output 
    
    # output, hidden_states = self.encoder(x, mask) # layers_hstates, batch, seq, dim
    # hidden_states = torch.stack((hidden_states))
    # hidden_states = hidden_states.permute(1,2,0,3).reshape(output.size(0), output.size(1), self.dim* self.layers) # batch, seq_len,dim* layers
    # #print(output.shape)
    # first_token_tensor = hidden_states[:, 0, :]
    # pooled_output = self.dense(first_token_tensor)
    # output = self.activation(pooled_output)
    # output = self.fc(output[:, 0, :])
    
    #print(output.shape)
    # return output 
    

class BertForSequenceClassification(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        #self.model.resize_token_embeddings(len(tokenizer))

        self.fc = torch.nn.Linear(768, 19)
        
    def forward(self, input_ids):
        outputs = self.model(input_ids, output_hidden_states=True)
        output = self.fc(outputs[0][:,0,:])
        return output

