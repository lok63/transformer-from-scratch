"""
1. Load the dataset: You will need a dataset for training and testing the model. You can use existing datasets such as
the WMT14 English-German translation dataset or the COCO image captioning dataset, or create your own dataset.

2. Preprocess the data: Preprocessing involves cleaning and normalizing the data, as well as converting it into numerical
form that can be used as input to the model. You may need to tokenize the text, convert words to their embeddings,
and perform data augmentation.

3. Build the model: The Transformer model consists of an encoder and a decoder, each of which is composed of multiple
layers of self-attention and feed-forward neural networks. You will need to implement these layers using PyTorch,
TensorFlow or another deep learning framework. You may also need to implement other components such as positional
encodings and residual connections.

4. Train the model: Train the model on the preprocessed dataset using a suitable optimizer and loss function.
You can use techniques such as gradient clipping and learning rate scheduling to improve the training process.
Monitor the training process using metrics such as loss and accuracy.

5. Evaluate the model: Once the model is trained, evaluate its performance on a separate test dataset using metrics
such as BLEU score or perplexity. You may also want to visualize the attention maps to see which parts of the input
the model is attending to.
"""

import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, dropout=0.1):
        super(Transformer, self).__init__()

        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.embedding_encoder = nn.Embedding(input_vocab_size, d_model)
        self.embedding_decoder = nn.Embedding(target_vocab_size, d_model)

        self.linear = nn.Linear(d_model, target_vocab_size)

    def forward(self, input_seq, target_seq, input_mask=None, target_mask=None):
        enc_output = self.embedding_encoder(input_seq)
        dec_output = self.embedding_decoder(target_seq)

        for layer in self.encoder:
            enc_output = layer(enc_output, input_mask)

        for layer in self.decoder:
            dec_output = layer(dec_output, enc_output, input_mask, target_mask)

        output = self.linear(dec_output)

        return output


class EncoderLayer(nn.Module):
    """
    1. Define the self-attention layer: The self-attention layer takes in an input tensor, computes the dot product of
    the input with its own transpose, applies a softmax activation to obtain attention scores, and computes a weighted
    sum of the input using the attention scores. You can define the self-attention layer using the nn.MultiheadAttention
    module provided by PyTorch.

    2. Define the feed-forward layer: The feed-forward layer takes in the output of the self-attention layer, applies
    a non-linear activation function (such as ReLU), and computes a linear transformation to obtain the final output.
    You can define the feed-forward layer using the nn.Sequential module provided by PyTorch.

    3. Define the residual connection and layer normalization: The residual connection adds the input tensor to the
    output tensor of the feed-forward layer, while the layer normalization normalizes the output tensor along the feature
    dimension. You can define the residual connection and layer normalization using the nn.LayerNorm module provided by PyTorch.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Self-attention layer
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        attn_output = self.dropout(attn_output)
        # Residual connection and layer normalization
        x = self.norm1(x + attn_output)

        # Feed-forward layer
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        # Residual connection and layer normalization
        x = self.norm2(x + ff_output)

        return x


"""
1. Define the self-attention layer: The self-attention layer in the decoder takes in the output of the previous decoder 
layer (i.e., the "query" tensor), computes the dot product of the query tensor with the "key" and "value" tensors 
(which are the outputs of the encoder self-attention layer), applies a softmax activation to obtain attention scores, 
and computes a weighted sum of the "value" tensor using the attention scores. You can define the self-attention layer 
using the nn.MultiheadAttention module provided by PyTorch.

2. Define the encoder-decoder attention layer: The encoder-decoder attention layer in the decoder takes in the output 
of the self-attention layer in the decoder (i.e., the "query" tensor), computes the dot product of the query tensor 
with the "key" and "value" tensors (which are the outputs of the encoder self-attention layer), applies a softmax 
activation to obtain attention scores, and computes a weighted sum of the "value" tensor using the attention scores. 
You can define the encoder-decoder attention layer using the nn.MultiheadAttention module provided by PyTorch.

3. Define the feed-forward layer: The feed-forward layer in the decoder takes in the output of the encoder-decoder 
attention layer, applies a non-linear activation function (such as ReLU), and computes a linear transformation to 
obtain the final output. You can define the feed-forward layer using the nn.Sequential module provided by PyTorch.

4. Define the residual connection and layer normalization: The residual connection adds the input tensor to the output 
tensor of the feed-forward layer, while the layer normalization normalizes the output tensor along the feature dimension. 
You can define the residual connection and layer normalization using the nn.LayerNorm module provided by PyTorch.

"""

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, num_heads)
        self.enc_dec_attn = nn.MultiheadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, self_mask=None, enc_mask=None):
        # Self-attention layer
        attn_output, _ = self.self_attn(x, x, x, attn_mask=self_mask)
        attn_output = self.dropout(attn_output)
        # Residual connection and layer normalization
        x = self.norm1(x + attn_output)

        # Encoder-decoder attention layer
        enc_dec_attn_output, _ = self.enc_dec_attn(x, enc_output, enc_output, attn_mask=enc_mask)
        enc_dec_attn_output = self.dropout(enc_dec_attn_output)
        # Residual connection and layer normalization
        x = self.norm2(x + enc_dec_attn_output)

        # Feed-forward layer
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        # Residual connection and layer normalization
        x = self.norm3(x + ff_output)

        return x


