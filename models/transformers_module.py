import math
from typing import Optional, List
import torch
from torch import nn


#add zeros to tensors to augment its len
def padTensor(t: torch.tensor, targetLen: int) -> torch.tensor:
    """
    Pads a tensor with zeros along the first dimension to reach the target length.

    This function augments the length of the input tensor by appending zeros to it. It's useful for
    ensuring that all tensors in a batch have the same length, which is often required for batch processing.

    Args:
        t (torch.Tensor): The input tensor to be padded with shape `(original_length, dim)`.
        targetLen (int): The desired length after padding.

    Returns:
        torch.Tensor: The padded tensor with shape `(targetLen, dim)`.

    """
    oriLen, dim = t.size()
    return torch.cat((t, torch.zeros(targetLen - oriLen, dim).to(t.device)), dim=0)



class WrappedTransformerEncoder(nn.Module):

    
    """
    A wrapped Transformer Encoder module with optional classification token prepending and padding support.

    This module encapsulates PyTorch's `TransformerEncoder`, providing additional functionality such as
    prepending a classification token and handling variable-length input sequences with padding.

    Args:
        dim (int): The number of expected features in the encoder input (model dimension).
        num_layers (int): The number of `TransformerEncoderLayer` layers in the encoder.
        num_heads (int): The number of heads in the multihead attention models.

    Attributes:
        dim (int): The model dimension.
        encoder (nn.TransformerEncoder): The Transformer encoder composed of multiple layers.
        cls_emb (nn.Embedding): Embedding layer for the classification token.
    """

    def __init__(self, dim, num_layers, num_heads):
        super(WrappedTransformerEncoder, self).__init__()
        self.dim = dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=dim)

    def prepend_cls(self, inputs):
        """
        Prepends a classification token embedding to each sequence in the batch.

        This method adds a learnable classification token (`CLS`) at the beginning of each input sequence,
        which can be used to aggregate information for classification tasks.

        Args:
            inputs (torch.Tensor): Input tensor with shape `(batch_size, seq_length, dim)`.

        Returns:
            torch.Tensor: Tensor with the classification token prepended, shape `(batch_size, seq_length + 1, dim)`.
        """
        
        index = torch.LongTensor([0]).to(device=inputs.device)
        
        cls_emb = self.cls_emb(index)
       
        cls_emb = cls_emb.expand(inputs.size(0), 1, self.dim)
      
        outputs = torch.cat((cls_emb, inputs), dim=1)
       
        return outputs

    def forward(self, inputs: torch.Tensor, lens: Optional[List[int]] = None, get_cls: Optional[bool] = False):
        """
        Processes the input sequence through the model, optionally prepending a classification token
        and handling variable sequence lengths with padding.

        Args:
            inputs (torch.Tensor): Input tensor with shape `(total_seq_length, dim)` or `(batch_size, seq_length, dim)`.
            lens (Optional[List[int]]): List of sequence lengths for each batch element. If provided, the input will 
                be split, padded, and a mask will be created to handle variable lengths. Default is `None`.
            get_cls (Optional[bool]): If `True`, a classification token will be prepended to each sequence, 
                and only the output of this token will be returned. Default is `False`.

        Returns:
            torch.Tensor: 
                - If `get_cls` is `True`: The output corresponding to the classification token, with shape `(batch_size, dim)`.
                - Otherwise: The processed sequence without the classification token, with shape `(batch_size, seq_length, dim)`.
        """
        if lens is not None:
            max_len = max(lens)

            mask = [([False] * (l + int(get_cls)) + [True] * (max_len - l)) for l in lens]
            mask = torch.tensor(mask).to(device=inputs.device)

            inputs = list(inputs.split(lens, dim=0))
            inputs = [padTensor(inp, max_len) for inp in inputs]
            inputs = torch.stack(inputs, dim=0)
            # print(inputs.shape)
        else:
            mask = None

        if get_cls:
            inputs = self.prepend_cls(inputs)
            # print(inputs)

        inputs = inputs.permute(1, 0, 2)
        # print("input shape")  ##
        # print(inputs.shape)  ##
        inputs = self.encoder(src=inputs, src_key_padding_mask=mask) # (seq_len, bs, dim)

        if get_cls:
            return inputs[0]

        return inputs[1:].permute(1, 0, 2)