
import torch
from torch import Tensor, nn
from typing import List , Union
from einops import  rearrange

from transformers import AutoTokenizer , AddedToken
from transformers import PreTrainedTokenizerBase

from typing import Dict, List, Tuple, TypeVar


Hidden = List[Tuple[Tensor, ...]]
T = TypeVar('T')
D = TypeVar('D')

def exists(var : Union[T, None]) -> bool:
    return var is not None

def default(var :Union[T, None], val : D) -> Union[T , D]:
    return var if exists(var) else val

class CausalConv1d(nn.Conv1d):
    """
    Causal convolution layer that ensures the output only depends on previous inputs.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        dilation (int, optional): Dilation rate for the convolution. Defaults to 1.
        groups (int, optional): Number of groups for the convolution. Defaults to 1.
        bias (bool, optional): Whether to include a bias term. Defaults to True.

    
    """
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,dilation=1,groups=1,bias=True):
        self._padding = (kernel_size - 1) * dilation #Ce décalage détermine combien de zéros seront ajoutés à l'entrée pour garantir que la sortie ne dépend que des échantillons précédents dans l'entrée.

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self._padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, inp : Tensor) -> Tensor:
        # Handle the case where input has only two dimensions
        # we expect them to have semantics (batch, channels),
        # so we add the missing dimension manually
        if inp.dim() == 2: inp = rearrange(inp, 'b i -> b 1 i')

        result = super(CausalConv1d, self).forward(inp)
        """
        Dans une convolution causale, les échantillons de rembourrage sont ajoutés à l'entrée pour que la sortie ne dépende que des échantillons précédents dans l'entrée.
        En supprimant les échantillons de rembourrage de la sortie, nous nousassurons que la sortie reste causale et ne dépend que des échantillons passés dans l'entrée.
        """
        if self._padding != 0: return result[..., :-self._padding]
        return result

class BlockLinear(nn.Module):
    """
    A block linear layer that applies a block-diagonal linear transformation to the input.

    Args:
        block_dims (List[Union[int, List[int]]]): List of dimensions for each block.
        bias (bool, optional): Whether to include a bias term. Defaults to False.

    
    """
    def __init__(self,block_dims : List[Union[int, List[int]]],bias : bool = False):
        super(BlockLinear, self).__init__()

        self._blocks = nn.ParameterList([
            nn.Parameter(torch.randn(size, requires_grad=True))
            for size in block_dims
        ])

        self._bias = nn.Parameter(torch.zeros(sum(block_dims))) if bias else None

    def forward(self, inp : Tensor) -> Tensor:
        # Assemble the blocks into a block-diagonal matrix
        full = torch.block_diag(*self._blocks)

        out = torch.matmul(full, inp)

        if self._bias is not None:
            out = out + self._bias

        return out
def enlarge_as(src : Tensor, other : Tensor) -> Tensor:
    '''
        Add sufficient number of singleton dimensions
        to tensor a **to the right** so to match the
        shape of tensor b. NOTE that simple broadcasting
        works in the opposite direction.
    '''
    return rearrange(src, f'... -> ...{" 1" * (other.dim() - src.dim())}').contiguous()





class TokenizerWrapper:
    '''
    A wrapper class for tokenizers.

    This class provides a convenient way to initialize and access tokenizers for various pretrained models.

    Args:
        pretrained_model_name_or_path (str): The name or path of the pretrained model.

    Attributes:
        tokenizer (PreTrainedTokenizerBase): The tokenizer object.

    Methods:
        get_tokenizer: Returns the tokenizer object.

    Example:
        > tokenizer = TokenizerWrapper('bert-base-uncased')
        > tokenizer.get_tokenizer()
        <transformers.tokenization_bert.BertTokenizer object at 0x7f0a5e7a3e10>
    '''

    def __init__(self,pretrained_model_name_or_path: str,special_tokens: Dict[str,  Union[str , AddedToken]] = {}):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer.add_special_tokens(special_tokens)


    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        '''
        Returns the tokenizer object.

        Returns:
            PreTrainedTokenizerBase: The tokenizer object.
        '''
        return self.tokenizer
