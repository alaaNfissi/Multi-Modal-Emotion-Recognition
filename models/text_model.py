from torch import nn
from transformers import AlbertModel

class ALBERT(nn.Module):
    """
    ALBERT Model Wrapper for Text Classification.

    Args:
        feature_dim (int): The dimensionality of the ALBERT model's output features.
        num_classes (int, optional): The number of target classes for classification. Defaults to 6.
        size (str, optional): The size variant of the ALBERT model to load (e.g., 'base', 'large').
            Defaults to 'base'.

    Attributes:
        albert (AlbertModel): The pre-trained ALBERT model.
        text_feature_affine (nn.Linear): A linear layer that maps ALBERT's features to the target
            number of classes.
    """        
    def __init__(self, feature_dim, num_classes=6, size='base'):
        super(ALBERT, self).__init__()
        self.albert = AlbertModel.from_pretrained(f'albert-{size}-v2')
        self.text_feature_affine = nn.Linear(feature_dim, num_classes)



    def forward(self, text):
        """
        Processes the input text through the ALBERT model and extracts the classification token feature.


        Args:
            text (Dict[str, torch.Tensor]): Dictionary containing tokenized input text, typically with keys 
                like 'input_ids', 'attention_mask', etc., matching the expected input format for ALBERT.

        Returns:
            torch.Tensor: The feature corresponding to the `[CLS]` token from the last hidden state, 
                          with shape `(batch_size, hidden_dim)`.
        """
        last_hidden_state = self.albert(**text).last_hidden_state
        cls_feature = last_hidden_state[:,0]
        return cls_feature

        