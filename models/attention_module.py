import torch
import torch.nn as nn


class CrossModalAttentionT(nn.Module):
    """
    Cross-Modal Attention Layer for Text attending to Visual-Audio Features.


    Parameters:
        text_dim (int): Dimension of the text input features.
        va_dim (int): Dimension of the visual/audio input features.
        num_heads (int): Number of attention heads for the multi-head attention mechanism. Default is 4.
        

    """    

    def __init__(self, text_dim, va_dim, num_heads):
        super(CrossModalAttentionT, self).__init__()
        self.text_dim = text_dim
        self.va_dim = va_dim
        self.num_heads = num_heads

        # Linear layers to transform visual features to the same dimension as text features
        self.visual_to_text_dim = nn.Linear(va_dim, text_dim)
        self.attention = nn.MultiheadAttention(embed_dim=text_dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(text_dim)
        
        # Linear layer to project the output back to the desired dimension
        self.output_projection = nn.Linear(text_dim, va_dim)

    def forward(self, query, key, value):
        """
        Forward pass of the CrossModalAttentionVA module. Applies multi-head attention 
        where text features attend to visual/audio features.

        Arguments:
            query (Tensor): Visual/audio features, shape (batch_size, va_dim).
            key (Tensor): Text features for attention, shape (batch_size, text_dim).
            value (Tensor): Text features for attention, shape (batch_size, text_dim).
            


        Returns:
            Tensor: The output features after attention and projection, shape (batch_size, text_dim).
        """

        # Transform visual features to the same dimension as text features
        key = self.visual_to_text_dim(key).unsqueeze(0)
        value = self.visual_to_text_dim(value).unsqueeze(0)
        query=query.unsqueeze(0)
       
 
        # Apply multi-head attention: text features (query) attend to visual features (key, value)
        attn_output, _ = self.attention(query, key, value)
        
        # Add & Norm: Add the attention output to the query and normalize
        #attn_output = self.norm(value + attn_output)
        attn_output = self.norm( value+attn_output)
        return  self.output_projection(attn_output)




class CrossModalAttentionVA(nn.Module):
    """
    Cross-Modal Attention Layer for Visual-Audio attending to Text Features.

    
    Parameters:
        text_dim (int): The dimensionality of the text features.
        va_dim (int): The dimensionality of the visual-audio features.
        num_heads (int): The number of attention heads for the multi-head attention.
        emb_dim (int): Dimension of the shared embedding space. Default is 256.

    """      
    def __init__(self,text_dim, va_dim, num_heads=4,emb_dim=256):  # Adjusted dimensions
        super(CrossModalAttentionVA, self).__init__()
        self.text_dim = text_dim
        self.va_dim = va_dim
        self.num_heads = num_heads

        # Linear layers to transform visual features to the same dimension as text features
        self.va_to_emb_dim = nn.Linear(va_dim, emb_dim)
        self.text_to_emb_dim = nn.Linear(text_dim, emb_dim)
        self.attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(emb_dim)
        
        # Linear layer to project the output back to the desired dimension
        self.output_projection = nn.Linear(emb_dim, text_dim)

    def forward(self, query, key, value):

        """
        Forward pass of the CrossModalAttentionVA module. Applies multi-head attention 
        where text features attend to visual/audio features.

        Arguments:
            query (Tensor): Visual/audio features, shape (batch_size, va_dim).
            key (Tensor): Text features for attention, shape (batch_size, text_dim).
            value (Tensor): Text features for attention, shape (batch_size, text_dim).
            


        Returns:
            Tensor: The output features after attention and projection, shape (batch_size, text_dim).
        """
        
        # Transform visual features to the same dimension as text features
        query = self.va_to_emb_dim(query).unsqueeze(0)
        value = self.text_to_emb_dim(value).unsqueeze(0)
        key = self.text_to_emb_dim(key).unsqueeze(0)
       
        # Apply multi-head attention: text features (query) attend to visual features (key, value)
        attn_output, _ = self.attention(query, key, value)
        
        # Add & Norm: Add the attention output to the query and normalize
        attn_output = self.norm(value + attn_output)
        return self.output_projection(attn_output)