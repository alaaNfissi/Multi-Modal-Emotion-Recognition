from facenet_pytorch import MTCNN
import torch
import cv2

from torch import nn
from torchvision import transforms


from text_model import ALBERT
from video_model import RepVGGBlock
from models.audio_model.audio_model import CNN18biXLSTM
from transformers_module import WrappedTransformerEncoder
from attention_module import CrossModalAttentionT

def crop_img_center(self, img, target_size=48):
        """
        Crops the center of the image to the target size.

        Args:
            img (torch.Tensor): Image tensor with shape (C, H, W).
            target_size (int, optional): Desired size for height and width. Defaults to 48.

        Returns:
            torch.Tensor: Cropped image tensor.
        """
        current_size = img.size(1)
        off = (current_size - target_size) // 2
        return img[:, off:off + target_size, off:off + target_size]



class MODEL(nn.Module):
    """
    Multi-Modal Emotion Recognition Model.

    Args:
        args (Dict[str, Any]): Dictionary containing model configurations and hyperparameters.
            - 'num_emotions' (int): Number of emotion classes for classification.
        device (torch.device): The device (CPU or GPU) on which computations will be performed.
        text_model_size (str, optional): Size variant of the ALBERT model to use ('base', 'large', 'xlarge').
            Defaults to 'base'.

    Attributes:
        num_classes (int): Number of emotion classes.
        args (Dict[str, Any]): Model configuration parameters.
        mod (List[str]): List of modalities to be used ('t' for text, 'v' for visual, 'a' for audio).
        device (torch.device): Device for computations.
        feature_dim (int): Dimensionality of feature representations.
        T (ALBERT): Text encoder using the ALBERT model.
        mtcnn (MTCNN): Face detection model for preprocessing visual inputs.
        normalize (transforms.Normalize): Normalization layer for visual features.
        V (nn.Sequential): Visual encoder composed of convolutional and RepVGG blocks.
        A (nn.Sequential): Audio encoder using a CNN-LSTM hybrid architecture.
        v_flatten (nn.Sequential): Flattening and projection layers for visual features.
        a_flatten (nn.Sequential): Flattening and projection layers for audio features.
        v_transformer (WrappedTransformerEncoder): Transformer encoder for visual features.
        v_cross_modal_attention (CrossModalAttentionT): Cross-modal attention layer for text attending to visual features.
        a_cross_modal_attention (CrossModalAttentionT): Cross-modal attention layer for text attending to audio features.
        v_out (nn.Linear): Output layer for visual modality.
        t_out (nn.Linear): Output layer for text modality.
        a_out (nn.Linear): Output layer for audio modality.
        weighted_fusion (nn.Linear): Weighted fusion layer to aggregate logits from all modalities.
    """
    def __init__(self, args, device, text_model_size='base'):
        super(MODEL, self).__init__()
        self.num_classes = args['num_emotions']
        self.args = args

        self.mod = args['modalities'] #['t', 'v', 'a'] :  Modalities: text, visual, audio
        self.device = device
        self.feature_dim = 256

        # Transformer configuration
        nlayers, nheads, trans_dim, audio_dim = 4, 4, 64, 144
        text_cls_dim = 1024 if text_model_size == 'large' else 2048 if text_model_size == 'xlarge' else 768  # Dimension for text classification

        # Text encoder
        self.T = ALBERT(feature_dim=self.feature_dim, size=text_model_size)

        # Face detection using MTCNN
        self.mtcnn = MTCNN(image_size=48, margin=2, post_process=False, device=device)

        # Normalization for face images
        self.normalize = transforms.Normalize(mean=[159, 111, 102], std=[37, 33, 32])

        # Visual encoder
        self.V = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            *[RepVGGBlock(64, 64) for _ in range(2)],
            RepVGGBlock(64, 128),
            nn.MaxPool2d(2, stride=2),
            RepVGGBlock(128, 256),
            nn.MaxPool2d(2, stride=2),
            RepVGGBlock(256, 512),
            nn.MaxPool2d(2, stride=2)
        )

        # Audio encoder configuration
        config = {
            "n_input": 1,
            "hidden_dim": 16,
            "n_layers": 18,
            "n_output": 6,
            "lr": self.args['learning_rate'],
            "batch_size": 8
        }

        # Audio encoder
        self.A = nn.Sequential(
            CNN18biXLSTM(
                n_input=config["n_input"],
                hidden_dim=config["hidden_dim"],
                n_layers=config["n_layers"],
                n_output=config["n_output"]
            )
        )

        # Flattening layers for visual and audio features
        self.v_flatten = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, trans_dim)
        )
        self.a_flatten = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, trans_dim)
        )

        # Transformers for visual and audio features
        self.v_transformer = WrappedTransformerEncoder(
            dim=trans_dim,
            num_layers=nlayers,
            num_heads=nheads
        )

        # Uncomment if audio transformer is needed
        # self.a_transformer = WrappedTransformerEncoder(dim=trans_dim, num_layers=nlayers, num_heads=nheads)

        # Cross-modal attention
        self.v_cross_modal_attention = CrossModalAttentionT(
            text_dim=768,
            va_dim=trans_dim,
            num_heads=nheads
        )
        self.a_cross_modal_attention = CrossModalAttentionT(
            text_dim=768,
            va_dim=audio_dim,
            num_heads=nheads
        )
     

        # Output layers for text, visual, and audio
        self.v_out = nn.Linear(trans_dim, self.num_classes)
        self.t_out = nn.Linear(text_cls_dim, self.num_classes)
        self.a_out = nn.Linear(audio_dim, self.num_classes)

        # Weighted fusion layer
        self.weighted_fusion = nn.Linear(len(self.mod), 1, bias=False)

    def forward(self, imgs, imgs_lens, waves,  text):
        """
        Processes input data from text, visual, and audio modalities, applies respective models, 
        and fuses their outputs using cross-modal attention for classification.

        Args:
            imgs (List[torch.Tensor]): List of input images (faces) for the visual modality.
            imgs_lens (List[int]): Sequence lengths for the images (batch-wise).
            waves (torch.Tensor): Raw waveforms for the audio modality.
            text (Dict[str, torch.Tensor]): Tokenized input text for the text modality.

        Returns:
            torch.Tensor: Fused logits from different modalities for classification.

        Notes:
            Returns a single modality's logits if only one modality is specified.
        """
        
        all_logits = []
        imgs_lens2 = []

        # Process text modality
        if 't' in self.mod:
            text_features = self.T(text)
            all_logits.append(self.t_out(text_features))

        # Process visual modality
        if 'v' in self.mod:
            faces = []
            k = [0] * 8  # Initialize counters for each sequence

            for i, img in enumerate(imgs):
                if img is not None and img.any():
                    face = self.mtcnn(img)
                else:
                    face = self.crop_img_center(torch.tensor(img).permute(2, 0, 1))

                if face is not None:
                    face = self.normalize(face)
                    faces.append(face)
                else:
                    for j in range(len(imgs_lens)):
                        if i < sum(imgs_lens[:j+1]):
                            k[j] += 1
                            break

            if faces:
                faces = torch.stack(faces, dim=0).to(self.device)
                imgs_lens2 = [imgs_lens[j] - k[j] for j in range(len(imgs_lens))]
                
                # Flatten visual features and apply transformer
                faces = self.v_flatten(self.V(faces).flatten(start_dim=1))
                visual_features = self.v_transformer(faces, imgs_lens2, get_cls=True)

                # Apply cross-modal attention: text attending to visual features
                visual_features = self.v_cross_modal_attention(
                    query=text_features,
                    key=visual_features,
                    value=visual_features
                ).squeeze(0)

                all_logits.append(self.v_out(visual_features))

        # Process audio modality
        if 'a' in self.mod:
            audio_features = self.A(waves)
            audio_features = self.a_cross_modal_attention(
                query=text_features,
                key=audio_features,
                value=audio_features
            ).squeeze(0)
            all_logits.append(self.a_out(audio_features))

        # Fuse logits from different modalities
        if len(self.mod) == 1:
            return all_logits[0]
        else:
            return self.weighted_fusion(torch.stack(all_logits, dim=-1)).squeeze(-1)

