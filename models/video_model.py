import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Le SEBlock module implémente un mécanisme d'attention au niveau des canaux.
En réajustant dynamiquement l'importance de chaque canal en fonction des informations globales du canal, ce bloc peut aider 
à améliorer les performances du réseau neuronal en se concentrant sur les caractéristiques les plus importantes et en atténuant 
les moins pertinentes
"""
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block.

    The SEBlock implements the squeeze-and-excitation mechanism, which adaptively recalibrates channel-wise feature responses
    by explicitly modeling interdependencies between channels. It enhances the representational power of a network by
    enabling it to perform dynamic channel-wise feature scaling.

    Args:
        input_channels (int): Number of input channels.
        internal_neurons (int): Number of neurons in the internal (bottleneck) layer.

    Attributes:
        down (nn.Conv2d): Convolutional layer to reduce the number of channels.
        up (nn.Conv2d): Convolutional layer to restore the original number of channels.
        input_channels (int): Stores the number of input channels for scaling.
    """

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

        
    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3)) #Squeeze (Compression): réduit chaque canal à une seule valeur moyenne, capturant les informations globales de chaque canal.
        #Excitation 
        x = self.down(x) #applique une convolution pour réduire le nombre de canaux.
        x = F.relu(x)  #Cette opération permet d'apprendre une représentation plus compacte  des canaux de caractéristiques
        x = self.up(x) #applique une autre convolution 1x1 pour remonter le nombre de canaux à la dimension d'origine
        x = torch.sigmoid(x)   #obtenir des valeurs dans l'intervalle (0, 1), qui serviront de coefficients d'importance pour chaque canal.
        #Scaling
        x = x.view(-1, self.input_channels, 1, 1)  #edimensionne les sorties pour qu'elles soient compatibles avec les dimensions des entrées initiales, en ajoutant deux dimensions de taille 1.
        # Cette opération ajuste dynamiquement l'importance de chaque canal de caractéristiques en fonction des informations apprises par le bloc SE.
        return inputs * x
    
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    """
    Creates a sequential block consisting of a Convolutional layer followed by Batch Normalization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        padding (int): Zero-padding added to both sides of the input.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.

    Returns:
       result (nn.Sequential): A sequential container with Conv2d and BatchNorm2d layers.
    """
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGBlock(nn.Module):



    """
    RepVGG Block.

    The RepVGGBlock is a building block for the RepVGG architecture, which allows for the re-parameterization of the network
    during deployment to improve inference speed without sacrificing performance. It supports optional Squeeze-and-Excitation
    (SE) blocks and can switch between training and deployment modes.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Zero-padding added to both sides of the input. Defaults to 1.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        padding_mode (str, optional): Padding mode for the convolution. Defaults to 'zeros'.
        deploy (bool, optional): If `True`, the block is in deployment mode (no BatchNorm layers). Defaults to `False`.
        use_se (bool, optional): If `True`, includes a Squeeze-and-Excitation (SE) block. Defaults to `False`.

    Attributes:
        deploy (bool): Indicates if the block is in deployment mode.
        groups (int): Number of groups for grouped convolutions.
        in_channels (int): Number of input channels.
        nonlinearity (nn.ReLU): ReLU activation function.
        se (nn.Module): Squeeze-and-Excitation block or identity.
        rbr_reparam (nn.Conv2d, optional): Re-parameterized convolutional layer used in deployment mode.
        rbr_identity (nn.BatchNorm2d, optional): Identity branch with BatchNorm used in training mode.
        rbr_dense (nn.Sequential, optional): Dense convolutional branch used in training mode.
        rbr_1x1 (nn.Sequential, optional): 1x1 convolutional branch used in training mode.
        id_tensor (torch.Tensor, optional): Identity tensor for BatchNorm fusion.
    """


    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        # est calculé pour ajuster le padding lorsque nous passons d'un noyau de taille quelconque à un noyau 1×1
        padding_11 = padding - kernel_size // 2 #padding_11 calcule la valeur du padding pour le noyau 1x1.

        self.nonlinearity = nn.ReLU()

        #ertaines parties du réseau sont optionnelles et 
        #peuvent être activées ou désactivées. Utiliser nn.Identity() pour les parties désactivées permet de garder une structure cohérente.
        if use_se:
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            #Si le bloc est en mode déploiement (deploy=True), il utilise une seule couche de convolution sans couche de normalisation BatchNorm.
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            #Une convolution suivie d'une normalisation par lots.
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            #Une convolution 1x1 suivie d'une normalisation par lots.
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)

            #print('RepVGG Block, identity = ', self.rbr_identity) pour le debug

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))


    def get_custom_L2(self): 
        
        """
        Computes a custom L2 regularization penalty for the RepVGGBlock.

        This includes penalties for both the dense (3x3) and 1x1 convolutional weights, excluding the central
        weight of the 3x3 convolution to prevent redundancy.

        Returns:
            torch.Tensor: The computed L2 regularization loss.
        """
        K3 = self.rbr_dense.conv.weight #weight of conv 3x3
        K1 = self.rbr_1x1.conv.weight #weight of conv 1x1
        
        #Facteurs de normalisation basés sur les poids de BatchNorm et la variance des convolutions 3x3 et 1x1.
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1,1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1,1).detach()
        
        """
        self.rbr_dense.bn.weight : Ce sont les poids gamma (ou facteurs d'échelle) de la couche BatchNorm associée à la convolution 3x3.
        self.rbr_dense.bn.running_var : C'est la variance cumulée utilisée par la BatchNorm lors de l'inférence.
        self.rbr_dense.bn.eps : C'est un petit terme de stabilisation ajouté pour éviter les divisions par zéro.
        L'objectif de cette ligne de code est de normaliser les poids gamma par l'écart-type calculé à partir de la variance cumulée, 
        et de les préparer pour une utilisation ultérieure.
        
        Le .detach() est utilisé pour détacher ce tenseur du graphe computationnel, signifiant qu'il ne sera pas utilisé pour calculer les gradients pendant l'entraînement.
        """
        # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2,1:2] ** 2).sum()  
        # The equivalent resultant central point of 3x3 kernel.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1  
        # Normalize for an L2 coefficient comparable to regular L2.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()  
        return l2_loss_eq_kernel + l2_loss_circle

    def get_equivalent_kernel_bias(self): 
        """
        Fuses the convolutional and BatchNorm layers to obtain an equivalent single convolutional kernel and bias.

        This method combines the weights and biases from the 3x3, 1x1, and identity branches to form a single
        convolutional kernel and bias that can be used during deployment for faster inference.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the fused kernel and bias.
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """
        Pads a 1x1 kernel to a 3x3 kernel by adding zeros around the original kernel.

        Args:
            kernel1x1 (Optional[torch.Tensor]): 1x1 convolutional kernel. If `None`, returns 0.

        Returns:
            torch.Tensor: Padded 3x3 convolutional kernel or zero tensor.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """
        Fuses a convolutional branch with its BatchNorm layer.

        Args:
            branch (Optional[nn.Module]): The convolutional branch (Conv + BatchNorm or BatchNorm alone).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the fused kernel and bias.
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self): 
        """
        Converts the RepVGG block to deployment mode by fusing all convolutional and BatchNorm layers into a single convolutional layer.

        This method replaces the multiple branches used during training with a single convolutional layer that
        has the equivalent weights and biases, thus optimizing the model for faster inference.
        """
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True    