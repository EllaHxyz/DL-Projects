import torch
from torch.autograd import Function as TorchFunc
import torch.nn.functional as F
import numpy as np
from PIL import Image


# The "deconvolution" is equivalent to a backward pass through the network, except that 
# when propagating through a nonlinearity, its gradient is solely computed based on the 
# top gradient signal, ignoring the bottom input. In case of the ReLU nonlinearity this 
# amounts to setting to zero certain entries based on the top gradient. We propose to 
# combine these two methods: rather than masking out values corresponding to negative 
# entries of the top gradient ("deconvnet") or bottom data (backpropagation), we mask 
# out the values for which at least one of these values is negative.

class CustomReLU(TorchFunc):
    """
    Define the custom change to the standard ReLU function necessary to perform guided backpropagation.
    We have already implemented the forward pass for you, as this is the same as a normal ReLU function.
    """

    @staticmethod
    def forward(self, x):
        output = torch.addcmul(torch.zeros(x.size()), x, (x > 0).type_as(x))
        self.save_for_backward(x, output)
        return output

    @staticmethod
    def backward(self, dout):
        ##############################################################################
        # TODO: Implement this function. Perform a backwards pass as described in    #
        # the guided backprop paper ( there is also a brief description at the top   #
        # of this page).                                                             #
        # Note: torch.addcmul might be useful, and you can access  the input/output  #
        # from the forward pass with self.saved_tensors.
        # input:
        #   dout is the upstream gradient
        # output:
        #   return downstream gradient
        ##############################################################################
        dgrad = None
        
        x,output = self.saved_tensors
        
        #convert all dout<0 to 0, all others remain the same.
        dgrad = torch.addcmul(torch.zeros(dout.size()),dout,(dout>0).type_as(dout))
        #convert dout to 0, where output=0 (or x<=0)
        dgrad = torch.addcmul(torch.zeros(dgrad.size()),dgrad,(output>0).type_as(dgrad))
        
        return dgrad
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################


class GradCam:
    
    def __init__(self):
        self.activation_value = None
        self.gradient_value = None

    @staticmethod
    def guided_backprop(X_tensor, y_tensor, gc_model):
        """
        Compute a guided backprop visualization using gc_model for images X_tensor and 
        labels y_tensor.

        Input:
        - X_tensor: Input images; Tensor of shape (N, 3, H, W)
        - y: Labels for X; LongTensor of shape (N,)
        - model: A pretrained CNN that will be used to compute the guided backprop.

        Returns:
        - guided backprop: A numpy of shape (N, H, W, 3) giving the guided backprop for 
        the input images.
        """
        for param in gc_model.parameters():
            param.requires_grad = True

        for idx, module in gc_model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                gc_model.features._modules[idx] = CustomReLU.apply
            elif module.__class__.__name__ == 'Fire':
                for idx_c, child in gc_model.features[int(idx)].named_children():
                    if child.__class__.__name__ == 'ReLU':
                        gc_model.features[int(idx)]._modules[idx_c] = CustomReLU.apply
        ##############################################################################
        # TODO: Implement guided backprop as described in paper.                     #
        # (Hint): Now that you have implemented the custom ReLU function, this       #
        # method will be similar to a single training iteration.                     #
        #                                                                            #
        # Also note that the output of this function is a numpy.                     #
        ##############################################################################
        score_mat = gc_model(X_tensor) #(N,C)
        #score = score_mat[:,y_tensor] #(N,1)
        y = torch.unsqueeze(y_tensor, dim=-1)
        score = torch.gather(score_mat,1,y)
        
        score.sum().backward()
        gb_grad = X_tensor.grad.detach().numpy() #(N,3,H,W)
        
        #transpose  to (N,H,W,3)
        gb_grad = gb_grad.transpose(0,2,3,1)
        
        return gb_grad
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

    def grad_cam(self, X_tensor, y_tensor, gc_model):
        """
        Input:
        - X_tensor: Input images; Tensor of shape (N, 3, H, W)
        - y: Labels for X; LongTensor of shape (N,)
        - model: A pretrained CNN that will be used to compute the gradcam.
        """
        #.features holds output of model without last classification layer or last fully connected layer.
        #.features[:3] will slice first 3 layers from features of the model
        conv_module = gc_model.features[12]
        self.gradient_value = None  # Stores gradient of the module you chose above during a backwards pass.
        self.activation_value = None  # Stores the activation of the module you chose above during a forwards pass.

        def gradient_hook(a, b, gradient):
            self.gradient_value = gradient[0]

        def activation_hook(a, b, activation):
            self.activation_value = activation

        conv_module.register_forward_hook(activation_hook)
        conv_module.register_backward_hook(gradient_hook)
        
        '''
        What the hook does?
        1. hook(module, input, output) works on a specific layer/module and take the layer input
        and output to perform certain fcn. For instance, activiation_hook stores the layer output
        to self.activation_value
        2. hook is called right after forward() is called if register as forward hook. same for backward ones.
        see for detail "https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/#accessing-a-particular-layer-from-the-model"
        '''  
        ##############################################################################
        # TODO: Implement GradCam as described in paper.                             #
        #                                                                            #
        # Compute a gradcam visualization using gc_model and convolution layer as    #
        # conv_module for images X_tensor and labels y_tensor.                       # 
        #                                                                            #
        # Hint: All recipe steps can be vectorized. However, if you are stuck on     #
        # vectorizing the weighting of activations by gradients, you can use a       #
        # for-loop for this step. This will work but it is not the recommended       #
        # approach.                                                                  #
        #                                                                            #
        # Return:                                                                    #
        # If the activation map of the convolution layer we are using is (K, K) ,    #
        # student code should end with assigning a numpy of shape (N, K, K) to       #
        # a variable 'cam'. Instructor code would then take care of rescaling it     #
        # back                                                                       #
        ##############################################################################
        
        #1. forward path
        score_mat = gc_model(X_tensor)
        #activation_hook is applied. activation stored.
        
        #score of each input to the correct class (N,C)->(N,1)
        y = torch.unsqueeze(y_tensor, dim=-1)
        score = torch.gather(score_mat,1,y)
        
        #2. backward path
        score.sum().backward()
        #gradient_hook is applied, gradient stored. ->(N,channel,H,W)
        
        #3.global average pooling over each channel ->(N,channel)
        alpha = torch.mean(self.gradient_value,dim=[2,3])
        
        #4.apply alpha weights to each activation channel, sum up
        
        #get activation ->(N,C,H,W)
        act = self.activation_value
        
        #reshape alpha (N,C) ->(N,C,1,1)
        alpha = torch.unsqueeze(alpha,dim=-1)
        alpha = torch.unsqueeze(alpha,dim=-1)
        
        #dot product two matrix element-wise ->(N,C,H,W)
        weighted_act = torch.mul(act,alpha)
        
        #sum over channel ->(N,H,W)
        out = torch.sum(weighted_act,dim=[1])
        
        #ReLu
        m=torch.nn.ReLU()
        cam = m(out).detach().numpy()
       
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

        # Rescale GradCam output to fit image.
        cam_scaled = []
        for i in range(cam.shape[0]):
            cam_scaled.append(np.array(Image.fromarray(cam[i]).resize(X_tensor[i, 0, :, :].shape, Image.BICUBIC)))
        cam = np.array(cam_scaled)
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam
