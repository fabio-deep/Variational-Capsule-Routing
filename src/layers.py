import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class PrimaryCapsules2d(nn.Module):
    '''Primary Capsule Layer'''
    def __init__(self, in_channels, out_caps, kernel_size, stride,
        padding=0, pose_dim=4, weight_init='xavier_uniform'):
        super().__init__()

        self.A = in_channels
        self.B = out_caps
        self.P = pose_dim
        self.K = kernel_size
        self.S = stride
        self.padding = padding

        w_kernel = torch.empty(self.B*self.P*self.P, self.A, self.K, self.K)
        a_kernel = torch.empty(self.B, self.A, self.K, self.K)

        if weight_init == 'kaiming_normal':
            nn.init.kaiming_normal_(w_kernel)
            nn.init.kaiming_normal_(a_kernel)
        elif weight_init == 'kaiming_uniform':
            nn.init.kaiming_uniform_(w_kernel)
            nn.init.kaiming_uniform_(a_kernel)
        elif weight_init == 'xavier_normal':
            nn.init.xavier_normal_(w_kernel)
            nn.init.xavier_normal_(a_kernel)
        elif weight_init == 'xavier_uniform':
            nn.init.xavier_uniform_(w_kernel)
            nn.init.xavier_uniform_(a_kernel)
        else:
            NotImplementedError('{} not implemented.'.format(weight_init))

        # Out ← [B*(P*P+1), A, K, K]
        self.weight = nn.Parameter(torch.cat([w_kernel, a_kernel], dim=0))

        self.BN_a = nn.BatchNorm2d(self.B, affine=True)
        self.BN_p = nn.BatchNorm3d(self.B, affine=True)

    def forward(self, x): # [?, A, F, F] ← In

        # Out ← [?, B*(P*P+1), F, F]
        x = F.conv2d(x, weight=self.weight, stride=self.S, padding=self.padding)

        # Out ← ([?, B*P*P, F, F], [?, B, F, F]) ← [?, B*(P*P+1), F, F]
        poses, activations = torch.split(x, [self.B*self.P*self.P, self.B], dim=1)

        # Out ← [?, B, P*P, F, F]
        poses = self.BN_p(poses.reshape(-1, self.B, self.P*self.P, *x.shape[2:]))

        # Out ← [?, B, P, P, F, F] ← [?, B, P*P, F, F] ← In
        poses = poses.reshape(-1, self.B, self.P, self.P, *x.shape[2:])

        # Out ← [?, B, F, F])
        activations = torch.sigmoid(self.BN_a(activations))

        return (activations, poses)

class ConvCapsules2d(nn.Module):
    '''Convolutional Capsule Layer'''
    def __init__(self, in_caps, out_caps, pose_dim, kernel_size, stride, padding=0,
        weight_init='xavier_uniform', share_W_ij=False, coor_add=False):
        super().__init__()

        self.B = in_caps
        self.C = out_caps
        self.P = pose_dim
        self.PP = np.max([2, self.P*self.P])
        self.K = kernel_size
        self.S = stride
        self.padding = padding

        self.share_W_ij = share_W_ij # share the transformation matrices across (F*F)
        self.coor_add = coor_add # embed coordinates

        # Out ← [1, B, C, 1, P, P, 1, 1, K, K]
        self.W_ij = torch.empty(1, self.B, self.C, 1, self.P, self.P, 1, 1, self.K, self.K)

        if weight_init.split('_')[0] == 'xavier':
            fan_in = self.B * self.K*self.K * self.PP # in_caps types * receptive field size
            fan_out = self.C * self.K*self.K * self.PP # out_caps types * receptive field size
            std = np.sqrt(2. / (fan_in + fan_out))
            bound = np.sqrt(3.) * std

            if weight_init.split('_')[1] == 'normal':
                self.W_ij = nn.Parameter(self.W_ij.normal_(0, std))
            elif weight_init.split('_')[1] == 'uniform':
                self.W_ij = nn.Parameter(self.W_ij.uniform_(-bound, bound))
            else:
                raise NotImplementedError('{} not implemented.'.format(weight_init))

        elif weight_init.split('_')[0] == 'kaiming':
            # fan_in preserves magnitude of the variance of the weights in the forward pass.
            fan_in = self.B * self.K*self.K * self.PP # in_caps types * receptive field size
            # fan_out has same affect as fan_in for backward pass.
            # fan_out = self.C * self.K*self.K * self.PP # out_caps types * receptive field size
            std = np.sqrt(2.) / np.sqrt(fan_in)
            bound = np.sqrt(3.) * std

            if weight_init.split('_')[1] == 'normal':
                self.W_ij = nn.Parameter(self.W_ij.normal_(0, std))
            elif weight_init.split('_')[1] == 'uniform':
                self.W_ij = nn.Parameter(self.W_ij.uniform_(-bound, bound))
            else:
                raise NotImplementedError('{} not implemented.'.format(weight_init))

        elif weight_init == 'noisy_identity' and self.PP > 2:
            b = 0.01 # U(0,b)
            # Out ← [1, B, C, 1, P, P, 1, 1, K, K]
            self.W_ij = nn.Parameter(torch.clamp(.1*torch.eye(self.P,self.P).repeat( \
                1, self.B, self.C, 1, 1, 1, self.K, self.K, 1, 1) + \
                torch.empty(1, self.B, self.C, 1, 1, 1, self.K, self.K, self.P, self.P).uniform_(0,b), \
                max=1).permute(0, 1, 2, 3, -2, -1, 4, 5, 6, 7))
        else:
            raise NotImplementedError('{} not implemented.'.format(weight_init))

        if self.padding != 0:
            if isinstance(self.padding, int):
                self.padding = [self.padding]*4

    def forward(self, activations, poses): # ([?, B, F, F], [?, B, P, P, F, F]) ← In

        if self.padding != 0:
            activations = F.pad(activations, self.padding) # [1,1,1,1]
            poses = F.pad(poses, self.padding + [0]*4) # [0,0,1,1,1,1]

        if self.share_W_ij: # share the matrices over (F*F), if class caps layer
            self.K = poses.shape[-1] # out_caps (C) feature map size

        self.F = (poses.shape[-1] - self.K) // self.S + 1 # featuremap size

        # Out ← [?, B, P, P, F', F', K, K] ← [?, B, P, P, F, F]
        poses = poses.unfold(4, size=self.K, step=self.S).unfold(5, size=self.K, step=self.S)

        # Out ← [?, B, 1, P, P, 1, F', F', K, K] ← [?, B, P, P, F', F', K, K]
        poses = poses.unsqueeze(2).unsqueeze(5)

        # Out ← [?, B, F', F', K, K] ← [?, B, F, F]
        activations = activations.unfold(2, size=self.K, step=self.S).unfold(3, size=self.K, step=self.S)

        # Out ← [?, B, 1, 1, 1, F', F', K, K] ← [?, B, F', F', K, K]
        activations = activations.reshape(-1, self.B, 1, 1, 1, *activations.shape[2:4], self.K, self.K)

        # Out ← [?, B, C, P, P, F', F', K, K] ← ([?, B, 1, P, P, 1, F', F', K, K] * [1, B, C, 1, P, P, 1, 1, K, K])
        V_ji = (poses * self.W_ij).sum(dim=4) # matmul equiv.

        # Out ← [?, B, C, P*P, 1, F', F', K, K] ← [?, B, C, P, P, F', F', K, K]
        V_ji = V_ji.reshape(-1, self.B, self.C, self.P*self.P, 1, *V_ji.shape[-4:-2], self.K, self.K)

        if self.coor_add:
            if V_ji.shape[-1] == 1: # if class caps layer (featuremap size = 1)
                self.F = self.K # 1->4

            # coordinates = torch.arange(self.F, dtype=torch.float32) / self.F
            coordinates = torch.arange(self.F, dtype=torch.float32).add(1.) / (self.F*10)
            i_vals = torch.zeros(self.P*self.P,self.F,1).cuda()
            j_vals = torch.zeros(self.P*self.P,1,self.F).cuda()
            i_vals[self.P-1,:,0] = coordinates
            j_vals[2*self.P-1,0,:] = coordinates

            if V_ji.shape[-1] == 1: # if class caps layer
                # Out ← [?, B, C, P*P, 1, 1, 1, K=F, K=F] (class caps)
                V_ji = V_ji + (i_vals + j_vals).reshape(1,1,1,self.P*self.P,1,1,1,self.F,self.F)
                return activations, V_ji

            # Out ← [?, B, C, P*P, 1, F, F, K, K]
            V_ji = V_ji + (i_vals + j_vals).reshape(1,1,1,self.P*self.P,1,self.F,self.F,1,1)

        return activations, V_ji
