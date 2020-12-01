import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from layers import *
from vb_routing import *

class CapsuleNet(nn.Module):
    ''' Example: Simple 3 layer CapsNet '''
    def __init__(self, args):
        super(CapsuleNet, self).__init__()

        self.P = args.pose_dim
        self.PP = int(np.max([2, self.P*self.P]))
        self.A, self.B, self.C, self.D = args.arch[:-1]
        self.n_classes = args.n_classes = args.arch[-1]
        self.in_channels = args.n_channels

        self.Conv_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.A,
            kernel_size=5, stride=2, bias=False)
        nn.init.kaiming_uniform_(self.Conv_1.weight)
        self.BN_1 = nn.BatchNorm2d(self.A)

        self.PrimaryCaps = PrimaryCapsules2d(in_channels=self.A, out_caps=self.B,
            kernel_size=1, stride=1, pose_dim=self.P)

        self.ConvCaps_1 = ConvCapsules2d(in_caps=self.B, out_caps=self.C,
            kernel_size=3, stride=2, pose_dim=self.P)

        self.ConvRouting_1 = VariationalBayesRouting2d(in_caps=self.B, out_caps=self.C,
            kernel_size=3, stride=2, pose_dim=self.P,
            cov='diag', iter=args.routing_iter,
            alpha0=1., m0=torch.zeros(self.PP), kappa0=1.,
            Psi0=torch.eye(self.PP), nu0=self.PP+1)

        self.ConvCaps_2 = ConvCapsules2d(in_caps=self.C, out_caps=self.D,
            kernel_size=3, stride=1, pose_dim=self.P)

        self.ConvRouting_2 = VariationalBayesRouting2d(in_caps=self.C, out_caps=self.D,
            kernel_size=3, stride=1, pose_dim=self.P,
            cov='diag', iter=args.routing_iter,
            alpha0=1., m0=torch.zeros(self.PP), kappa0=1.,
            Psi0=torch.eye(self.PP), nu0=self.PP+1)

        self.ClassCaps = ConvCapsules2d(in_caps=self.D, out_caps=self.n_classes,
            kernel_size=1, stride=1, pose_dim=self.P, share_W_ij=True, coor_add=True)

        self.ClassRouting = VariationalBayesRouting2d(in_caps=self.D, out_caps=self.n_classes,
            kernel_size=4, stride=1, pose_dim=self.P, # adjust final kernel_size K depending on input H/W, for H=W=32, K=4.
            cov='diag', iter=args.routing_iter,
            alpha0=1., m0=torch.zeros(self.PP), kappa0=1.,
            Psi0=torch.eye(self.PP), nu0=self.PP+1, class_caps=True)

    def forward(self, x):

        # Out ← [?, A, F, F]
        x = F.relu(self.BN_1(self.Conv_1(x)))

        # Out ← a [?, B, F, F], v [?, B, P, P, F, F]
        a,v = self.PrimaryCaps(x)

        # Out ← a [?, B, 1, 1, 1, F, F, K, K], v [?, B, C, P*P, 1, F, F, K, K]
        a,v = self.ConvCaps_1(a,v)

        # Out ← a [?, C, F, F], v [?, C, P, P, F, F]
        a,v = self.ConvRouting_1(a,v)

        # Out ← a [?, C, 1, 1, 1, F, F, K, K], v [?, C, D, P*P, 1, F, F, K, K]
        a,v = self.ConvCaps_2(a,v)

        # Out ← a [?, D, F, F], v [?, D, P, P, F, F]
        a,v = self.ConvRouting_2(a,v)

        # Out ← a [?, D, 1, 1, 1, F, F, K, K], v [?, D, n_classes, P*P, 1, F, F, K, K]
        a,v = self.ClassCaps(a,v)

        # Out ← yhat [?, n_classes], v [?, n_classes, P, P]
        yhat, v = self.ClassRouting(a,v)

        return yhat

class tinyCapsuleNet(nn.Module):
    ''' Example: Simple 1 layer CapsNet '''
    def __init__(self, args):
        super(CapsuleNet, self).__init__()

        self.P = args.pose_dim
        self.D = int(np.max([2, self.P*self.P]))
        self.A, self.B = args.arch[0], args.arch[2]
        self.n_classes = args.n_classes = args.arch[-1]
        self.in_channels = args.n_channels

        self.Conv_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.A,
            kernel_size=5, stride=2, bias=False)
        nn.init.kaiming_uniform_(self.Conv_1.weight)
        self.BN_1 = nn.BatchNorm2d(self.A)

        self.PrimaryCaps = PrimaryCapsules2d(in_channels=self.A, out_caps=self.B,
            kernel_size=3, stride=2, pose_dim=self.P)

        # same as dense (FC) caps, no weights are shared between class_caps
        self.ClassCaps = ConvCapsules2d(in_caps=self.B, out_caps=self.n_classes,
            kernel_size=6, stride=1, pose_dim=self.P) # adjust K depending on input size

        self.ClassRouting = VariationalBayesRouting2d(in_caps=self.B, out_caps=self.n_classes,
            kernel_size=6, stride=1, pose_dim=self.P,
            cov='diag', iter=args.routing_iter,
            alpha0=1., m0=torch.zeros(self.D), kappa0=1.,
            Psi0=torch.eye(self.D), nu0=self.D+1, class_caps=True)

    def forward(self, x):

        # Out ← [?, A, F, F]
        x = F.relu(self.BN_1(self.Conv_1(x)))

        # Out ← a [?, B, F, F], v [?, B, P, P, F, F]
        a,v = self.PrimaryCaps(x)

        # Out ← a [?, B, 1, 1, 1, F, F, K, K], v [?, B, C, P*P, 1, F, F, K, K]
        a,v = self.ClassCaps(a,v)

        # Out ← yhat [?, C], v [?, C, P*P, 1]
        yhat, v = self.ClassRouting(a,v)

        return yhat
