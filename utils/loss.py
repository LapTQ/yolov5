# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Loss functions."""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441."""
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    """Modified BCEWithLogitsLoss to reduce missing label effects in YOLOv5 training with optional alpha smoothing."""

    def __init__(self, alpha=0.05):
        """Initializes a modified BCEWithLogitsLoss with reduced missing label effects, taking optional alpha smoothing
        parameter.
        """
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        """Computes modified BCE loss for YOLOv5 with reduced missing label effects, taking pred and true tensors,
        returns mean loss.
        """
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    """Applies focal loss to address class imbalance by modifying BCEWithLogitsLoss with gamma and alpha parameters."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    """Implements Quality Focal Loss to address class imbalance by modulating loss based on prediction confidence."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes Quality Focal Loss with given loss function, gamma, alpha; modifies reduction to 'none'."""
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Computes the focal loss between `pred` and `true` using BCEWithLogitsLoss, adjusting for imbalance with
        `gamma` and `alpha`.
        """
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    """Computes the total loss for YOLOv5 model predictions, including classification, box, and objectness losses."""

    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        """Initializes ComputeLoss with model and autobalance option, autobalances losses if True."""
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

        # laptq
        """ For a 5-class dataset
        >>> pprint(('self.cp', self.cp))
        ('self.cp', 1.0)
        >>> pprint(('self.cn', self.cn))
        ('self.cn', 0.0)
        >>> pprint(('na', self.na))
        ('na', 3)
        >>> pprint(('nc', self.nc))
        ('nc', 5)
        >>> pprint(('nl', self.nl))
        ('nl', 3)
        """

    def __call__(self, p, targets):  # predictions, targets
        """Performs forward pass, calculating class, box, and object loss for given predictions and targets."""

        # laptq
        """ For a 5-class dataset
        The dataloader yield:
            - imgs.shape = torch.Size([2, 3, 640, 640])    # batch-size = 2
            - targets.shape = torch.Size([8, 6])
        >>> print(('p', type(p), len(p), (p[0].shape, p[1].shape, p[2].shape)))
        ('p', <class 'list'>, 3, (torch.Size([2, 3, 80, 80, 10]), torch.Size([2, 3, 40, 40, 10]), torch.Size([2, 3, 20, 20, 10])))    # list: 3layers x [torch.Size([batch-size, 3, 80, 80, (x,y,w,h,objectness-score,*5class-score)])];
        >>> pprint(('targets', targets.shape))
        ('targets', torch.Size([8, 6]))    # 8 boxes x (image,class,x,y,w,h)
        """
        
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # laptq
        """
        >>> print(('tcls', type(tcls), len(tcls), (tcls[0].shape, tcls[1].shape, tcls[2].shape)))    # list: 3 layer x ...
        ('tcls', <class 'list'>, 3, (torch.Size([54]), torch.Size([24]), torch.Size([6])))
        >>> print(('tbox', type(tbox), len(tbox), (tbox[0].shape, tbox[1].shape, tbox[2].shape)))    # list: 3 layer x ...
        ('tbox', <class 'list'>, 3, (torch.Size([54, 4]), torch.Size([24, 4]), torch.Size([6, 4])))
        >>> print(('indices', type(indices), len(indices), (type(indices[0]), type(indices[1]), type(indices[2]))))    # list: 3 layer x ...
        ('indices', <class 'list'>, 3, (<class 'tuple'>, <class 'tuple'>, <class 'tuple'>))
        >>> print(('indices[0]', type(indices), len(indices[0]), indices[0][0].shape, indices[0][1].shape, indices[0][2].shape, indices[0][3].shape))    # image, anchor, gridy, gridx
        ('indices[0]', <class 'list'>, 4, torch.Size([54]), torch.Size([54]), torch.Size([54]), torch.Size([54]))
        >>> print(('indices[1]', type(indices), len(indices[1]), indices[1][0].shape, indices[1][1].shape, indices[1][2].shape, indices[1][3].shape))    # image, anchor, gridy, gridx
        ('indices[1]', <class 'list'>, 4, torch.Size([24]), torch.Size([24]), torch.Size([24]), torch.Size([24]))
        >>> print(('indices[2]', type(indices), len(indices[2]), indices[2][0].shape, indices[2][1].shape, indices[2][2].shape, indices[2][3].shape))    # image, anchor, gridy, gridx
        ('indices[2]', <class 'list'>, 4, torch.Size([6]), torch.Size([6]), torch.Size([6]), torch.Size([6]))
        >>> print(('anchors', type(anchors), len(anchors), (anchors[0].shape, anchors[1].shape, anchors[2].shape)))    # list: 3 layer x ...
        ('anchors', <class 'list'>, 3, (torch.Size([54, 2]), torch.Size([24, 2]), torch.Size([6, 2])))
        """

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            # laptq
            """
            >>> print(('tobj', tobj.shape))
            ('tobj', torch.Size([2, 3, 80, 80]))
            """

            n = b.shape[0]  # number of targets
            # laptq
            """
            >>> print(('n', n))
            ('n', 54)
            """
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # laptq
                """
                >>> print(('pi[b, a, gj, gi]', pi[b, a, gj, gi].shape))
                ('pi[b, a, gj, gi]', torch.Size([54, 10]))
                >>> print(('pxy', pxy.shape))
                ('pxy', torch.Size([54, 2]))
                >>> print(('pwh', pwh.shape))
                ('pwh', torch.Size([54, 2]))
                >>> print(('_', _.shape, (_.min().item(), _.max().item())))
                ('_', torch.Size([54, 1]), (-7.35546875, -6.0))                            # objectness score
                >>> print(('pcls', pcls.shape, (pcls.min().item(), pcls.max().item())))
                ('pcls', torch.Size([54, 5]), (-2.505859375, -1.046875))                   # class score
                """

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                # laptq
                """
                >>> print(('pbox', pbox.shape))
                ('pbox', torch.Size([54, 4]))
                >>> print(('tbox[i]', tbox[i].shape))
                ('tbox[i]', torch.Size([54, 4]))
                >>> print(('iou', iou.shape, (iou.min().item(), iou.max().item())))
                ('iou', torch.Size([54]), (-0.00167, 0.73278))
                """
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:      # laptq False
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:            # laptq False
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    # laptq
                    """
                    >>> print(('t', t.shape, (t.min().item(), t.max().item())))
                    ('t', torch.Size([54, 5]), (0.0, 1.0))
                    """
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)        # laptq pi[..., 4] is objectness score, obji is scalar
            # laptq
            """
            >>> print(('pi[..., 4]', pi[..., 4].shape))
            ('pi[..., 4]', torch.Size([2, 3, 80, 80]))
            >>> print(('tobj', tobj.shape))
            ('tobj', torch.Size([2, 3, 80, 80]))
            """
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        """

        # laptq
        """ For a 5-class dataset
        The dataloader yield:
            - imgs.shape = torch.Size([2, 3, 640, 640])    # batch-size = 2
            - targets.shape = torch.Size([8, 6])
        >>> print(('p', type(p), len(p), (p[0].shape, p[1].shape, p[2].shape)))
        ('p', <class 'list'>, 3, (torch.Size([2, 3, 80, 80, 10]), torch.Size([2, 3, 40, 40, 10]), torch.Size([2, 3, 20, 20, 10])))
        >>> print(('targets', targets.shape))
        ('targets', torch.Size([8, 6]))    # 8 boxes x (image,class,x,y,w,h)
        """
        
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # laptq: append anchor indices to keep track

        # laptq: It will, at each layer, assign a GT (target) box to every anchor-boxes that fits 
        # (1 GT box can be assigned to many of these (3) anchor-boxes as long as the w,h fit) so we
        # want to keep track of anchor-box indices

        # laptq
        """
        >>> print(('ai', ai.shape, ai))
        ('ai', torch.Size([3, 8]), tensor(
               [[0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1.],
                [2., 2., 2., 2., 2., 2., 2., 2.]])
        )
        >>> print(('targets', targets.shape))
        ('targets', torch.Size([3, 8, 7]))    # laptq: repeat targets 3 times corresponding to the number of anchor-boxes (3) for each layer
        """

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):    # for each layers
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # laptq
            """
            >>> print(('anchors', anchors.shape))
            ('anchors', torch.Size([3, 2]))        # 3 anchor-boxes for this layer
            >>> print(('shape', shape))
            ('shape', torch.Size([2, 3, 80, 80, 10]))
            >>> print(('gain', gain))
            ('gain', tensor([ 1.,  1., 80., 80., 80., 80.,  1.]))
            """

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)    # laptq: scale GT boxes (range 0-1) to the layer's scale (80x80 in this example) 
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # laptq
                """
                >>> print((t[..., 4:6].shape, anchors[:, None].shape))
                (torch.Size([3, 8, 2]), torch.Size([3, 1, 2]))
                >>> print(j.shape)
                torch.Size([3, 8])
                """
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # laptq
                """
                >>> print(('t', t.shape))
                ('t', torch.Size([18, 7]))
                """

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                # laptq
                """
                >>> print(len(t), len(t[j]), len(t[k]), len(t[l]), len(t[m]))
                18 5 1 13 17
                """
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                # laptq
                """
                >>> print(('j', j.shape))
                ('j', torch.Size([5, 18]))
                >>> print(('t.repeat((5, 1, 1))', t.repeat((5, 1, 1)).shape))
                ('t.repeat((5, 1, 1))', torch.Size([5, 18, 7]))
                """
                t = t.repeat((5, 1, 1))[j]
                # laptq
                """
                >>> print(('t', t.shape))
                ('t', torch.Size([54, 7]))
                """
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # laptq: (image, class), grid xy, grid wh, anchor indices
            # laptq
            """
            >>> print(('bc', bc.shape))
            ('bc', torch.Size([54, 2]))
            >>> print(('gxy', gxy.shape))
            ('gxy', torch.Size([54, 2]))
            >>> print(('gwh', gwh.shape))
            ('gwh', torch.Size([54, 2]))
            >>> print(('a', a.shape))
            ('a', torch.Size([54, 1]))
            """
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor indices, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
