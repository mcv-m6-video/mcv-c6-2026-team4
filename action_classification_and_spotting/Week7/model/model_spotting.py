"""
File containing the main model.
"""

#Standard imports
import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F


#Local imports
from model.modules import BaseRGBModel, FCLayers, step
from model.neck import create_neck
from model.backbone_3d import VIDEO_3D_ARCHS, build_video_backbone
from model.shift import make_temporal_shift


_CLIP_ARCHS = {
    'clip_vitb32': ('ViT-B-32', 'openai'),
    'clip_vitb16': ('ViT-B-16', 'openai'),
    'clip_vitl14': ('ViT-L-14', 'openai'),
    'clip_rn50':   ('RN50',     'openai'),
}

_TIMM_ALIASES = {
    'rny002': 'regnety_002',
    'rny004': 'regnety_004',
    'rny008': 'regnety_008',
    'rny016': 'regnety_016',
    'rny032': 'regnety_032',
    'rny064': 'regnety_064',
}


def _build_backbone(arch, freeze=False, clip_len=None):
    # Check for temporal shift suffix (_gsf or _gsm)
    shift_mode = None
    base_arch = arch
    if arch.endswith('_gsf'):
        shift_mode = 'gsf'
        base_arch = arch[:-4]
    elif arch.endswith('_gsm'):
        shift_mode = 'gsm'
        base_arch = arch[:-4]

    if base_arch in VIDEO_3D_ARCHS:
        model, feat_dim = build_video_backbone(base_arch, pretrained=True)
        if freeze:
            for p in model.parameters():
                p.requires_grad = False
        return model, feat_dim

    if base_arch in _CLIP_ARCHS:
        import open_clip
        clip_model, _ = open_clip.create_model_and_transform(*_CLIP_ARCHS[base_arch])
        model = clip_model.visual.float()
        feat_dim = clip_model.visual.output_dim
        freeze = True  # CLIP backbone is always frozen
    else:
        timm_name = _TIMM_ALIASES.get(base_arch, base_arch)
        model = timm.create_model(timm_name, pretrained=True, num_classes=0)
        feat_dim = model.num_features

    # Apply temporal shift modules (GSF/GSM) to the backbone
    if shift_mode is not None:
        assert clip_len is not None, \
            f'clip_len is required for temporal shift mode "{shift_mode}"'
        make_temporal_shift(model, clip_len, mode=shift_mode)

    if freeze:
        for p in model.parameters():
            p.requires_grad = False

    return model, feat_dim


class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args = None):
            super().__init__()
            self._feature_arch = args.feature_arch

            self._features, self._d = _build_backbone(
                self._feature_arch,
                freeze=getattr(args, 'freeze_backbone', False),
                clip_len=getattr(args, 'clip_len', None),
            )

            # Flag: True when using a 3-D video backbone (X3D, R3D, SlowFast).
            # Controls the forward path (no per-frame flatten for 3-D models).
            _base_arch = self._feature_arch.replace('_gsf', '').replace('_gsm', '')
            self._is_3d = _base_arch in VIDEO_3D_ARCHS

            # Temporal neck
            self._neck, neck_out_dim = create_neck(
                getattr(args, 'neck_architecture', 'identity'),
                self._d,
                getattr(args, 'neck_parameters', None),
            )

            # MLP for classification
            self._fc = FCLayers(neck_out_dim, args.num_classes+1) # +1 for background class (we now perform per-frame classification with softmax, therefore we have the extra background class)

            #Augmentations and crop
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
                T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.GaussianBlur(5)], p = 0.25),
                T.RandomHorizontalFlip(),
            ])

            # Normalization stats per backbone family:
            #   CLIP   -> CLIP stats
            #   3-D    -> Kinetics-400 stats (models pretrained on Kinetics)
            #   2-D    -> ImageNet stats
            if self._feature_arch.startswith('clip_'):
                self.standarization = T.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711))
            elif self._is_3d:
                self.standarization = T.Normalize(
                    mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225))
            else:
                self.standarization = T.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

            # CLIP ViT models require 224x224 input
            self._clip_vit_resize = None
            if self._feature_arch in ('clip_vitb32', 'clip_vitb16', 'clip_vitl14'):
                self._clip_vit_resize = T.Resize((224, 224))

        def forward(self, x):
            x = self.normalize(x) #Normalize to 0-1
            batch_size, clip_len, channels, height, width = x.shape #B, T, C, H, W

            if self.training:
                x = self.augment(x) #augmentation per-batch

            x = self.standarize(x) #standarization

            if self._is_3d:
                # 3-D backbone expects (B, C, T, H, W).
                # Augmentation/standardization worked on (B, T, C, H, W); permute now.
                x_video = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W)
                im_feat = self._features(x_video)                 # (B, T', D)
            else:
                # 2-D backbone: flatten batch and time into a single leading dim.
                frames = x.view(-1, channels, height, width)

                if self._clip_vit_resize is not None:
                    frames = self._clip_vit_resize(frames)

                if self._feature_arch.startswith('clip_'):
                    im_feat = self._features(frames).float()
                else:
                    im_feat = self._features(frames)

                im_feat = im_feat.reshape(batch_size, clip_len, self._d) #B, T, D

            # Temporal neck: (B, T, D) -> (B, T, D')
            im_feat = self._neck(im_feat)

            #MLP
            im_feat = self._fc(im_feat) #B, T, num_classes+1

            return im_feat
        
        def normalize(self, x):
            return x / 255.
        
        def augment(self, x):
            for i in range(x.shape[0]):
                x[i] = self.augmentation(x[i])
            return x

        def standarize(self, x):
            for i in range(x.shape[0]):
                x[i] = self.standarization(x[i])
            return x

        def print_stats(self):
            print('Model params:',
                sum(p.numel() for p in self.parameters()))

    def __init__(self, args=None):
        self.device = "cpu"
        if torch.cuda.is_available() and ("device" in args) and (args.device == "cuda"):
            self.device = "cuda"

        self._model = Model.Impl(args=args)
        self._model.print_stats()
        self._args = args

        self._model.to(self.device)
        self._num_classes = args.num_classes
        self._focal_gamma = getattr(args, 'focal_gamma', 0.0)
        self._focal_alpha = getattr(args, 'focal_alpha', 5.0)
        self._label_mode = getattr(args, 'label_mode', 'onehot')
        self._label_sigma = float(getattr(args, 'label_sigma', 1.0))
        assert self._label_mode in ('onehot', 'gaussian'), \
            f"Unknown label_mode '{self._label_mode}'"

        # Knowledge distillation settings
        self._teacher = None
        self._kd_alpha = float(getattr(args, 'kd_alpha', 0.0))  # 0 = no KD
        self._kd_temperature = float(getattr(args, 'kd_temperature', 4.0))

        # Mixup
        self._mixup = getattr(args, 'mixup', False)

    def set_teacher(self, teacher):
        """Attach a frozen teacher model for knowledge distillation."""
        self._teacher = teacher
        print(f'[KD] Teacher attached. alpha={self._kd_alpha}, T={self._kd_temperature}')

    def _kd_loss(self, student_logits, teacher_logits, temperature):
        """KL-divergence distillation loss on softened predictions."""
        s = F.log_softmax(student_logits / temperature, dim=-1)
        t = F.softmax(teacher_logits / temperature, dim=-1)
        return F.kl_div(s, t, reduction='batchmean') * (temperature ** 2)

    def _loss(self, pred, label, weights):
        """Weighted cross-entropy, optionally with focal modulation."""
        if self._focal_gamma == 0.0:
            return F.cross_entropy(pred, label, weight=weights, reduction='mean')

        # Focal loss: CE * (1 - p_t)^gamma, with per-class weights playing the alpha role
        log_p = F.log_softmax(pred, dim=-1)                             # (N, C)
        p_t = torch.exp(log_p.gather(1, label.unsqueeze(1))).squeeze(1) # (N,)
        focal_weight = (1.0 - p_t) ** self._focal_gamma                 # (N,)
        ce = F.nll_loss(log_p, label, weight=weights, reduction='none') # (N,)
        return (focal_weight * ce).mean()

    def _gaussian_targets(self, labels):
        """
        Build per-frame soft targets by placing a Gaussian bump around every
        event. labels: (B, T) long, values in [0, num_classes] (0=background).
        Returns (B, T, num_classes+1) float targets summing to 1 per frame.
        """
        B, T = labels.shape
        C = self._num_classes + 1
        device = labels.device
        targets = torch.zeros(B, T, C, device=device, dtype=torch.float32)

        t_range = torch.arange(T, device=device, dtype=torch.float32)
        two_sigma_sq = 2.0 * self._label_sigma * self._label_sigma

        # Iterate only over event positions (sparse).
        event_positions = (labels > 0).nonzero(as_tuple=False)  # (N, 2)
        for b, t0 in event_positions.tolist():
            c = int(labels[b, t0].item())
            bump = torch.exp(-((t_range - t0) ** 2) / two_sigma_sq)
            targets[b, :, c] = torch.maximum(targets[b, :, c], bump)

        # Background class fills the leftover probability mass.
        fg_sum = targets[:, :, 1:].sum(dim=-1).clamp(max=1.0)
        targets[:, :, 0] = 1.0 - fg_sum
        return targets

    def _soft_ce_loss(self, pred, targets, weights):
        """Weighted soft cross-entropy. pred and targets: (N, C); weights: (C,)."""
        log_p = F.log_softmax(pred, dim=-1)
        per_sample = -(targets * log_p * weights.unsqueeze(0)).sum(dim=-1)
        return per_sample.mean()

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        weights = torch.tensor(
            [1.0] + [self._focal_alpha] * self._num_classes,
            dtype=torch.float32,
        ).to(self.device)

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label']
                label = label.to(self.device).long()

                # Mixup: blend pairs of samples with Beta(0.2, 0.2) weights
                mixup_targets = None
                if self._mixup and not inference and frame.shape[0] > 1:
                    import random as _rnd
                    B = frame.shape[0]
                    C = self._num_classes + 1
                    lam = [_rnd.betavariate(0.2, 0.2) for _ in range(B)]
                    # Shuffle indices for pairing
                    perm = torch.randperm(B)
                    mixup_targets = torch.zeros(B, frame.shape[1], C,
                                                device=self.device, dtype=torch.float32)
                    for i in range(B):
                        frame[i] = lam[i] * frame[i] + (1 - lam[i]) * frame[perm[i]]
                        mixup_targets[i, range(frame.shape[1]), label[i]] += lam[i]
                        mixup_targets[i, range(frame.shape[1]), label[perm[i]]] += (1 - lam[i])

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)                              # (B, T, C+1)

                    # Get teacher logits if distilling (only during training)
                    teacher_logits = None
                    if self._teacher is not None and not inference and self._kd_alpha > 0:
                        teacher_logits = self._teacher.get_logits(frame)   # (B, T, C+1)

                    if mixup_targets is not None:
                        # Mixup produces soft targets directly
                        pred_flat = pred.view(-1, self._num_classes + 1)
                        soft_targets = mixup_targets.view(-1, self._num_classes + 1)
                        hard_loss = self._soft_ce_loss(pred_flat, soft_targets, weights)
                    elif self._label_mode == 'gaussian':
                        soft_targets = self._gaussian_targets(label)       # (B, T, C+1)
                        pred_flat = pred.view(-1, self._num_classes + 1)
                        soft_targets = soft_targets.view(-1, self._num_classes + 1)
                        hard_loss = self._soft_ce_loss(pred_flat, soft_targets, weights)
                    else:
                        pred_flat = pred.view(-1, self._num_classes + 1)
                        label_flat = label.view(-1)
                        hard_loss = self._loss(pred_flat, label_flat, weights)

                    if teacher_logits is not None:
                        kd_loss = self._kd_loss(
                            pred.reshape(-1, self._num_classes + 1),
                            teacher_logits.reshape(-1, self._num_classes + 1),
                            self._kd_temperature,
                        )
                        loss = (1 - self._kd_alpha) * hard_loss + self._kd_alpha * kd_loss
                    else:
                        loss = hard_loss

                if optimizer is not None:
                    step(optimizer, scaler, loss,
                        lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)     # Avg loss

    def predict(self, seq):

        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4: # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = self._model(seq)

            # apply sigmoid
            pred = torch.softmax(pred, dim=-1)
            
            return pred.cpu().numpy()
