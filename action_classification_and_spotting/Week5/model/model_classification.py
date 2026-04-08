import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F

from model.modules import BaseRGBModel, FCLayers, step
from model.neck import create_neck


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
}


def _build_backbone(arch, freeze=False):
    if arch in _CLIP_ARCHS:
        import open_clip
        clip_model, _ = open_clip.create_model_and_transform(*_CLIP_ARCHS[arch])
        model = clip_model.visual.float()
        feat_dim = clip_model.visual.output_dim
        freeze = True
    else:
        timm_name = _TIMM_ALIASES.get(arch, arch)
        model = timm.create_model(timm_name, pretrained=True, num_classes=0)
        feat_dim = model.num_features

    if freeze:
        for p in model.parameters():
            p.requires_grad = False

    return model, feat_dim


class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args=None):
            super().__init__()
            self._feature_arch = args.feature_arch

            self._features, feat_dim = _build_backbone(
                self._feature_arch,
                freeze=getattr(args, 'freeze_backbone', False),
            )

            self._neck, neck_out_dim = create_neck(
                getattr(args, 'neck_architecture', None),
                feat_dim,
                getattr(args, 'neck_parameters', None),
            )

            self._fc = FCLayers(neck_out_dim, args.num_classes)

            self._clip_vit_resize = None
            if self._feature_arch in ('clip_vitb32', 'clip_vitb16', 'clip_vitl14'):
                self._clip_vit_resize = T.Resize((224, 224))

            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
                T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.GaussianBlur(5)], p=0.25),
                T.RandomHorizontalFlip(),
            ])

            if self._feature_arch.startswith('clip_'):
                self.standarization = T.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711))
            else:
                self.standarization = T.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        def forward(self, x):
            x = x / 255.
            batch_size, clip_len, channels, height, width = x.shape

            if self.training:
                for i in range(x.shape[0]):
                    x[i] = self.augmentation(x[i])

            for i in range(x.shape[0]):
                x[i] = self.standarization(x[i])

            frames = x.view(-1, channels, height, width)

            if self._clip_vit_resize is not None:
                frames = self._clip_vit_resize(frames)

            if self._feature_arch.startswith('clip_'):
                im_feat = self._features(frames).float()
            else:
                im_feat = self._features(frames)

            im_feat = im_feat.reshape(batch_size, clip_len, -1)
            im_feat = self._neck(im_feat)
            im_feat = self._fc(im_feat)

            return im_feat

        def print_stats(self):
            print('Model params:', sum(p.numel() for p in self.parameters()))

    def __init__(self, args=None):
        self.device = 'cpu'
        if torch.cuda.is_available() and ('device' in args) and (args.device == 'cuda'):
            self.device = 'cuda'

        self._model = Model.Impl(args=args)
        self._model.print_stats()
        self._args = args
        self._model.to(self.device)
        self._num_classes = args.num_classes
        self._loss_fn = None

    def configure_loss(self, loss_type, pos_weight=None, loss_parameters=None):
        loss_parameters = loss_parameters or {}

        if loss_type == 'bce':
            self._loss_fn = lambda pred, label: F.binary_cross_entropy_with_logits(pred, label)

        elif loss_type == 'weighted_bce':
            if pos_weight is None:
                raise ValueError('pos_weight must be provided for weighted_bce loss')
            w = torch.tensor(pos_weight, dtype=torch.float32).to(self.device)
            self._loss_fn = lambda pred, label: F.binary_cross_entropy_with_logits(
                pred, label, pos_weight=w)

        elif loss_type == 'focal':
            gamma = loss_parameters.get('gamma', 2.0)
            alpha = loss_parameters.get('alpha', 0.25)

            def focal_loss(pred, label):
                bce = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
                p = torch.sigmoid(pred)
                pt = p * label + (1 - p) * (1 - label)
                weight = (1 - pt) ** gamma
                if alpha >= 0:
                    alpha_t = alpha * label + (1 - alpha) * (1 - label)
                    weight = alpha_t * weight
                return (weight * bce).mean()

            self._loss_fn = focal_loss

        else:
            raise ValueError(f'Unknown loss type: {loss_type!r}. Choose from bce, weighted_bce, focal.')

        print(f'Loss configured: {loss_type}  params={loss_parameters}')

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):
        if optimizer is None:
            self._model.eval()
        else:
            optimizer.zero_grad()
            self._model.train()

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch in tqdm(loader):
                frame = batch['frame'].to(self.device).float()
                label = batch['label'].to(self.device).float()

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    loss = self._loss_fn(pred, label)

                if optimizer is not None:
                    step(optimizer, scaler, loss, lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)

    def predict(self, seq):
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4:
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = self._model(seq)
            pred = torch.sigmoid(pred)
            return pred.cpu().numpy()
