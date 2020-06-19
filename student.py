import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import detectron2
from detectron2.structures import ImageList
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import sem_seg_postprocess


class Student_block_same(nn.Module):

    def __init__(self, stride, in_channels, out_channels):
        super().__init__()
        assert in_channels == out_channels, f"Use Student_block_diff for {in_channels} -> {out_channels}"

        self.preact_path = nn.Sequential(
            nn.BatchNorm2d(in_channels, track_running_stats=False),
            nn.ReLU()
        )

        if stride == 1:
            self.short_path = nn.Identity()
        else:
            self.short_path = lambda x: x[:,:,::stride,::stride]

        self.long_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1,3), stride=1, padding=(0,1)),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,1), stride=1, padding=(1,0))
        )

    def forward(self, x):
        preact = self.preact_path(x)
        short = self.short_path(x)
        long = self.long_path(preact)
        return short + long


class Student_block_diff(nn.Module):

    def __init__(self, stride, in_channels, out_channels):
        super().__init__()
        assert in_channels != out_channels, f"Use Student_block_same for {in_channels} -> {out_channels}"

        self.preact_path = nn.Sequential(
            nn.BatchNorm2d(in_channels, track_running_stats=False),
            nn.ReLU()
        )

        self.short_path = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

        self.long_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1,3), stride=1, padding=(0,1)),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,1), stride=1, padding=(1,0))
        )

    def forward(self, x):
        preact = self.preact_path(x)
        short = self.short_path(preact)
        long = self.long_path(preact)
        return short + long


@META_ARCH_REGISTRY.register()
class Student(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.num_classes = cfg.DATASETS.NUM_CLASSES
        self.device = torch.device(cfg.MODEL.DEVICE)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        
        self.in_conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.in_conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1)

        self.enc1 = Student_block_diff(in_channels=8, out_channels=64, stride=2)
        self.enc2 = Student_block_same(in_channels=64, out_channels=64, stride=2)
        self.enc3 = Student_block_diff(in_channels=64, out_channels=128, stride=2)
                                                                                          
        self.dec1 = Student_block_diff(in_channels=128, out_channels=64, stride=1)
        self.dec2 = Student_block_diff(in_channels=128, out_channels=32, stride=1)
        self.dec3 = Student_block_diff(in_channels=96, out_channels=32, stride=1)

        self.out_conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.out_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.out_conv3 = nn.Conv2d(in_channels=32, out_channels=self.num_classes, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()        

        self.to(self.device)

    def run_model(self, x : torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv1(x)
        x2 = self.in_conv2(x1)

        enc1 = self.enc1(x2)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        dec1 = F.interpolate(self.dec1(enc3), size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec2 = F.interpolate(self.dec2(torch.cat([dec1, enc2], dim=1)), size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec3 = F.interpolate(self.dec3(torch.cat([dec2, enc1], dim=1)), size=x1.shape[2:], mode='bilinear', align_corners=False)

        out = self.out_conv1(dec3)
        out = self.out_conv2(out)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        out = self.out_conv3(out)

        return out

    def single_spatially_weighted_CE(self, preds, targets, weights):
        """
        Spatially weighted cross entropy loss.
        Reduction is done with "mean".

        Args:
            preds: prediction logits. shape: [NC, H, W]
            targets: label class indices. shape: [H, W]
            weights: spatial weight to multiply for each pixel. shape: [H, W]
        """
        spatial_logits = F.log_softmax(preds, dim=0)
        spatial_target_probs = torch.gather(input=spatial_logits, dim=0, index=targets.unsqueeze(0)).squeeze()
        weighted_spatial_target_probs = spatial_target_probs * weights
        return -torch.mean(weighted_spatial_target_probs)

    def batch_spatially_weighted_CE_ig(self, preds, targets, weights, ignore_index):
        """
        Spatially weighted cross entropy loss with ignore_index support.
        Reduction is done with "mean".

        Args:
            preds: prediction logits. shape: [B, NC, H, W]
            targets: label class indices. shape: [B, H, W]
            weights: spatial weight to multiply for each pixel. shape: [B, H, W]
            ignore_index: index in targets to ignore.
        """
        ig_mask = torch.ne(targets, ignore_index)
        targets[~ig_mask] = self.num_classes
        spatial_logits = \
            torch.cat([F.log_softmax(preds, dim=1), torch.zeros(targets.shape).unsqueeze(1).to(self.device)], dim=1)
        spatial_target_probs = \
            torch.gather(input=spatial_logits, dim=1, index=targets.unsqueeze(1)).squeeze()
        weighted_spatial_target_probs = spatial_target_probs * weights
        return -torch.sum(weighted_spatial_target_probs) / torch.sum(ig_mask)
    
    def forward(self, batched_inputs : list):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, size_divisibility=16)

        preds = self.run_model(images.tensor)

        if self.training:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(targets, size_divisibility=16, pad_value=255).tensor
            return dict(loss_sem_seg=F.cross_entropy(preds, targets, reduction="mean", ignore_index=255))

        processed_preds = []
        for pred, input_per_image, image_size in zip(preds, batched_inputs, images.image_sizes):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            r = sem_seg_postprocess(pred, image_size, height, width)
            processed_preds.append({"sem_seg": r})

        return processed_preds
    
    def single_forward(self, x : torch.Tensor):
        """
        Assumes that x is on CUDA
        Assumes x shape [3, 720, 1280]
        """
        x = self.normalizer(x)
        x = x.unsqueeze(0)
        x = self.run_model(x).squeeze()
        return x

    def single_inference(self, x : torch.Tensor):
        """
        Assumes that x is on CUDA
        Assumes x shape [3, 1080, 1920]
        """
        x = self.normalizer(x)
        x = F.pad(x,(0,0,4,4))
        x = x.unsqueeze(0)
        x = self.run_model(x).squeeze()
        x = x[:, 4:-4, :]
        return x
