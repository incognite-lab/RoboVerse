"""Shared RGB augmentation for recurrent DAgger and the Nyx calibration viewer.

Inputs: float RGB in [0,1], [B,3,H,W] or [B,T,3,H,W]. Apply BEFORE the
student's existing ImageNet normalization. No geometry/actions are changed.
"""
from dataclasses import dataclass, fields
import math

import torch
import torch.nn.functional as F

# Edit augmentation strength here. These are mild starting values, not a
# measured optimum. Restart training/viewer after editing this file.
BRIGHTNESS_RANGE = (0.8, 1.2)
CONTRAST_RANGE = (0.8, 1.2)
GAMMA_RANGE = (0.8, 1.2)
WHITE_BALANCE_RANGE = (0.8, 1.1)  # Independent RGB gains.
NOISE_STD_RANGE = (0.0, 3.0 / 255.0)
BLUR_PROBABILITY = 0.2             # Fixed 3x3 Gaussian kernel [1,2,1] outer product.
CLEAN_PROBABILITY = 0.5            # Keep half the episodes/sequences clean.
IMAGE_CHUNK_SIZE = 16             # Bound temporary memory even at 640x360.
DEFAULT_SEED = 90210


def augmentation_settings():
    return dict(brightness=list(BRIGHTNESS_RANGE),contrast=list(CONTRAST_RANGE),
                gamma=list(GAMMA_RANGE),white_balance=list(WHITE_BALANCE_RANGE),
                noise_std=list(NOISE_STD_RANGE),blur_probability=BLUR_PROBABILITY,
                clean_probability=CLEAN_PROBABILITY,image_chunk_size=IMAGE_CHUNK_SIZE)


@dataclass
class AugmentationParameters:
    # CPU tensors, one row per environment or sequence; never one per frame.
    enabled: torch.Tensor
    brightness: torch.Tensor
    contrast: torch.Tensor
    gamma: torch.Tensor
    white_balance: torch.Tensor
    noise_std: torch.Tensor
    blur: torch.Tensor

    def describe(self):
        return {field.name:getattr(self,field.name).tolist() for field in fields(self)}

    def replace(self, indices, other):
        for field in fields(self):
            getattr(self,field.name).index_copy_(0,indices,getattr(other,field.name))


def sample_augmentation(batch_size, *, generator=None, force=False):
    """Sample one camera style per sequence/env. force previews augmented cases."""
    if batch_size <= 0:
        raise ValueError('Augmentation batch size must be positive')
    def uniform(bounds, shape):
        return bounds[0]+(bounds[1]-bounds[0])*torch.rand(shape,generator=generator,device='cpu')
    enabled = (torch.ones(batch_size,dtype=torch.bool,device='cpu') if force else
               torch.rand(batch_size,generator=generator,device='cpu')>=CLEAN_PROBABILITY)
    return AugmentationParameters(enabled=enabled,
        brightness=uniform(BRIGHTNESS_RANGE,(batch_size,)),contrast=uniform(CONTRAST_RANGE,(batch_size,)),
        gamma=uniform(GAMMA_RANGE,(batch_size,)),white_balance=uniform(WHITE_BALANCE_RANGE,(batch_size,3)),
        noise_std=uniform(NOISE_STD_RANGE,(batch_size,)),
        blur=torch.rand(batch_size,generator=generator,device='cpu')<BLUR_PROBABILITY)


@torch.no_grad()
def augment_visual_input(images, parameters, *, noise_generator=None, inplace=False):
    """Return augmented RGB. Default preserves the source tensor/cache exactly.

    In-place mode is ONLY for owned float batches (e.g. sampled replay), never
    renderer-owned uint8 buffers. Slow changes share parameters across T;
    sensor noise is independently drawn for each pixel/frame.
    """
    if images.ndim not in (4,5) or images.shape[-3] != 3 or not images.is_floating_point():
        raise ValueError('Expected float RGB [B,3,H,W] or [B,T,3,H,W] in [0,1]')
    if images.shape[0] != len(parameters.enabled):
        raise ValueError('Need one augmentation parameter row per environment/sequence')
    if inplace and not images.is_contiguous():
        raise ValueError('In-place augmentation needs an owned contiguous float batch')
    output = images if inplace else images.clone(memory_format=torch.contiguous_format)
    steps = images.shape[1] if images.ndim==5 else 1
    flat = output.reshape(-1,*images.shape[-3:])
    # Only process augmented rows; clean inputs remain bit-for-bit unchanged.
    active = parameters.enabled.repeat_interleave(steps).nonzero().flatten()
    for indices in active.split(IMAGE_CHUNK_SIZE):
        rows = indices//steps
        ids = indices.to(images.device)
        value = flat.index_select(0,ids)
        def scalar(name):
            return getattr(parameters,name).index_select(0,rows).to(device=images.device,dtype=images.dtype).view(-1,1,1,1)
        value.mul_(scalar('brightness'))
        mean = value.mean(dim=(1,2,3),keepdim=True)
        value.sub_(mean).mul_(scalar('contrast')).add_(mean)
        balance = parameters.white_balance.index_select(0,rows).to(device=images.device,dtype=images.dtype)
        value.mul_(balance[:,:,None,None]).clamp_(0,1).pow_(scalar('gamma'))
        blur_rows = parameters.blur.index_select(0,rows).nonzero().flatten()
        if len(blur_rows):
            blur_ids = blur_rows.to(images.device)
            weights = torch.tensor([1.,2.,1.],device=images.device,dtype=images.dtype)
            kernel = (weights[:,None]*weights[None,:]/16).expand(3,1,3,3).contiguous()
            blurred = F.conv2d(F.pad(value.index_select(0,blur_ids),(1,1,1,1),mode='replicate'),kernel,groups=3)
            value.index_copy_(0,blur_ids,blurred)
        if bool((parameters.noise_std.index_select(0,rows)>0).any()):
            noise = torch.randn(value.shape,device=images.device,dtype=images.dtype,generator=noise_generator)
            value.add_(noise.mul_(scalar('noise_std')))
        value.clamp_(0,1)
        flat.index_copy_(0,ids,value)
    return output


class VisualAugmentation:
    """Independent RNG + persistent episode styles; shared by all consumers."""
    def __init__(self, seed=DEFAULT_SEED, saved=None):
        for name,bounds in [('brightness',BRIGHTNESS_RANGE),('contrast',CONTRAST_RANGE),
                            ('gamma',GAMMA_RANGE),('white balance',WHITE_BALANCE_RANGE),('noise',NOISE_STD_RANGE)]:
            if (len(bounds)!=2 or not all(math.isfinite(x) for x in bounds) or bounds[0]>bounds[1]
                    or bounds[0] < 0 or name!='noise' and bounds[0]==0):
                raise ValueError(f'Invalid {name} augmentation range')
        if not 0<=CLEAN_PROBABILITY<=1 or not 0<=BLUR_PROBABILITY<=1 or IMAGE_CHUNK_SIZE<=0:
            raise ValueError('Invalid augmentation probability/chunk size')
        self.seed = int(seed)
        # Genesis changes Torch's default device to CUDA; style RNG/rows stay CPU.
        self.generator = torch.Generator(device='cpu').manual_seed(self.seed)
        self.noise_generators = {}
        self.pending_noise_states = dict((saved or {}).get('noise_states',{}))
        if saved is not None:
            self.generator.set_state(saved['parameter_rng'].cpu())
        self.episode_parameters = None

    def sample(self, batch_size, *, force=False):
        return sample_augmentation(batch_size,generator=self.generator,force=force)

    def apply(self, images, parameters, *, inplace=False):
        key = str(images.device)
        if key not in self.noise_generators:
            generator = torch.Generator(device=images.device).manual_seed(self.seed+1)
            if key in self.pending_noise_states:
                generator.set_state(self.pending_noise_states.pop(key).cpu())
            self.noise_generators[key] = generator
        return augment_visual_input(images,parameters,noise_generator=self.noise_generators[key],inplace=inplace)

    def sequence(self, images, *, inplace=False):
        return self.apply(images,self.sample(images.shape[0]),inplace=inplace)

    def rollout(self, images, reset_mask, *, inplace=False):
        if self.episode_parameters is None or len(self.episode_parameters.enabled)!=images.shape[0]:
            self.episode_parameters = self.sample(images.shape[0])
        else:
            indices = torch.as_tensor(reset_mask,device='cpu',dtype=torch.bool).nonzero().flatten()
            if len(indices):
                self.episode_parameters.replace(indices,self.sample(len(indices)))
        return self.apply(images,self.episode_parameters,inplace=inplace)

    def state_dict(self):
        # Episodes restart on simulator resume; preserve future sampling/noise RNG.
        states = dict(self.pending_noise_states)
        states.update({key:generator.get_state() for key,generator in self.noise_generators.items()})
        return dict(parameter_rng=self.generator.get_state(),noise_states=states)
