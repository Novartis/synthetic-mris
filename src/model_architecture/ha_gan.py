import os
import pickle

import numpy as np
import pytorch_lightning as ptl
import torch
import torch.nn.functional as F
import torchvision
from loguru import logger
from torch import nn
from torch.utils.data.dataloader import DataLoader

from data import BaseImageDataset, ConditionalDataset, make_train_validation_split
from helpers.display.visualization_utils import make_horiz_and_vert_slices


class Upsample(nn.Module):
    def __init__(self, scale_factor: int, mode: str = "trilinear"):
        """Upsample layer used in Generator blocks.

        Args:
            scale_factor (int): Integer factor by which tensor dimensions are scaled.
            mode (str, optional): Interpolation mode to emplay. Defaults to 'trilinear'. Available methods are those of torch.nn.functional.interpolate.
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        out = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return out


class GeneratorA(nn.Module):
    def __init__(self, latent_dim: int, interpolation: str = "trilinear"):
        """Shared generator block used in all HAGAN architectures.

        Args:
            latent_dim (int): Input dimension for the generator & size of noise vector.
            interpolation (str, optional): Type of interpolation to be employed during upsampling. Defaults to 'trilinear'.
                Available methods are those of torch.nn.functional.interpolate.
        """

        super().__init__()
        self.latent_dim = latent_dim
        self.initial_size = 4
        self.current_size = 4
        self.initial_chans = 512
        self.interpolation = interpolation

        self.in_layer = nn.Sequential(nn.Linear(self.latent_dim, self.initial_chans * self.initial_size**3))
        self.model = nn.Sequential(
            *self.convblock(self.initial_chans, self.initial_chans),
            *self.convblock(self.initial_chans, self.initial_chans),
            *self.convblock(self.initial_chans, self.initial_chans // 2),
            *self.convblock(self.initial_chans // 2, self.initial_chans // 4),
            *self.convblock(self.initial_chans // 4, self.initial_chans // 8, upsample=False),
        )

    def convblock(self, in_chans, out_chans, upsample=True):
        block = [
            nn.Conv3d(in_chans, out_chans, 3, 1, 1),
            nn.GroupNorm(1, out_chans),
            nn.ReLU(inplace=True),
        ]
        if upsample:
            block.append(
                nn.Upsample(
                    scale_factor=2,
                    mode=self.interpolation,
                    align_corners=False if self.interpolation in ["linear", "bilinear", "bicubic", "trilinear"] else None,
                )
            )
        return block

    def forward(self, z):
        out = self.in_layer(z)
        out = out.reshape(
            out.shape[0],
            self.initial_chans,
            self.initial_size,
            self.initial_size,
            self.initial_size,
        )
        out = self.model(out)
        return out


class C_GeneratorA(GeneratorA):
    def __init__(
        self,
        latent_dim: int,
        num_embeddings: int,
        embedding_dim: int = 32,
        num_conditions: int = 1,
        interpolation: str = "trilinear",
    ):
        """Conditional version of shared generator, adds embeddings for conditioning to the signal flow.

        Args:
            latent_dim (int): Input dimension for the generator & size of noise vector.
            num_embeddings (int): Size of the dictionary of embeddings. Generally equal to num_conditions.
            embedding_dim (int, optional): The size of each embedding vector. Defaults to 32.
            num_conditions (int, optional): The number of conditions. Defaults to 1.
            interpolation (str, optional): Type of interpolation to be employed during upsampling. Defaults to 'trilinear'.
                Available methods are those of torch.nn.functional.interpolate.
        """
        super().__init__(latent_dim, interpolation=interpolation)
        self.condition_path = nn.Sequential(
            nn.Embedding(num_embeddings, embedding_dim),
            nn.Flatten(),
            nn.Linear(embedding_dim * num_conditions, self.initial_size**3),
        )
        input_layers = self.convblock(self.initial_chans + 1, self.initial_chans)
        conv_layers = list(self.model.children())[len(input_layers) :]
        all_layers = input_layers + conv_layers
        self.model = torch.nn.Sequential(*all_layers)

    def forward(self, model_input):
        z, c = model_input
        # c = torch.Tensor(c)
        noise_in = self.in_layer(z)
        noise_in = noise_in.reshape(
            noise_in.shape[0],
            self.initial_chans,
            self.initial_size,
            self.initial_size,
            self.initial_size,
        )

        condition_in = self.condition_path(c)
        condition_in = condition_in.reshape(
            condition_in.shape[0],
            1,
            self.initial_size,
            self.initial_size,
            self.initial_size,
        )

        conv_in = torch.cat([noise_in, condition_in], dim=1)

        return self.model(conv_in)


class GeneratorL(nn.Module):
    def __init__(self, activation_dim: int):
        """Generator for low-resolution branch of HAGAN architectures.

        Args:
            activation_dim (int): Dimension of incoming hypercube, the output of the shared generator.
                As this is isotropic, the actual shape is batch_size x activation_dim x activation_dim x activation_dim x activation_dim
        """
        super().__init__()
        self.activation_dim = activation_dim

        def convblock(in_dim, out_dim, last=False):
            block = [nn.Conv3d(in_dim, out_dim, 3, 1, 1)]
            if not last:
                block.append(nn.GroupNorm(1, out_dim))
                block.append(nn.ReLU(inplace=True))
            else:
                block.append(nn.Tanh())
            return block

        self.model = nn.Sequential(
            *convblock(activation_dim, activation_dim // 2),
            *convblock(activation_dim // 2, activation_dim // 4),
            *convblock(activation_dim // 4, 1, last=True),
        )

    def forward(self, latent_tensor):
        out = self.model(latent_tensor)
        return out


class GeneratorH(nn.Module):
    def __init__(self, activation_dim: int, interpolation: str = "trilinear"):
        """Generator for high-resolution branch of HAGAN architectures.

        Args:
            activation_dim (int): Dimension of incoming hypercube, the output of the shared generator.
                As this is isotropic, the actual shape is batch_size x activation_dim x activation_dim x activation_dim x activation_dim
            interpolation (str, optional): Type of interpolation to be employed during upsampling. Defaults to 'trilinear'.
                Available methods are those of torch.nn.functional.interpolate.
        """
        super().__init__()
        self.activation_dim = activation_dim
        self.interpolation = interpolation
        layers = [
            nn.Upsample(
                scale_factor=2,
                mode=interpolation,
                align_corners=False if interpolation in ["linear", "bilinear", "bicubic", "trilinear"] else None,
            ),
            nn.Conv3d(activation_dim, activation_dim // 2, 3, 1, 1),
            nn.GroupNorm(1, activation_dim // 2),
        ]

        layers += [
            nn.ReLU(inplace=True),
            nn.Upsample(
                scale_factor=2,
                mode=interpolation,
                align_corners=False if interpolation in ["linear", "bilinear", "bicubic", "trilinear"] else None,
            ),
            nn.Conv3d(activation_dim // 2, 1, 3, 1, 1),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, latent_tensor):
        out = self.model(latent_tensor)
        return out


class DiscriminatorL(nn.Module):
    def __init__(self, low_res_dim: int, num_classes: int = 0):
        """Discriminator for low-resolution branch.

        Args:
            low_res_dim (int): Dimension of low-resolution image.
                Isotropic images are produced, so the actual shape is batch_size x 1 x low_res_dim x low_res_dim x low_res_dim.
            num_classes (int, optional):  number of classes. Discriminator can also be used with auxiliary classification.
                Defaults to 0. use num_classes>0 for conditional discriminitor

        """
        super().__init__()
        self.low_res_dim = low_res_dim

        def convblock(in_chan, out_chan):
            block = torch.nn.ModuleList(
                [
                    nn.utils.spectral_norm(nn.Conv3d(in_chan, out_chan, 4, 2, 1)),
                    nn.LeakyReLU(0.2),
                ]
            )
            return block

        self.model = nn.Sequential(
            *convblock(1, low_res_dim // 2),
            *convblock(low_res_dim // 2, low_res_dim),
            *convblock(low_res_dim, low_res_dim * 2),
            *convblock(low_res_dim * 2, low_res_dim * 4),
            nn.Conv3d(low_res_dim * 4, 1 + num_classes, 4, 1, 0),
            nn.Flatten(),
        )
        self.num_classes = num_classes

    def forward(self, low_res_image):
        out = {}
        out_discr = self.model(low_res_image)

        if self.num_classes == 0:
            out["prob"] = out_discr
            out["class_probs"] = None
        else:
            # conditional case:
            out["prob"] = out_discr[:, 0:1]
            out["class_probs"] = out_discr[:, 1:]

        return out


class DiscriminatorH(nn.Module):
    def __init__(self, num_classes: int = 0):
        """Discriminator for low-resolution branch.

        Args:
            num_classes (int, optional): number of classes. Discriminator can also be used with auxiliary classification.
                In this case, the number of classes must be defined. Defaults to 0. use num_classes>0 for conditional discriminitor

        """
        super().__init__()

        def convblock(in_chan, out_chan, kernel_sz=4, stride=2, padding=1):
            block = torch.nn.ModuleList(
                [
                    nn.utils.spectral_norm(
                        nn.Conv3d(
                            in_chan,
                            out_chan,
                            kernel_size=kernel_sz,
                            stride=stride,
                            padding=padding,
                        )
                    ),
                    nn.LeakyReLU(0.2),
                ]
            )
            return block

        def linblock(in_features, out_features):
            block = torch.nn.ModuleList(
                [
                    nn.utils.spectral_norm(nn.Linear(in_features, out_features)),
                    nn.LeakyReLU(0.2),
                ]
            )
            return block

        self.model = nn.Sequential(
            *convblock(1, 16),
            *convblock(16, 32),
            *convblock(32, 64),
            *convblock(64, 128, kernel_sz=(2, 4, 4), stride=2, padding=(0, 1, 1)),
            *convblock(128, 256, kernel_sz=(2, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            *convblock(256, 512, kernel_sz=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            *convblock(512, 128, kernel_sz=(1, 4, 4), stride=1, padding=0),
            nn.Flatten(),
            *linblock(128, 64),
            *linblock(64, 32),
            nn.Linear(32, 1 + num_classes),
        )
        self.num_classes = num_classes

    def forward(self, hi_res_image):
        out = {}
        out_discr = self.model(hi_res_image)

        if self.num_classes == 0:
            out["prob"] = out_discr
            out["class_probs"] = None
        else:
            # conditional case:
            out["prob"] = out_discr[:, 0:1]
            out["class_probs"] = out_discr[:, 1:]

        return out


class EncoderH(nn.Module):
    def __init__(self, channel=64):
        """Encoder module for high-resolution images.
        When used in combination with EncoderG,
        this enables reconstruction functionality of HAGAN models.
        """

        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(1, channel // 2, kernel_size=4, stride=2, padding=1),  # in:[32,256,256], out:[16,128,128]
            nn.GroupNorm(1, channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel // 2, channel // 2, kernel_size=3, stride=1, padding=1),  # out:[16,128,128]
            nn.GroupNorm(1, 32),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel // 2, channel, kernel_size=4, stride=2, padding=1),  # out:[8,64,64]
            nn.GroupNorm(1, channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_image):
        out = self.model(input_image)
        return out


class EncoderG(nn.Module):
    def __init__(self, channel=256, latent_dim=1024):
        """Encoder module for latent hypercube.
        When used in combination with EncoderH, this enables reconstruction
        functionality of HAGAN models.
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv3d(channel // 4, channel // 8, kernel_size=4, stride=2, padding=1),  # in:[64,64,64], out:[32,32,32]
            nn.GroupNorm(8, channel // 8),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel // 8, channel // 4, kernel_size=4, stride=2, padding=1),  # out:[16,16,16]
            nn.GroupNorm(8, channel // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel // 4, channel // 2, kernel_size=4, stride=2, padding=1),  # out:[8,8,8]
            nn.GroupNorm(8, channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel // 2, channel, kernel_size=4, stride=2, padding=1),  # out:[4,4,4]
            nn.GroupNorm(8, channel),
            nn.ReLU(inplace=True),
        )

        self.conv5 = nn.Conv3d(channel, latent_dim, kernel_size=4, stride=1, padding=0)  # out:[1,1,1,1]

    def forward(self, latent_tensor):
        z = self.model(latent_tensor)  # z.shape:[batch_size, 256, 4, 4, 4]
        z = self.conv5(z).squeeze()  # z.shape:[batch_size, 1024] note: if batch_size == 1 then z.shape = [1024]
        if len(z.shape) == 1:  # to ensure output of shape [batch_size, latent_dim] ie. [2, 1024]
            # fixing dimension problems, this occurs if batch_size==1, or the last loop with batch_size==2 when only 1 image is left
            z = torch.unsqueeze(z, dim=0)  # TODO: is there a better way to handle this? drop_last in the DataLoader did not solve the problem
        return z


class SubspaceSampler(nn.Module):
    def __init__(self, image_dim: int, activation_dim: int, num_image_slices: int, device: str):
        """Module used to generate synchronous subsamples in image and activation space.
            Use prepare_slice() to initialize a new sample pair and extract using get_image_subvolume and get_activation_subvolume.

        Args:
            image_dim (int): slice dim of input images
            activation_dim (int): slice dim of activations
            num_image_slices (int): thickness (number of slices) desired for output image. This is downsampled appropriately for activation subvolume.
            device (str): device used for processing. Can be 'cpu' or 'cuda'.
        """
        super().__init__()
        self.num_image_slices = num_image_slices
        self.image_dim = image_dim
        self.activation_dim = activation_dim
        self.start_slice_l = None
        self.start_slice_h = None
        self.image_subvolumes = None
        self.activation_subvolume = None
        self.device = device

        downsampling_factor = image_dim / activation_dim
        assert downsampling_factor % 1 == 0, f"Downsampling factormust be integer value, but {downsampling_factor} is given."
        self.downsampling_factor = int(downsampling_factor)
        self.num_activation_slices = int(num_image_slices / downsampling_factor)

    def prepare_slice(self, start_slice_act: int = None):
        if start_slice_act is None:
            self.start_slice_l = torch.randint(
                0,
                int(self.activation_dim - self.num_activation_slices),
                (1,),
                device=self.device,
            )
        else:
            self.start_slice_l = start_slice_act

        self.start_slice_h = self.start_slice_l * self.downsampling_factor

    def get_image_subvolume(self, input_volume):
        self.image_subvolumes = input_volume[:, :, self.start_slice_h : self.start_slice_h + self.num_image_slices]
        return self.image_subvolumes

    def get_activation_subvolume(self, input_activation):
        self.activation_subvolume = input_activation[:, :, self.start_slice_l : self.start_slice_l + self.num_activation_slices]
        return self.activation_subvolume


class HAGAN(ptl.LightningModule):
    def __init__(self, cfg, **kwargs):
        """Base class for hierarchical amortized GAN architecture HAGAN, a memory-efficient method to create large 3D images. This method is based on
        https://arxiv.org/abs/2008.01910 and implemented in pytorch lightning. The model uses a shared generator which splits off into two separate heads for
        low-resolution, full volume generation and high-resolution subvolume generation. At test time, the high-resolution branch can be used to generate
        full-volume, high resolution images.

        Args:
            latent_dim (int, optional): Dimensionality of noise input vector for the shared generator. Defaults to 1024.
            activation_dim (int, optional): Size of hypercube, acting as input for both generator heads.
                Isotropic tensors are created, so the actual dim is batch_size  x activation_dim x activation_dim x activation_dim x activation_dim.
                    Defaults to 64.
            low_res_dim (int, optional): Output dimension of low-resolution image head. Isotropic tensors are created, so the actual dim
                is batch_size x 1 x low_res_dim x low_res_dim x low_res_dim. Defaults to 64.
            high_res_dim (int, optional): Output dimension of high-resolution head. Only a subvolume is created, so the final output dim is
                batch_size x 1 x subspace_slices x high_res_dim x high_res_dim. Defaults to 256.
            subspace_slices (int, optional): Number of slices for high-resolution subvolume output. Defaults to 32.
            gen_lr (float, optional): Learning rate for the generator modules. Defaults to 0.0001.
            disc_lr (float, optional): Learning rate for the discriminator modules. Defaults to 0.0004.
            b1 (float, optional): Adam first momentum. Defaults to 0.
            b2 (float, optional): Adam second momentum. Defaults to 0.999.
            weight_decay (float, optional): Weight decay for optimizers. Defaults to 0.
            batch_size (int, optional): Batch size for training. Defaults to 4.
            num_workers (int, optional): Number of workers to be employed in the DataLoader. Defaults to 4.
            data_crop_size (int, optional): Dataloader can crop input images on the fly to match the output dimension of the HAGAN. Defaults to None.
                If inputs are smaller than crop size in any dimension, the image is zero-padded.
            image_crop (str, optional): Type of cropping to be applied. Current options are 'vertical_top', 'vertical_center', and 'vertical_bottom'.
            data_downsampling_factor (int, optional): If set != 1, images are scaled after cropping. Defaults to 1.
                Note that the scaled and cropped image must match the output size of the model.
            interpolation (str, optional): Interpolation method to be employed in the generator blocks during upsampling . Defaults to 'trilinear'.
                Available methods are those of torch.nn.functional.interpolate.
            num_validation_images (int, optional): Number of images to be generated during the validation step and stored on the tensorboard. Defaults to 4.
            disc_train_steps (int, optional): Number of iterations of discriminator training to perform per generator update. Defaults to 1.
        """
        super().__init__()
        self.latent_dim = cfg.model.architecture.latent_dim
        self.activation_dim = cfg.model.architecture.activation_dim
        self.low_res_dim = cfg.model.architecture.low_res_dim
        self.high_res_dim = cfg.model.architecture.high_res_dim
        self.subspace_slices = cfg.model.architecture.subspace_slices
        self.disc_train_steps = cfg.model.architecture.disc_train_steps
        self.interpolation = cfg.model.architecture.interpolation

        self.g_high_res_loss_weight = cfg.model.architecture.g_high_res_loss_weight
        self.d_high_res_loss_weight = cfg.model.architecture.d_high_res_loss_weight

        self.loss_fcn = cfg.model.architecture.loss_fcn

        try:
            self.disc_iter = cfg.model.architecture.disc_iter
        except:
            self.disc_iter = 1
            logger.warning("no disc_iter defined use 1:1 generator:discriminator update")

        try:
            self.gen_iter = cfg.model.architecture.gen_iter
        except:
            self.gen_iter = 1
            logger.warning("no disc_iter defined use 1:1 generator:discriminator update")

        self.gen_lr = cfg.train.optimizers.gen_lr
        self.disc_lr = cfg.train.optimizers.disc_lr
        self.enc_lr = cfg.train.optimizers.enc_lr
        self.b1 = cfg.train.optimizers.b1
        self.b2 = cfg.train.optimizers.b2
        self.weight_decay = cfg.train.optimizers.weight_decay
        self.batch_size = cfg.train.optimizers.batch_size

        self.num_validation_images = cfg.train.log_num_validation_images
        self.num_workers = cfg.data.dataloader_params.num_workers

        self.data_crop_size = cfg.data.image_processing.data_crop_size
        self.crop_mode = cfg.data.image_processing.crop_mode
        self.data_downsampling_factor = cfg.data.image_processing.data_downsampling_factor
        self.clinical_data_processing = cfg.data.clinical_data_processing

        self.conditions = cfg.model.conditions

        self.add_encoder = cfg.model.encoders.add_encoder
        self.attach_encoder = cfg.model.encoders.attach_encoder

        self.train_generators = cfg.model.architecture.train_generators

        # save parameters
        self.save_hyperparameters()

        # set up model
        self.automatic_optimization = False

        image_crop = (tuple([self.data_crop_size] * 3)) if self.data_crop_size is not None else None

        # construct DataLoader
        if self.conditions:  # conditional case
            data_dataset = ConditionalDataset(
                dataset_dir=cfg.data.paths.training_data,
                clinical_data_dir=cfg.data.paths.clinical_data,
                conditions_list=self.conditions,
                clinical_data_processing=self.clinical_data_processing,
                image_crop=image_crop,
                downsample=tuple([self.data_downsampling_factor] * 3),
                crop_mode=self.crop_mode,
            )
        else:
            data_dataset = BaseImageDataset(
                dataset_dir=cfg.data.paths.training_data,
                image_crop=image_crop,
                downsample=tuple([self.data_downsampling_factor] * 3),
                crop_mode=self.crop_mode,
            )

        # Handle conditions - this needs us to have a conditionaldataset first so we know the sizes of the
        # categories & embeddings
        if self.conditions:
            # add condition vector to validation data
            self.c_dims = tuple(dim for _cond, dim in data_dataset.get_conditions())
            self.num_embeddings = np.prod(self.c_dims)
            self.conditions_dict = data_dataset.conditions_dict
            self.gen_a = C_GeneratorA(
                self.latent_dim,
                interpolation=self.interpolation,
                num_embeddings=self.num_embeddings,
                num_conditions=len(self.conditions),
            )  # condition-specific inputs

        else:
            self.gen_a = GeneratorA(self.latent_dim, self.interpolation)
            self.num_embeddings = 0
            self.c_dims = tuple()

        self.gen_l = GeneratorL(self.activation_dim)
        self.discriminator_l = DiscriminatorL(self.low_res_dim, num_classes=self.num_embeddings)

        self.gen_h = GeneratorH(self.activation_dim, self.interpolation)
        self.discriminator_h = DiscriminatorH(num_classes=self.num_embeddings)

        self.sampler = SubspaceSampler(self.high_res_dim, self.activation_dim, self.subspace_slices, self.device)

        # Encoders
        if self.add_encoder:
            # transform it to an activation hypercube
            self.enc_h = EncoderH()
            # take a hypercube and encode it to a latent vector
            self.enc_g = EncoderG()

        self.sampler.prepare_slice()

        # define loss function
        # loss function in the discriminator and generator
        if not self.conditions:
            self.gans_loss = torch.nn.BCEWithLogitsLoss()
        else:
            self.gans_loss = self.conditional_adversarial_loss

        self.encoder_loss = torch.nn.L1Loss()  # mse_loss used in the encoders

        # ensure number of validation images is at most one full batch
        self.num_validation_images = min(self.batch_size, self.num_validation_images)

        # prepare noise vector for validation
        self.validation_z = torch.randn(self.num_validation_images, self.latent_dim)

        try:
            train_val_split_ratio = cfg.data.dataloader_params.train_val_split
            if train_val_split_ratio == None:
                self.train_dataset = data_dataset
                self.validation_dataset = []
            else:
                (
                    self.train_dataset,
                    self.validation_dataset,
                ) = make_train_validation_split(data_dataset, split=train_val_split_ratio)
        except:
            self.train_dataset = data_dataset
            self.validation_dataset = []

        logger.info(f"train/validation dataset size: {len(self.train_dataset)}/{len(self.validation_dataset)}")

    def conditional_adversarial_loss(self, logits: torch.Tensor, true: torch.Tensor, label: torch.Tensor):
        """loss function for conditional model training.
        Uses equal parts of binary crossentropy loss and multiclass loss.

        Args:
            logits (_type_): _description_
            true (_type_): _description_
            label (_type_): _description_

        Returns:
            _type_: _description_
        """
        # split estimate into real/fake probability and class probabilities
        prob, class_probs = logits

        # compute binary cross-entropy for real/fake discrimination
        bce_loss = F.binary_cross_entropy_with_logits(prob, true)

        # calculate class loss
        multiclass_loss_fn = torch.nn.CrossEntropyLoss()
        mc_loss = multiclass_loss_fn(class_probs, label)

        return bce_loss + mc_loss

    def forward_from_activation(self, activation: torch.Tensor):
        """Forward method that starts with the output of the shared generator gen_a.

        Args:
            activation (torch.Tensor): Hypercube coming from the shared generator.

        Returns:
            tuple: containing
                - torch.Tensor: low-resolution image of shape batch_size x 1 x low_res_dim x low_res_dim x low_res_dim
                - torch.Tensor: high-resolution image of shape batch_size x 1 x subspace_slices x high_res_dim x high_res_dim
                    is False
        """

        # use subvolume sampler to get the correct subvolume from the hypercube
        subvolume_activation = self.sampler.get_activation_subvolume(activation)

        # pass hypercube through respective generator heads, if required
        low_res_image = self.gen_l(activation)
        high_res_subvolume = self.gen_h(subvolume_activation)
        return low_res_image, high_res_subvolume

    def forward(self, z: torch.Tensor):
        """Forward pass through entire model.

        Args:
            z (torch.Tensor): Noise vector of shape batch_size x latent_dim

        Returns:
            tuple: containing
                - torch.Tensor: low-resolution image of shape batch_size x 1 x low_res_dim x low_res_dim x low_res_dim
                - torch.Tensor: high-resolution image of shape batch_size x 1 x subspace_slices x high_res_dim x high_res_dim
                    is False
        """

        # create hypercube and pass to remaining model
        activation = self.gen_a(z)  # note: for the conditional case z is a tuple
        return self.forward_from_activation(activation)

    def compute_generative_loss(
        self,
        generated_images: torch.Tensor,
        generated_subvolumes: torch.Tensor,
        c_int_labels: torch.Tensor,
    ):
        log = {}
        valid = torch.ones(generated_images.shape[0], 1, device=self.device)

        out_discr_l_fake = self.discriminator_l(generated_images)  # TODO: ask c_prob_gen_l ??  in the W there is an additiopnal loss with this
        out_discr_h_fake = self.discriminator_h(generated_subvolumes)

        if not self.conditions:
            gen_pred_l = out_discr_l_fake["prob"]
            gen_pred_h = out_discr_h_fake["prob"]
            # note for the unconditional case only the first input is required for the loss
            g_loss_l = self.gans_loss(gen_pred_l, valid)
            g_loss_h = self.gans_loss(gen_pred_h, valid)
        else:
            gen_pred_l = (out_discr_l_fake["prob"], out_discr_l_fake["class_probs"])
            gen_pred_h = (out_discr_h_fake["prob"], out_discr_h_fake["class_probs"])

            g_loss_l = self.gans_loss(gen_pred_l, valid, c_int_labels)
            g_loss_h = self.gans_loss(gen_pred_h, valid, c_int_labels)

        # combine loss for both heads, pass to optimizer & backpropagate
        if self.loss_fcn == "low_res_only":
            g_loss = g_loss_l

        elif self.loss_fcn == "high_res_only":
            g_loss = g_loss_h
        else:
            g_loss = g_loss_l + self.g_high_res_loss_weight * g_loss_h
            # populate additional log
            log["Gen-H weighted-Loss"] = self.g_high_res_loss_weight * g_loss_h.item()

        # populate log
        log["Gen Loss"] = g_loss.item()
        log["Gen-L Loss"] = g_loss_l.item()
        log["Gen-H Loss"] = g_loss_h.item()
        return g_loss, log

    def compute_discriminator_loss(
        self,
        generated_images,
        generated_subvolumes,
        downsampled_images,
        image_subvolumes,
        c_int_labels,
    ):
        log = {}
        # create labels for images
        valid = torch.ones(generated_images.shape[0], 1, device=self.device)
        fake = torch.zeros(generated_images.shape[0], 1, device=self.device)

        # low resolution:
        out_discr_l_r = self.discriminator_l(downsampled_images)  # TODO: c_prob_gen_l and c_prob_gen_h not used?
        out_discr_l_f = self.discriminator_l(generated_images.detach())  # here the gradient does not propagate using detach
        if not self.conditions:
            logits_l_r = out_discr_l_r["prob"]
            logits_l_f = out_discr_l_f["prob"]
            d_loss_l_real = self.gans_loss(logits_l_r, valid)
            d_loss_l_fake = self.gans_loss(logits_l_f, fake)
        else:
            logits_l_r = (out_discr_l_r["prob"], out_discr_l_r["class_probs"])
            logits_l_f = (out_discr_l_f["prob"], out_discr_l_f["class_probs"])
            d_loss_l_real = self.gans_loss(logits_l_r, valid, c_int_labels)
            d_loss_l_fake = self.gans_loss(logits_l_f, fake, c_int_labels)

        d_loss_l = (d_loss_l_real + d_loss_l_fake) / 2

        # high resolution
        out_discr_h_r = self.discriminator_h(image_subvolumes)  # TODO: not used label_h_r adn label_h_f?
        out_discr_h_f = self.discriminator_h(generated_subvolumes.detach())

        if not self.conditions:
            logits_h_r = out_discr_h_r["prob"]
            logits_h_f = out_discr_h_f["prob"]
            d_loss_h_real = self.gans_loss(logits_h_r, valid)
            d_loss_h_fake = self.gans_loss(logits_h_f, fake)
        else:
            logits_h_r = (out_discr_h_r["prob"], out_discr_h_r["class_probs"])
            logits_h_f = (out_discr_h_f["prob"], out_discr_h_f["class_probs"])
            d_loss_h_real = self.gans_loss(logits_h_r, valid, c_int_labels)
            d_loss_h_fake = self.gans_loss(logits_h_f, fake, c_int_labels)
        d_loss_h = (d_loss_h_real + d_loss_h_fake) / 2

        # combine loss of both heads and pass to optimizer & backpropagate
        if self.loss_fcn == "low_res_only":
            d_loss = d_loss_l
        elif self.loss_fcn == "high_res_only":
            d_loss = d_loss_h
        else:
            d_loss = d_loss_l + self.d_high_res_loss_weight * d_loss_h
            log["Disc High res weighted-loss"] = self.d_high_res_loss_weight * d_loss_h.item()

        # log metrics and losses
        log["Disc loss"] = d_loss
        log["Disc Low res loss"] = d_loss_l
        log["Disc High res loss"] = d_loss_h
        log["Disc-L Real"] = d_loss_l_real
        log["Disc-L Fake"] = d_loss_l_fake
        log["Disc-H Real"] = d_loss_h_real
        log["Disc-H Fake"] = d_loss_h_fake
        return d_loss, log

    def training_step(self, batch: tuple, batch_idx: int):
        # depending if more loops in gen or discrim call the corresponding training fcn
        if self.gen_iter > 1:
            self.training_step_gen_loop(batch, batch_idx)  # more loops over generator
        elif self.disc_iter >= 1:
            self.training_step_discr_loop(batch, batch_idx)  # more loops over discriminator
        else:
            # TODO
            raise Exception("training_step to decide")

    def training_step_discr_loop(self, batch: tuple, batch_idx: int):
        """Perform training step on a single batch of data (more iterations in discriminator).
        the training starts with generator optimization -> then the discriminator optimization
        with more loops over discriminator (specified in self.disc_iter)
        -> finally (optionally) encoder optimization

        Args:
            batch (tuple): Tuple containing
                - images (torch.Tensor): real images of shape batch_size x high_res_dim x high_res_dim x high_res_dim
                - c_labels (torch.Tensor): class labels of shape batch_size x num_conditions
        """

        # get optimizers
        optimizers = self.optimizers()
        opt_disc, opt_gen = optimizers[0], optimizers[1]

        if not self.conditions:
            images = batch
        else:
            # split batch into images and class labels
            images, c_labels = batch

        # add feature dimension to images
        images = torch.unsqueeze(images, dim=1)

        # sample point in latent space
        z = torch.randn(images.shape[0], self.latent_dim, device=self.device)

        # get image subvolume and prepare sampler # TODO: why is this performed prior to preparing the sampler? verify
        self.sampler.prepare_slice()
        # reate low-resolution image and subvolume
        downsampled_images = images[:, :, ::4, ::4, ::4]
        image_subvolumes = self.sampler.get_image_subvolume(images)

        # set all params to False as start and then set to true only what needed:
        all_params = [
            self.gen_a.parameters(),
            self.gen_h.parameters(),
            self.gen_l.parameters(),
            self.discriminator_h.parameters(),
            self.discriminator_l.parameters(),
        ]
        for params in all_params:
            for p in params:
                p.requires_grad = False

        # define gen and disc params
        gen_params, disc_params = [], []
        if "gen_a" in self.train_generators or "all" in self.train_generators:
            # activate common generator
            gen_params = gen_params + [self.gen_a.parameters()]

        if "gen_l" in self.train_generators or "all" in self.train_generators:
            gen_params = gen_params + [self.gen_l.parameters()]
            disc_params = disc_params + [self.discriminator_l.parameters()]

        if "gen_h" in self.train_generators or "all" in self.train_generators:
            gen_params = gen_params + [self.gen_h.parameters()]
            disc_params = disc_params + [self.discriminator_h.parameters()]

        ###########################
        # optimize Generators #
        ###########################
        # set grads
        if self.add_encoder:
            for params in [self.enc_h.parameters(), self.enc_g.parameters()]:
                for p in params:
                    p.requires_grad = False

        for params in gen_params:
            for p in params:
                p.requires_grad = True

        # forward pass through model
        if not self.conditions:
            generated_images, generated_subvolumes = self(z)
            c_int_labels = torch.empty((0, 0), dtype=torch.int64)
        else:
            generated_images, generated_subvolumes = self((z, c_labels))
            # convert condition vector to integer representation c_int_labels
            c_int_labels = torch.from_numpy(np.ravel_multi_index(c_labels.T.cpu().numpy(), dims=self.c_dims)).to(torch.int64).to(self.device)

        g_loss, log_gen = self.compute_generative_loss(generated_images, generated_subvolumes, c_int_labels)
        opt_gen.zero_grad()
        self.manual_backward(g_loss)
        opt_gen.step()
        # log metrics and losses
        for log_item in log_gen.items():
            self.log(*log_item, prog_bar=True, sync_dist=True)

        ###########################
        # optimize Discriminators #
        ###########################
        # set grads
        for params in gen_params:
            for p in params:
                p.requires_grad = False
        for params in disc_params:
            for p in params:
                p.requires_grad = True

        for iters in range(self.disc_iter):
            d_loss, log_disc = self.compute_discriminator_loss(
                generated_images,
                generated_subvolumes,
                downsampled_images,
                image_subvolumes,
                c_int_labels,
            )
            opt_disc.zero_grad()
            self.manual_backward(d_loss)
            opt_disc.step()
            for log_item in log_disc.items():
                self.log(*log_item, prog_bar=True, sync_dist=True)

        # set grads
        for params in disc_params:
            for p in params:
                p.requires_grad = False

        ###########################
        # optimize Encoders       #
        ###########################
        if self.add_encoder:
            # set the train parameters
            for p in self.enc_h.parameters():
                p.requires_grad = True

            if self.attach_encoder is True:
                for model_params in [
                    self.gen_a.parameters(),
                    self.gen_h.parameters(),
                    self.gen_l.parameters(),
                ]:
                    for param in model_params:
                        param.requires_grad = True
            else:
                for model_params in [
                    self.gen_a.parameters(),
                    self.gen_h.parameters(),
                    self.gen_l.parameters(),
                ]:
                    for param in model_params:
                        param.requires_grad = False

            # train EncoderH
            opt_enc_h = optimizers[2]

            opt_enc_h.zero_grad()

            z_hat = self.enc_h(image_subvolumes)
            x_hat = self.gen_h(z_hat)

            e_loss = self.encoder_loss(x_hat, image_subvolumes)
            self.manual_backward(e_loss)
            opt_enc_h.step()

            # log metrics and losses
            self.log("enc_h Loss", e_loss, prog_bar=True)

            for p in self.enc_h.parameters():
                p.requires_grad = False

            #####################################
            # train EncoderG
            #####################################
            for p in self.enc_g.parameters():
                p.requires_grad = True

            opt_enc_g = optimizers[3]
            opt_enc_g.zero_grad()
            img_size = images.shape[-1]

            with torch.no_grad():
                z_hat_i_list = []
                # Process all sub-volume and concatenate
                for crop_idx_i in range(0, img_size, img_size // 8):
                    real_images_crop_i = images[:, :, crop_idx_i : crop_idx_i + img_size // 8, :, :]

                    z_hat_i = self.enc_h(real_images_crop_i)
                    z_hat_i_list.append(z_hat_i)
                z_hat = torch.cat(z_hat_i_list, dim=2)

            sub_z_hat = self.enc_g(z_hat)

            # Reconstruction
            if not self.conditions:
                sub_x_hat_low_res_full_img, sub_x_hat_rec_subvolume = self(sub_z_hat)
            else:  # conditional
                sub_x_hat_low_res_full_img, sub_x_hat_rec_subvolume = self((sub_z_hat, c_labels))

            sub_e_loss = (
                self.encoder_loss(sub_x_hat_rec_subvolume, image_subvolumes) + self.encoder_loss(sub_x_hat_low_res_full_img, downsampled_images)
            ) / 2.0

            self.manual_backward(sub_e_loss)
            opt_enc_g.step()

            for p in self.enc_g.parameters():
                p.requires_grad = False

    def training_step_gen_loop(self, batch: tuple, batch_idx: int):
        """Perform training step on a single batch of data (more iterations in generator).
        the training starts with discriminatior optimization -> then the generator optimization
        with more loops over generator (specified in self.gen_iter)
        -> finally (optionally) encoder optimization

        Args:
            batch (tuple): Tuple containing
                - images (torch.Tensor): real images of shape batch_size x high_res_dim x high_res_dim x high_res_dim
                - c_labels (torch.Tensor): class labels of shape batch_size x num_conditions
        """

        # get optimizers
        optimizers = self.optimizers()
        opt_disc, opt_gen = optimizers[0], optimizers[1]

        if not self.conditions:
            images = batch
        else:
            # split batch into images and class labels
            images, c_labels = batch

        # add feature dimension to images
        images = torch.unsqueeze(images, dim=1)

        # sample point in latent space
        z = torch.randn(images.shape[0], self.latent_dim, device=self.device)

        # get image subvolume and prepare sampler # TODO: why is this performed prior to preparing the sampler? verify
        self.sampler.prepare_slice()

        # reate low-resolution image and subvolume
        downsampled_images = images[:, :, ::4, ::4, ::4]
        image_subvolumes = self.sampler.get_image_subvolume(images)

        # set all params to False as start and then set to true only what needed:
        all_params = [
            self.gen_a.parameters(),
            self.gen_h.parameters(),
            self.gen_l.parameters(),
            self.discriminator_h.parameters(),
            self.discriminator_l.parameters(),
        ]
        for params in all_params:
            for p in params:
                p.requires_grad = False

        gen_params, disc_params = [], []
        if "gen_a" in self.train_generators or "all" in self.train_generators:
            # activate common generator
            gen_params = gen_params + [self.gen_a.parameters()]

        if "gen_l" in self.train_generators or "all" in self.train_generators:
            gen_params = gen_params + [self.gen_l.parameters()]
            disc_params = disc_params + [self.discriminator_l.parameters()]

        if "gen_h" in self.train_generators or "all" in self.train_generators:
            gen_params = gen_params + [self.gen_h.parameters()]
            disc_params = disc_params + [self.discriminator_h.parameters()]

        ###########################
        # optimize Discriminator  #
        ###########################
        # set grads
        for params in disc_params:
            for p in params:
                p.requires_grad = True

        # generate synthetic data for the discriminator
        if not self.conditions:
            generated_images, generated_subvolumes = self(z)
            c_int_labels = torch.empty((0, 0), dtype=torch.int64)
        else:
            generated_images, generated_subvolumes = self((z, c_labels))
            # convert condition vector to integer representation c_int_labels
            c_int_labels = torch.from_numpy(np.ravel_multi_index(c_labels.T.cpu().numpy(), dims=self.c_dims)).to(torch.int64).to(self.device)

        d_loss, log_disc = self.compute_discriminator_loss(
            generated_images,
            generated_subvolumes,
            downsampled_images,
            image_subvolumes,
            c_int_labels,
        )
        opt_disc.zero_grad()
        self.manual_backward(d_loss)
        opt_disc.step()
        for log_item in log_disc.items():
            self.log(*log_item, prog_bar=True, sync_dist=True)

        # set grads
        for params in disc_params:
            for p in params:
                p.requires_grad = False

        ###########################
        # optimize Generator      #
        ###########################
        # set grads
        if self.add_encoder:
            for params in [self.enc_h.parameters(), self.enc_g.parameters()]:
                for p in params:
                    p.requires_grad = False

        for params in gen_params:
            for p in params:
                p.requires_grad = True

        for _ in range(self.gen_iter):
            # forward pass through model
            if not self.conditions:
                generated_images, generated_subvolumes = self(z)
                c_int_labels = torch.empty((0, 0), dtype=torch.int64)
            else:
                generated_images, generated_subvolumes = self((z, c_labels))
                # convert condition vector to integer representation c_int_labels
                c_int_labels = torch.from_numpy(np.ravel_multi_index(c_labels.T.cpu().numpy(), dims=self.c_dims)).to(torch.int64).to(self.device)

            g_loss, log_gen = self.compute_generative_loss(generated_images, generated_subvolumes, c_int_labels)
            opt_gen.zero_grad()
            self.manual_backward(g_loss)
            opt_gen.step()
            # log metrics and losses
            for log_item in log_gen.items():
                self.log(*log_item, prog_bar=True, sync_dist=True)

        ###########################
        # optimize Encoders #
        ###########################
        if self.add_encoder:
            # set the train parmetne
            for p in self.enc_h.parameters():
                p.requires_grad = True

            if self.attach_encoder is True:
                for model_params in gen_params:
                    for param in model_params:
                        param.requires_grad = True
            else:
                for model_params in gen_params:
                    for param in model_params:
                        param.requires_grad = False

            # train EncoderH
            opt_enc_h = optimizers[2]
            opt_enc_h.zero_grad()

            z_hat = self.enc_h(image_subvolumes)
            x_hat = self.gen_h(z_hat)

            e_loss = self.encoder_loss(x_hat, image_subvolumes)
            self.manual_backward(e_loss)
            opt_enc_h.step()

            # log metrics and losses
            self.log("enc_h Loss", e_loss, prog_bar=True)

            for p in self.enc_h.parameters():
                p.requires_grad = False

            #####################################
            # train EncoderG
            #####################################
            for p in self.enc_g.parameters():
                p.requires_grad = True

            opt_enc_g = optimizers[3]
            opt_enc_g.zero_grad()
            img_size = images.shape[-1]

            with torch.no_grad():
                z_hat_i_list = []
                # Process all sub-volume and concatenate
                for crop_idx_i in range(0, img_size, img_size // 8):
                    real_images_crop_i = images[:, :, crop_idx_i : crop_idx_i + img_size // 8, :, :]

                    z_hat_i = self.enc_h(real_images_crop_i)
                    z_hat_i_list.append(z_hat_i)
                z_hat = torch.cat(z_hat_i_list, dim=2)

            sub_z_hat = self.enc_g(z_hat)
            # Reconstruction
            if not self.conditions:
                sub_x_hat_low_res_full_img, sub_x_hat_rec_subvolume = self(sub_z_hat)
            else:  # conditional
                sub_x_hat_low_res_full_img, sub_x_hat_rec_subvolume = self((sub_z_hat, c_labels))
                # sub_x_hat_rec, sub_x_hat_rec_small = G(sub_z_hat, crop_idx=crop_idx, class_label=class_label_onehot)

            sub_e_loss = (
                self.encoder_loss(sub_x_hat_rec_subvolume, image_subvolumes) + self.encoder_loss(sub_x_hat_low_res_full_img, downsampled_images)
            ) / 2.0

            self.manual_backward(sub_e_loss)
            opt_enc_g.step()

            for p in self.enc_g.parameters():
                p.requires_grad = False

    def configure_optimizers(self):
        """Define and return all optimizers for the current model.

        Returns:
            tuple: tuple of lists containing
                - list: This list contains discriminator and generator optimizers
                - list: This list contains schedulers for discriminator and generator optimizers. Currently empty.
        """

        # optimizers for low and high resolution branches:
        opt_gen = torch.optim.Adam(
            [
                {"params": self.gen_a.parameters(), "lr": self.gen_lr},
                {"params": self.gen_l.parameters(), "lr": self.gen_lr},
                {"params": self.gen_h.parameters(), "lr": self.gen_lr},
            ],
            betas=(self.b1, self.b2),
            weight_decay=self.weight_decay,
        )

        opt_disc = torch.optim.Adam(
            [
                {"params": self.discriminator_l.parameters(), "lr": self.disc_lr},
                {"params": self.discriminator_h.parameters(), "lr": self.disc_lr},
            ],
            betas=(self.b1, self.b2),
            weight_decay=self.weight_decay,
        )

        # add encoder optimizers if necessary
        if self.add_encoder:
            opt_enc_h = torch.optim.Adam(
                [{"params": self.enc_h.parameters(), "lr": self.enc_lr}],
                betas=(self.b1, self.b2),
                weight_decay=self.weight_decay,
            )
            opt__enc_g = torch.optim.Adam(
                [{"params": self.enc_g.parameters(), "lr": self.enc_lr}],
                betas=(self.b1, self.b2),
                weight_decay=self.weight_decay,
            )
            return [opt_disc, opt_gen, opt_enc_h, opt__enc_g]
        else:
            return [opt_disc, opt_gen]

    def train_dataloader(self):
        """Defines dataloader for model. Uses Dataset object defined in config as used in train_dataset member variable.

        Returns:
            torch.utils.data.dataloader.DataLoader: Dataloader for the model.
        """
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        return self.train_dataloader

    def on_train_epoch_end(self):
        """Defines steps to be performed for evaluation and validation after every epoch."""

        # send precomputed validation noise vector to correct device and
        # perform forward pass
        z = self.validation_z.to(self.device)
        if not self.conditions:
            sample_imgs_l, sample_imgs_h = self(z)
        else:
            gen_condit = torch.cat(
                [torch.randint(c_dim, (self.num_validation_images, 1)) for c_dim in self.c_dims],
                dim=1,
            )
            c = gen_condit.to(self.device)
            sample_imgs_l, sample_imgs_h = self((z, c))

        # generate 2d representations of images and send to tensorboard
        logged_imgs_l = make_horiz_and_vert_slices(sample_imgs_l)
        grid_l = torchvision.utils.make_grid(logged_imgs_l, nrow=self.num_validation_images)
        self.logger.experiment.add_image("generated_low_resolution", grid_l, self.current_epoch)

        grid_h = torchvision.utils.make_grid(sample_imgs_h[:, :, sample_imgs_h.shape[2] // 2])
        self.logger.experiment.add_image("generated_full_resolution_images", grid_h, self.current_epoch)

    def generate(self, model_input):
        """Method used to sample high-resolution, full-volume images.

        Args:
            model_input: can ben a noise vector used for sampling the latent space (for uncoditional case):
                or a tuple (for the conditional case) consisting of:
                - z (torch.Tensor): noise vector of shape batch_size x latent_dim used for sampling the latent space
                - c (torch.Tensor, optional): class labels of shape batch_size x 1 used for conditioning the output.
                    If no class vector is supplied, the class labels are randomly sampled.


        Raises:
            NotImplementedError: Model must contain a high-resolution branch, otherwise generation of high-resolution images is impossible.

        Returns:
            torch.Tensor: full-scale synthetic images in the shape batch_size x 1 x high_res_dim x high_res_dim x high_res_dim
        """

        if not self.conditions:
            z = model_input
            # create activation hypercube
            act = self.gen_a(z)

            # pass full hypercube through high-resolution generator
            full_img = self.gen_h(act)
            return full_img
        else:  # conditional case
            # sample class labels if none are supplied
            if not isinstance(model_input, tuple):
                z = model_input
                c = torch.cat(
                    [[torch.randint(cond_value, (len(model_input),))] for cond_value in self.c_dims],
                    dim=1,
                )
            else:
                z, c = model_input

            # get activation hypercube
            activation = self.gen_a((z, c))

            # pass full hypercube through high-resolution generator branch
            return self.gen_h(activation)

    def generate_low_res(self, z):
        """Low-resolution analogon to generate(). Can be used to sample low-resolution images.

        Args:
            z (_type_): _description_

        Args:
            z (torch.Tensor): noise vector used for sampling the latent space

        Raises:
            NotImplementedError: Model must contain a low-resolution branch, otherwise generation of low-resolution images is impossible.

        Returns:
            torch.Tensor: low-resolution synthetic images in the shape batch_size x 1 x low_res_dim x low_res_dim x low_res_dim
        """
        # TODO: check where used, add the conditional case,
        # create activation hypercube
        activation = self.gen_a(z)

        # pass full hypercube through low-resolution generator
        low_res_image = self.gen_l(activation)
        return low_res_image

    def sample_batch(self, num_samples: int, z: torch.Tensor = None, conditions: list = None):
        """Samples batch of datapoints containing image and tabular data.

        Args:
            num_samples (int): Number of samples to generate.
            z (torch.Tensor, optional): noise vector to be used for sampling. If None is provided, a random vector is sampled.
                Defaults to None

        Returns:
            torch.Tensor: image on CPU in the shape num_samples x 1 x 256 x 256 x 256
        """
        # check model state. If training is True, remember this and reset before exiting method.
        if self.training:
            reset_to_train = True
            self.eval()
        else:
            reset_to_train = False

        # generate the samples accordingly
        image = torch.empty(num_samples, 1, self.high_res_dim, self.high_res_dim, self.high_res_dim)

        if z is None:
            # create noise vector if none supplied
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
        else:
            assert len(z) == num_samples, f"Supplied noise vector and num_samples do not match. Received {num_samples} and {len(z)}."

        # conditional case:
        if self.conditions:
            if conditions is None:
                conditions = torch.stack(
                    [torch.randint(cond_value, (num_samples,), device=self.device) for cond_value in self.c_dims],
                    dim=1,
                )
            z = (z, conditions)  # update input including conditions

        with torch.no_grad():
            image = self.generate(z)

        if reset_to_train:
            self.train()

        return image.cpu()

    def sample(
        self,
        num_samples: int,
        batch_size: int = 1,
        z: torch.Tensor = None,
        image_save_dir: str = None,
        return_samples: bool = False,
        filename_prefix: str = "",
        conditions: list = None,
    ):
        """Batchwise sampling and accumulation of synthetic datapoints.

        Args:
            num_samples (int): Number of samples to be created.
            batch_size (int, optional): Batch size used for processing. Defaults to 1.
            z (torch.Tensor, optional): noise vector to be used for sampling. If None is provided, random vectors are sampled per batch.
                Defaults to None
            image_save_dir (str, optional): If supplied, the generated images are stored to disk at the defined location. Defaults to None.
            return_samples (bool, optional): If set to true, the data are accumulated and returned.
                Warning: can lead to memory issues for large num_samples. Defaults to False.
            filename_prefix (str, optional): Optional prefix to the image filenames. Defaults to ''.

        Returns:
            Union[tuple, bool]: Returns exit state if return_samples is False, else, the accumulated data are returned.
        """
        # check model state. If training is True, remember this and reset before exiting method.
        if self.training:
            reset_to_train = True
            self.eval()
        else:
            reset_to_train = False

        # instantiate empty lists if samples are to be accumulated and returned.
        if return_samples:
            all_images = []

        # modify batch size, if num_samples is less than the given batch_size
        batch_size = min(batch_size, num_samples)

        # create according save directories if necessary
        if image_save_dir is not None:
            os.makedirs(image_save_dir, exist_ok=True)
            img_name_prefix = 0

        # loop through batches. Takes one extra step at the end for potential un-full batches.
        for batch_idx in range((num_samples // batch_size) + 1):
            # handle last (un-full) batch
            if batch_idx == num_samples // batch_size:
                # do nothing if num_samples cleanly divisible by batch_size
                if num_samples % batch_size == 0:
                    continue
                # otherwise set batch_size to remaining chunk
                else:
                    # extract the condition labels associated with last, un-full batch
                    if z is not None:
                        z_batch = z[batch_idx * batch_size : batch_idx * batch_size + num_samples % batch_size + 1]
                        if self.conditions:
                            # conditional case  #TODO: implement case where conditions are not given as input and it's the conditional case
                            c_batch = conditions[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                    else:
                        z_batch = None
                        c_batch = None  # used for conditional case
                    # sample last, un-full batch
                    images = self.sample_batch(num_samples % batch_size, z=z_batch, conditions=c_batch)
            else:
                if z is not None:
                    z_batch = z[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                    if self.conditions:
                        # conditional case  #TODO: implement case where conditions are not given as input and it's the conditional case
                        c_batch = conditions[batch_idx * batch_size : batch_idx * batch_size + num_samples % batch_size + 1]

                else:
                    z_batch = None
                    c_batch = None
                # sample current batch
                images = self.sample_batch(batch_size, z=z_batch, conditions=c_batch)

            # append batch of data to accumulators if required
            if return_samples:
                all_images.append(images)

            # save images to disk if save path is supplied
            if image_save_dir is not None:
                for image_idx in range(len(images)):
                    # construct full filename
                    filename = os.path.join(image_save_dir, f"{filename_prefix}{img_name_prefix:06d}.pickle")
                    img_name_prefix += 1
                    with open(filename, "wb") as f:
                        pickle.dump({"image": images[image_idx, 0].numpy()}, f)  # TODO contions not saved??

            # remove datapoints and clear cache to free memory
            del images
            torch.cuda.empty_cache()

        # reset model state if required
        if reset_to_train:
            self.train()

        # convert accumulated samples to torch.Tensor and pd.DataFrame objects if required
        if return_samples:
            return torch.cat(all_images)
        else:
            return True


class WGP_HAGAN(HAGAN):  # pylint: disable=invalid-name disable=too-many-ancestors
    def __init__(self, cfg, **kwargs):
        """Conditional HAGAN using Wasserstein loss and Gradient Penalty. Adds WGP_HAGAN parent class for gradient penalty loss method.

        Args:
            lambda_cls (int, optional): Mixture hyperparameter for class loss. Defaults to 10, an empirical value from literature.
            lambda_w (int, optional): Mixture hyperparameter for gradient penalty loss. Defaults to 10, an empirical value from literature.
            c_hagan_kwargs(dict): Keyword arguments used for C_HAGAN class. See C_HAGAN docstring for details.
        """
        # assign loss mixture hyperparameters
        self.lambda_cls = cfg.train.optimizers.lambda_cls
        self.lambda_w = cfg.train.optimizers.lambda_w

        # instantiate conditional hagan
        super().__init__(cfg, **kwargs)

        if self.conditions:
            self.conditional_loss = F.cross_entropy

    def calc_gradient_penalty(
        self,
        critic: torch.nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor,
        lambda_gp: int = 10,
    ):
        """Calculate gradient penalty for Wasserstein Gan training procedure.

        Args:
            critic (torch.nn.Module): Critic used to calculate score.
            real (torch.Tensor): Real datapoint
            fake (torch.Tensor): Generated datapoint.
            lambda_gp (int, optional): Mixture hyperparameter for gradient penalty loss. Defaults to 10, an empirical value from literature.

        Returns:
            torch.Tensor: Gradient penalty for current batch.
        """

        # get dimensions of input tensor
        batch_size, c, d, h, w = real.shape

        # sample random epsilon used for interpolating between real and fake image
        epsilon = torch.rand(((batch_size, 1, 1, 1, 1)), device=self.device)

        # create interpolated image and enable gradient for tensor
        interpolated_images = real * epsilon + (1 - epsilon) * fake
        interpolated_images.requires_grad = True

        # calculate score on interpolated image using critic
        interp_score = critic(interpolated_images)
        interp_score = interp_score["prob"]

        # get gradient
        gradient = torch.autograd.grad(
            outputs=interp_score.sum(),
            inputs=interpolated_images,
            create_graph=True,
            only_inputs=True,
        )[0]

        # flatten to one dimension
        gradient_flattened = gradient.flatten(start_dim=1)

        # calculate L2 vector norm along gradient axis
        gradient_norm = torch.linalg.vector_norm(gradient_flattened, ord=2, dim=1)

        # calculate loss term for gradient penalty
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

        # TODO: move lambda out of function into model
        return lambda_gp * gradient_penalty

    def compute_discriminator_loss(
        self,
        generated_images: torch.Tensor,
        generated_subvolumes: torch.Tensor,
        downsampled_images: torch.Tensor,
        image_subvolumes: torch.Tensor,
        c_labels: torch.Tensor,
    ):
        """Method used to calculate critic loss of model. Externalized for ease of use in composite models.
            this is the critic_loss
        Args:
            generated_images (torch.Tensor): Tensor containing generated images of shape batch_size x 1 x low_res_dim x low_res_dim x low_res_dim
            generated_subvolumes (torch.Tensor): Tensor containing generated subvolumes of shape
                batch_size x 1 x num_subvolume_slices x high_res_dim x high_res_dim
            downsampled_images (torch.Tensor): Tensor containing downsampled real images of shape batch_size x 1 x low_res_dim x low_res_dim x low_res_dim
            image_subvolumes (torch.Tensor): Tensor containing real subvolumes of shape
                batch_size x 1 x num_subvolume_slices x high_res_dim x high_res_dim
            c_labels (torch.Tensor): vector containing integer representation of class labels of shape batch_size x 1

        Returns:
            tuple: Tuple containing:
                - loss: Combined loss consisting of gradient penalty, class loss and wasserstein loss for both branches.
                - log (dict): dict of all losses for external logging
        """
        # instantiate dictionary for all log entries
        log = {}

        # individually process both branches of model

        # get critic low_res_branch:
        out_discr_l_real = self.discriminator_l(downsampled_images)
        logits_l_r = out_discr_l_real["prob"]
        c_prob_l = out_discr_l_real["class_probs"]

        out_discr_l_fake = self.discriminator_l(generated_images.detach())
        logits_l_f = out_discr_l_fake["prob"]

        # compute critic loss
        d_loss_l_real = -torch.mean(logits_l_r)
        d_loss_l_fake = torch.mean(logits_l_f)

        # get critic high_res_branch:
        out_discr_h_real = self.discriminator_h(image_subvolumes)
        logits_h_r = out_discr_h_real["prob"]
        c_prob_h = out_discr_h_real["class_probs"]

        out_discr_h_fake = self.discriminator_h(generated_subvolumes.detach())
        logits_h_f = out_discr_h_fake["prob"]

        # compute critic loss
        d_loss_h_real = -torch.mean(logits_h_r)
        d_loss_h_fake = torch.mean(logits_h_f)

        # conditional case
        if not self.conditions:
            cls_loss_h, cls_loss_l = 0, 0
        else:
            # compute cross-entropy class loss
            cls_loss_h = self.conditional_loss(c_prob_h, c_labels)
            cls_loss_l = self.conditional_loss(c_prob_l, c_labels)

        # compute gradient penalty for low and high resoluton
        gradient_penalty_l = self.calc_gradient_penalty(
            self.discriminator_l,
            downsampled_images.data,
            generated_images.data,
            lambda_gp=self.lambda_w,
        )
        gradient_penalty_h = self.calc_gradient_penalty(
            self.discriminator_h,
            image_subvolumes.data,
            generated_subvolumes.data,
            lambda_gp=self.lambda_w,
        )

        # combine all loss components
        d_loss_l = d_loss_l_real + d_loss_l_fake + gradient_penalty_l + self.lambda_cls * cls_loss_l
        d_loss_h = d_loss_h_real + d_loss_h_fake + gradient_penalty_h + self.lambda_cls * cls_loss_h

        d_loss = d_loss_l + self.d_high_res_loss_weight * d_loss_h

        # populate log
        log["Disc loss"] = d_loss
        log["Disc Low res loss"] = d_loss_l
        log["Disc High res loss"] = d_loss_h
        log["Disc-L Real"] = d_loss_l_real.item()
        log["Disc-L Fake"] = d_loss_l_fake.item()
        log["Disc-H Real"] = d_loss_h_real.item()
        log["Disc-H Fake"] = d_loss_h_fake.item()
        if self.conditions:
            log["Class L"] = cls_loss_l.item()
            log["Class H"] = cls_loss_h.item()

        return d_loss, log

    def compute_generative_loss(
        self,
        generated_images: torch.Tensor,
        generated_subvolumes: torch.Tensor,
        c_labels: torch.Tensor,
    ):
        """Method used to calculate generator loss of model. Externalized for ease of use in composite models.

        Args:
            generated_images (torch.Tensor): Tensor containing generated images of shape batch_size x 1 x low_res_dim x low_res_dim x low_res_dim
            generated_subvolumes (torch.Tensor): Tensor containing generated subvolumes of shape
                batch_size x 1 x num_subvolume_slices x high_res_dim x high_res_dim
            c_labels (torch.Tensor): vector containing integer representation of class labels of shape batch_size x 1

        Returns:
            tuple: Tuple containing:
                - loss: Combined loss consisting of class loss and wasserstein loss for both branches.
                - log (dict): dict of all losses for external logging
        """
        # instantiate log dictionary
        log = {}

        # get critic output:
        out_discr_l_fake = self.discriminator_l(generated_images)
        gen_pred_l, c_prob_gen_l = (
            out_discr_l_fake["prob"],
            out_discr_l_fake["class_probs"],
        )

        out_discr_h_fake = self.discriminator_h(generated_subvolumes)
        gen_pred_h, c_prob_gen_h = (
            out_discr_h_fake["prob"],
            out_discr_h_fake["class_probs"],
        )

        # compute critic loss
        g_loss_l = -torch.mean(gen_pred_l)
        g_loss_h = -torch.mean(gen_pred_h)

        # conditional case
        if not self.conditions:
            cls_loss_gen_l, cls_loss_gen_h = 0, 0
        else:
            # compute cross-entropy class loss
            cls_loss_gen_l = self.conditional_loss(c_prob_gen_l, c_labels)
            cls_loss_gen_h = self.conditional_loss(c_prob_gen_h, c_labels)

        # combine loss components according to mixture hyperparameter
        g_loss_l = g_loss_l + self.lambda_cls * cls_loss_gen_l
        g_loss_h = g_loss_h + self.lambda_cls * cls_loss_gen_h
        g_loss = g_loss_l + self.g_high_res_loss_weight * g_loss_h

        # populate log
        log["Gen Loss"] = g_loss.item()
        log["Gen-L Loss"] = g_loss_l.item()
        log["Gen-H Loss"] = g_loss_h.item()

        return g_loss, log
