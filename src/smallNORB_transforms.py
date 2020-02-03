# adapted pytorch transforms to take smallNORB's stereo pair format
import torch, random, numbers, torchvision
from PIL import Image, ImageOps
from torchvision.transforms import functional as F

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img0, img1):
        for t in self.transforms:
            img0, img1 = t(img0, img1)
        return img0, img1

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ToPILImage(object):
    """Convert 2 tensors or ndarrays to PIL Images.
    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL Image while preserving the value range.
    Args:
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
    """
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, img0, img1):
        """
        Args:
            img0, im1 (Tensor or numpy.ndarray): Images to be converted to PIL Images.
        Returns:
            PIL Images: Images converted to PIL Images.
        """
        return F.to_pil_image(img0, self.mode), F.to_pil_image(img1, self.mode)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        if self.mode is not None:
            format_string += 'mode={0}'.format(self.mode)
        format_string += ')'
        return format_string

class RandomCrop(object):
    """Crop 2 given PIL Images equally at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
             - constant: pads with a constant value, this value is specified with fill
             - edge: pads with the last value on the edge of the image
             - reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
             - symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img0, img1 (PIL Images): Images to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img0, img1):
        """
        Args:
            img0, img1 (PIL Images): Images to be cropped.
        Returns:
            PIL Images: Cropped images.
        """
        if self.padding is not None:
            img0 = F.pad(img0, self.padding, self.fill, self.padding_mode)
            img1 = F.pad(img1, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img0.size[0] < self.size[1]: # img0 and img1 are the same size
            img0 = F.pad(img0, (self.size[1] - img0.size[0], 0), self.fill, self.padding_mode)
            img1 = F.pad(img1, (self.size[1] - img1.size[0], 0), self.fill, self.padding_mode)

        # pad the height if needed
        if self.pad_if_needed and img0.size[1] < self.size[0]: # img0 and img1 are the same size
            img0 = F.pad(img0, (0, self.size[0] - img0.size[1]), self.fill, self.padding_mode)
            img1 = F.pad(img1, (0, self.size[0] - img1.size[1]), self.fill, self.padding_mode)

        # apply the same crop params to both img0 and img1
        i, j, h, w = self.get_params(img0, self.size)

        return F.crop(img0, i, j, h, w), F.crop(img1, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

class CenterCrop(object):
    """Crops 2 given PIL Images equally at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img0, img1):
        """
        Args:
            img0, img1 (PIL Images): Images to be cropped.
        Returns:
            PIL Images: Cropped images.
        """
        return F.center_crop(img0, self.size), F.center_crop(img1, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of 2 images equally.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = torchvision.transforms.Compose(transforms)

        return transform

    def __call__(self, img0, img1):
        """
        Args:
            img0, img1 (PIL Images): Input images.
        Returns:
            PIL Images: Color jittered images.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)

        # apply the same transform to each image channel (img0, img1) separately
        return transform(img0), transform(img1)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

class ToTensor(object):
    """Converts 2 ``PIL Images`` or ``numpy.ndarrays`` to tensors.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
    """
    def __call__(self, img0, img1):
        """
        Args:
            pics (PIL Images or numpy.ndarrays): Images to be converted to tensors.
        Returns:
            Tensors: Converted images.
        """
        return F.to_tensor(img0), F.to_tensor(img1)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class smallnorbStandardize(object):
    """standardizes 2 'PIL Images'"""
    def __call__(self, img0, img1):

        # each image has zero mean and unit variance, avoid division by zero
        img0 = (img0 - img0.mean()) / torch.max(img0.std(), 1./ torch.sqrt(torch.tensor(img0.numel(), dtype=torch.float32)))
        img1 = (img1 - img1.mean()) / torch.max(img1.std(), 1./ torch.sqrt(torch.tensor(img1.numel(), dtype=torch.float32)))

        return img0, img1

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Normalize(object):
    """Normalises 2 'PIL Images'"""
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, img0, img1):

        img0 = (img0 - self.mean[0]) / self.std[0]
        img1 = (img1 - self.mean[1]) / self.std[1]

        # return F.normalize(img0, self.mean[0], self.std[0], self.inplace), \
        #     F.normalize(img1, self.mean[1], self.std[1], self.inplace),
        return img0, img1

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """
    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'
