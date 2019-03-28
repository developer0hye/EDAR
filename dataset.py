import os
import io
import random
from PIL import Image

class Dataset(object):
    def __init__(self, images_dir, patch_size=48, jpeg_quality=40, transforms=None):
        self.images = os.walk(images_dir).__next__()[2]
        self.images_path = []
        for img_file in self.images:
            self.images_path.append(os.path.join(images_dir, img_file))

        self.patch_size = patch_size
        self.jpeg_quality = jpeg_quality
        self.transforms = transforms
        self.random_rotate = [0, 90, 180, 270]

    def __getitem__(self, idx):
        label = Image.open(self.images_path[idx]).convert('RGB')

        label = label.rotate(self.random_rotate[random.randrange(0,4)])

        # randomly crop patch from training set
        crop_x = random.randint(0, label.width - self.patch_size)
        crop_y = random.randint(0, label.height - self.patch_size)
        label = label.crop((crop_x, crop_y, crop_x + self.patch_size, crop_y + self.patch_size))


        # additive jpeg noise
        buffer = io.BytesIO()
        if random.randrange(0,101) > 95 :
            label.save(buffer, format='jpeg', quality=100)
        else :
            label.save(buffer, format='jpeg', quality=self.jpeg_quality)

        input = Image.open(buffer).convert('RGB')

        if self.transforms is not None:
            input = self.transforms(input)
            label = self.transforms(label)

        return input, label

    def __len__(self):
        return len(self.images_path)