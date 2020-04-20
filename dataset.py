import os
import io
import random
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Dataset(object):
    def __init__(self, images_dir, patch_size=48, jpeg_quality=40, transforms=None):
        self.images = os.walk(images_dir).__next__()[2]
        self.images_path = []
        for img_file in self.images:
            if img_file.endswith((".ppm")):
                try:
                    #print(os.path.join(images_dir, img_file))
                    label = Image.open(os.path.join(images_dir, img_file))
                    self.images_path.append(os.path.join(images_dir, img_file))
                except:
                    print(f"Image {os.path.join(images_dir, img_file)} didn't get loaded")
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
        label.save(buffer, format='jpeg', quality= random.randrange(self.jpeg_quality+1))

        input = Image.open(buffer).convert('RGB')

        if self.transforms is not None:
            input = self.transforms(input)
            label = self.transforms(label)
        #print("Image transformed")
        return input, label

    def __len__(self):
        return len(self.images_path)
