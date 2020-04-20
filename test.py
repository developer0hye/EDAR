import argparse
import os
import io
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
import PIL.Image as pil_image
import glob

from edar import EDAR

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=False)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--jpeg_quality', type=int, default=40)
    parser.add_argument('--input_dir', type=str, required=False)
    opt, unknown = parser.parse_known_args()
    model = EDAR()

    state_dict = model.state_dict()
    for n, p in torch.load(opt.weights_path, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model = model.to(device)
    print(device)
    model.eval()
    
    if opt.input_dir:
        filenames = [os.path.join(opt.input_dir, file) for file in os.listdir(opt.input_dir) if file.endswith(("ppm", "jpeg", "png", "jpg"))]
        print(filenames)
    else:
        filenames = opt.image_path
        
    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)
    
    for filename in filenames:
        print("file is", filename)
        input = pil_image.open(filename).convert('RGB')
        print("Input size:", input.size)

        #buffer = io.BytesIO()
        #input.save(buffer, format='jpeg', quality=opt.jpeg_quality)
        #input = pil_image.open(buffer)
        #input.save(os.path.join(opt.outputs_dir, '{}_jpeg_q{}.png'.format(filename, opt.jpeg_quality)))

        input = transforms.ToTensor()(input).unsqueeze(0).to(device)
        output_path = os.path.join(opt.outputs_dir, '{}-{}.jpeg'.format(filename.split("/")[-1].split(".")[0], "edar"))
        
        if not os.path.exists(output_path):
            with torch.no_grad():
                pred = model(input)[-1]

            pred = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
            output = pil_image.fromarray(pred, mode='RGB')
            print("Output size", output.size)
            print("Output dir is", opt.outputs_dir)
            output.save(output_path)
            #print(os.path.join(opt.outputs_dir, '{}_{}.png'.format(filename, "EDAR")))
            #print("Output saved")

