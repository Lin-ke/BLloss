import torch
import os
import numpy as np
from datasets.crowd import Crowd
from models.vgg import vgg19
import argparse
from matplotlib import pyplot as plt
args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='./dataset/UCF-Train-Val-Test',
                        help='training data directory')
    parser.add_argument('--save-dir', default='',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    datasets = Crowd(os.path.join(args.data_dir, 'test'), 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=False)
    model = vgg19()
    # using cpu
    device = torch.device('cpu')
    model.to(device)
    # checkpoint = torch.load(os.path.join(args.save_dir, '1.tar'),map_location="cpu")
    # model.load_state_dict(checkpoint['model_state_dict'], device)
    checkpoint = torch.load( 'best_model.pth',map_location="cpu")
    epoch_minus = []

    for inputs, count, name in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            temp_minu = count[0].item() - torch.sum(outputs).item()
            print(name, temp_minu, count[0].item(), torch.sum(outputs).item())
            epoch_minus.append(temp_minu)
            dm = outputs.squeeze().detach().cpu().numpy()
            plt.imshow(dm, cmap=plt.cm.jet)
            plt.savefig("./pics/"+name[0]+".jpg")
    # visual
    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    print(log_str)
