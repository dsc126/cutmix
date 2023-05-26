import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.utils import save_image, make_grid
from theconf.argument_parser import ConfigArgumentParser
from torch.utils.data.dataset import Subset, Dataset
from cutmix.utils import onehot, rand_bbox

import random
from copy import deepcopy

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = ConfigArgumentParser(conflict_handler='resolve')
parser.add_argument('--cifarpath', default='/data/private/pretrainedmodels/', type=str)
parser.add_argument('--only-eval', action='store_true')

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)


class PDemo(Dataset):
    def __init__(self, dataset, num_class, num_mix=1, beta=1., prob=1.0):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        global img_mu, img_co, img_cm
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb)
        img_b = torch.FloatTensor(img.shape).fill_(0)

        for _ in range(self.num_mix):
            lam = 0.5  # fix lam in demo
            rand_index = random.choice(range(len(self)))

            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = onehot(self.num_class, lb2)

            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)

            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            img_mu = img * lam + img2 * (1. - lam)
            img_co = deepcopy(img)
            img_co[:, bbx1:bbx2, bby1:bby2] = img_b[:, bbx1:bbx2, bby1:bby2]
            img_cm = deepcopy(img)
            img_cm[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            # lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        return img_mu, img_co, img_cm

    def __len__(self):
        return len(self.dataset)


def save(train_loader, n: int = 3):
    for i in range(n):
        img0, img1, img2 = train_loader.__getitem__(i)
        save_image(img0, './pic/mu' + str(i) + '.png')
        save_image(img1, './pic/co' + str(i) + '.png')
        save_image(img2, './pic/cm' + str(i) + '.png')


def main():
    global args, best_err1, best_err5
    args = parser.parse_args()

    if args.dataset.startswith('cifar'):
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
        )

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        ds_train = datasets.CIFAR100(args.cifarpath, train=True, download=True, transform=transform_train)

        train_loader = PDemo(ds_train, 100, beta=args.cutmix_beta, prob=args.cutmix_prob, num_mix=args.cutmix_num)

        save(train_loader)

    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))


if __name__ == '__main__':
    main()
