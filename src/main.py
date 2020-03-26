# coding: utf-8
import os, warnings, logging, random, torch, torchvision, argparse
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from capsnet import *
from train import *
from datasets import *

warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

seed = 666
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(args):

    ''' --------------------------- LOAD DATA -------------------------------'''

    if args.dataset == 'smallnorb':
        dataset = 'smallNORB_48'
        working_dir = os.path.join(os.path.split(os.getcwd())[0], 'data', dataset)
        dataset_paths = {'train': os.path.join(working_dir,'train'),
                        'test':  os.path.join(working_dir,'test')}

        dataloaders = smallnorb(args, dataset_paths)

        args.class_names = ('car', 'animal', 'truck', 'airplane', 'human') # 0,1,2,3,4 labels
        args.n_channels, args.n_classes = 2, 5

    elif args.dataset == 'mnist':
        dataset = 'MNIST'
        working_dir = os.path.join(os.path.split(os.getcwd())[0], 'data', dataset)
        dataset_paths = {'train': os.path.join(working_dir,'train'),
                         'test':  os.path.join(working_dir,'test')}

        dataloaders = mnist(args, dataset_paths)

        args.class_names = ('zero', 'one', 'two', 'three', 'four',
            'five', 'six', 'seven', 'eight', 'nine') # 0,1,2,3,4,5,6,7,8,9 labels
        args.n_channels, args.n_classes = 1, 10

    elif args.dataset == 'svhn':
        dataset = 'SVHN'
        working_dir = os.path.join(os.path.split(os.getcwd())[0], 'data', dataset)
        dataset_paths = {'train': os.path.join(working_dir,'train'),
                         #'extra': os.path.join(working_dir,'extra'),
                         'test':  os.path.join(working_dir,'test')}

        dataloaders = svhn(args, dataset_paths)

        args.class_names = ('zero', 'one', 'two', 'three',
            'four', 'five', 'six', 'seven', 'eight', 'nine') # 0,1,2,3,4,5,6,7,8,9 labels
        args.n_channels, args.n_classes = 3, 10

    assert args.arch[-1] == args.n_classes, \
        'Set number of capsules in last layer to number of classes using --arch flag.'

    ''''----------------------- EXPERIMENT CONFIG ---------------------------'''

    # check number of models already saved in 'experiments' dir, add 1 to get new model number
    experiments_dir = os.path.join(os.path.split(os.getcwd())[0], 'experiments')
    os.makedirs(experiments_dir, exist_ok=True)
    model_num = len(os.listdir(experiments_dir)) + 1

    # create all save dirs
    model_dir = os.path.join(os.path.split(os.getcwd())[0],
        'experiments', 'Model_'+str(model_num))
    args.summaries_dir = os.path.join(model_dir, 'summaries')
    args.checkpoint_dir = os.path.join(model_dir, 'checkpoint.pt')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(args.summaries_dir, exist_ok=True)

    # save hyperparameters in .txt file
    with open(os.path.join(model_dir, 'hyperparams.txt'), 'w') as logs:
        for key, value in vars(args).items():
            logs.write('--{0}={1} '.format(str(key), str(value)))

    # reset root logger
    [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]
    # info logger for saving command line outputs during training
    logging.basicConfig(level=logging.INFO, format='%(message)s',
         handlers=[logging.FileHandler(os.path.join(model_dir, 'trainlogs.txt')),
            logging.StreamHandler()])

    ''' -------------------------- INIT MODEL -------------------------------'''

    model = CapsuleNet(args)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    print('\nUsing', torch.cuda.device_count(), 'GPU(s).\n')

    model.to(device)

    # print some info on architecture
    logging.info('-'*70)
    logging.info('{:>25} {:>27} {:>15}'.format('Layer.Parameter', 'Shape', 'Param #'))
    logging.info('-'*70)

    for param in model.state_dict():
        p_name = param.split('.')[-2]+'.'+param.split('.')[-1]
        if p_name[:2] != 'BN': # don't print batch norm layers
            logging.info('{:>25} {:>27} {:>15}'.format(
            p_name,
            str(list(model.state_dict()[param].squeeze().size())),
            '{0:,}'.format(np.product(list(model.state_dict()[param].size())))))
    logging.info('-'*70)

    logging.info('\nTotal params: {:,}\n\nSummaries dir: {}\n'.format(
        sum(p.numel() for p in model.parameters()),
        args.summaries_dir))

    for key, value in vars(args).items():
        logging.info('--{0}: {1}'.format(str(key), str(value)))

    ''' ---------------------- TRAIN/EVALUATE MODEL -------------------------'''

    if not args.inference:
        args.writer = SummaryWriter(args.summaries_dir) # initialise summary writer
        score = train(model, dataloaders, args)
        return score # loss of test set
    else:
        model.load_state_dict(torch.load(args.load_checkpoint_dir)) # load best saved model
        test_loss, test_acc = evaluate(model, args, dataloaders['test'])
        print('Test: Loss {:.4f} - Acc. {:.4f}'.format(test_loss, test_acc))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='smallnorb')
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--routing_iter', type=int, default=3)
    parser.add_argument('--pose_dim', type=int, default=4)
    parser.add_argument('--padding', type=int, default=4)
    parser.add_argument('--brightness', type=float, default=0)
    parser.add_argument('--contrast', type=float, default=0)
    parser.add_argument('--patience', default=1e+4)
    parser.add_argument('--crop_dim', type=int, default=32)
    parser.add_argument('--arch', nargs='+', type=int, default=[64,8,16,16,5]) # architecture n caps
    parser.add_argument('--load_checkpoint_dir', default='NA')
    parser.add_argument('--inference', dest='inference', action='store_true')
    parser.add_argument('--test_affnist', dest='test_affNIST', action='store_true')
    parser.set_defaults(inference=False)
    parser.set_defaults(test_affNIST=False)

    score = main(parser.parse_known_args()[0])
