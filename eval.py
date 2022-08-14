import time
import math
import argparse
import torch
import tensorboard_logger

from vit import ViT
from model import MAE
from lars import LARS
from model import EvalNet, LabelSmoothing
from util import *

# for re-produce
set_seed(0)
# def build_model(args):
#     '''
#     build EvalNet model and restore weights
#     :param args: model args
#     :return: model
#     '''
#     # build encoder
#     v = ViT(image_size=args.image_size,
#             patch_size=args.patch_size,
#             num_classes=args.n_class,
#             dim=args.vit_dim,
#             depth=args.vit_depth,
#             heads=args.vit_heads,
#             mlp_dim=args.vit_mlp_dim).to(args.device)
#
#     # build linear probing
#     enet = EvalNet(encoder=v,
#                    n_class=args.n_class,
#                    masking_ratio=0,
#                    device=args.device).to(args.device)
#
#     # restore weights
#     state_dict_encoder = enet.encoder.state_dict()
#     state_dict_loaded = torch.load(args.ckpt)['model']
#     for k in state_dict_encoder.keys():
#         state_dict_encoder[k] = state_dict_loaded['encoder.' + k]
#     enet.encoder.load_state_dict(state_dict_encoder)
#
#     return enet
def build_model(args):
    '''
    build MAE model.
    :param args: model args
    :return: model
    '''
    # build model
    v = ViT(image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=args.n_class,
            dim=args.vit_dim,
            depth=args.vit_depth,
            heads=args.vit_heads,
            mlp_dim=args.vit_mlp_dim)

    mae = MAE(encoder=v,
              masking_ratio=args.masking_ratio,
              decoder_dim=args.decoder_dim,
              decoder_depth=args.decoder_depth,
              device=args.device).to(args.device)

    return mae

def test(args, model=None, ckpt_path=None, data_loader=None):
    '''
    train the model
    :param args: args
    :param model: the test model
    :param ckpt_path: checkpoint path, if model is given, this is deactivated
    :param data_loader: data loader
    :return: accuracy
    '''

    # load data
    if data_loader is None:
        data_loader, args.n_class = load_data(args.data_dir,
                                              args.data_name,
                                              image_size=args.image_size,
                                              batch_size=args.batch_size,
                                              n_worker=args.n_worker,
                                              is_train=False)

    # restore mae model
    assert model is not None or ckpt_path is not None
    if model is None:
        model = build_model(args)
        model = load_ckpt(model, ckpt_path)
    model.eval()
    print(model)
    # test
    accs = AverageMeter()
    with torch.no_grad():
        for idx, (images, targets) in enumerate(data_loader):
            # put images into device
            images = images.to(args.device)
            # forward
            output = model(images)
            print(output.shape)
            # eval
            _, y_pred = torch.max(output, dim=1)
            acc = accuracy(targets.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
            # record
            accs.update(acc, args.batch_size)

    return accs.avg
def default_args(data_name, trail=0, ckpt_file='last.ckpt'):
    '''
    for default parameters. tune them upon your options
    :param data_name: dataset name, such as 'imagenet'
    :param trail: an int indicator to specify different runnings
    :param ckpt_file: path of the trained MAE model
    :return:
    '''
    # params
    args = argparse.ArgumentParser().parse_args()

    # device
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # data
    args.data_dir = '../KDGen/data/'
    args.data_name = data_name
    args.image_size = 256
    args.n_worker = 2

    # model
    # - use ViT-Base whose parameters are referred from "Dosovitskiy et al. An Image is Worth 16x16 Words: Transformers
    # - for Image Recognition at Scale. ICLR 2021. https://openreview.net/forum?id=YicbFdNTTy".
    args.patch_size = 32
    args.vit_dim = 768
    args.vit_depth = 12
    args.vit_heads = 12
    args.vit_mlp_dim = 3072
    args.masking_ratio = 0.75  # the paper recommended 75% masked patches
    args.decoder_dim = 512  # paper showed good results with 512
    args.decoder_depth = 8  # paper showed good results with 8

    # train
    args.batch_size = 4096 // 128
    args.epochs = 1
    args.base_lr = 1.5e-4
    args.lr = args.base_lr * args.batch_size / 256
    args.weight_decay = 5e-2
    args.momentum = (0.9, 0.95)
    args.epochs_warmup = 40
    args.warmup_from = 1e-4
    args.lr_decay_rate = 1e-2
    eta_min = args.lr * (args.lr_decay_rate ** 3)
    args.warmup_to = eta_min + (args.lr - eta_min) * (1 + math.cos(math.pi * args.epochs_warmup / args.epochs)) / 2
    # extra
    args.label_smoothing = True
    args.smoothing = 0.1

    # print and save
    args.print_freq = 5
    args.eval_freq = 5

    # tensorboard
    args.tb_folder = os.path.join('log', '{}_{}'.format(args.data_name, trail))
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)

    # ckpt
    args.ckpt_folder = os.path.join('ckpt', '{}_{}'.format(args.data_name, trail))
    args.ckpt = os.path.join(args.ckpt_folder, ckpt_file)

    return args

if __name__ == '__main__':

    data_name = 'cifar10'
    # train(default_args(data_name))
    # model = build_model(default_args(data_name))
    test(args=default_args(data_name),model=None,ckpt_path='ckpt/cifar10_0/last.ckpt',data_loader=None)
