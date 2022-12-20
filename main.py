import argparse
import warnings
import os
import datetime
import itertools
import random
import time
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
from os.path import join, basename, dirname, split

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

from data.dataset import CustomDataset, TestDataset
from models.generator import Generator
from models.discriminator import Discriminator
from utils.utils import *


warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--start-iters', type=int, default=0, help='number of total iterations for training D')
parser.add_argument('--num-iters', type=int, default=200000, help='number of total iterations for training D')
parser.add_argument('--num-iters-decay', type=int, default=100000, help='number of iterations for decaying lr')
parser.add_argument('--n-critic', type=int, default=5, help='number of D updates per each G update')
parser.add_argument('--test-iters', type=int, default=100000, help='test model from this step')

parser.add_argument('--model-save-step', type=int, default=1000)
parser.add_argument('--sample-step', type=int, default=1000)
parser.add_argument('--lr-update-step', type=int, default=1000)

parser.add_argument('--workers', type=int, default=4, help='Number of workers in dataset loader (default: 4)')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size in training (default: 32)')
parser.add_argument('--sampling-rate', type=int, default=16000, help='sampling rate')
parser.add_argument('--lr', default=1e-4)

# StarGAN
parser.add_argument('--num-speakers', type=int, default=10, help='dimension of speaker labels')
parser.add_argument('--lambda-cls', type=float, default=10, help='weight for domain classification loss')
parser.add_argument('--lambda-rec', type=float, default=10, help='weight for reconstruction loss')
parser.add_argument('--lambda-gp', type=float, default=10, help='weight for gradient penalty')

# Dir Path
parser.add_argument('--train-dir', type=str, default="../dataset/mc/train")
parser.add_argument('--test-dir', type=str, default="../dataset/mc/test")
parser.add_argument('--wav-dir', type=str, default="../dataset/VCTK-Corpus/wav16")
parser.add_argument('--sample-dir', type=str, default='./samples')

parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--print-freq', type=int, default=1)
parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help="model_args.resume")

# Distributed
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    summary = SummaryWriter()
    print("clear")

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # A: Noisy Sample / B: Clean_sample
    generator = Generator(n_speakers=args.num_speakers)
    discriminator = Discriminator(n_speakers=args.num_speakers)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            print("using Distributed train")
            torch.cuda.set_device(args.gpu)
            generator.cuda(args.gpu)
            discriminator.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True)
            discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True)

        else:
            generator.cuda()
            discriminator.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            generator = torch.nn.parallel.DistributedDataParallel(generator)
            discriminator = torch.nn.parallel.DistributedDataParallel(discriminator)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        generator = generator.cuda(args.gpu)
        discriminator = discriminator.cuda(args.gpu)

    else:
        generator = torch.nn.DataParallel(generator).cuda()
        discriminator = torch.nn.DataParallel(discriminator).cuda()

    # Optimizer / criterion / Scheduler
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    generator_optimizer = torch.optim.Adam(generator.parameters(),
                                           lr=args.lr,
                                           betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),
                                               lr=args.lr,
                                               betas=(0.5, 0.999))

    # Resume
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_iters = checkpoint['start_iters']

            generator.load_state_dict(checkpoint['G'])
            discriminator.load_state_dict(checkpoint['D'])

            generator_optimizer.load_state_dict(checkpoint['G_optimizer'])
            discriminator_optimizer.load_state_dict(checkpoint['D_optimizer'])

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Dataset / DataLoader
    train_dataset = CustomDataset(args.train_dir)
    test_dataset = TestDataset(data_dir=args.test_dir,
                               wav_dir=args.wav_dir,
                               src_spk='p262',
                               trg_spk='p272')

    # Sampler
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    # DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                               drop_last=True)

    # Summary Print
    print("[Generator]")
    print(generator)
    print("[Discriminator]")
    print(discriminator)
    G_p = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    D_p = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)

    param = (G_p + D_p)
    print("Total Param: ", param)
    print("G_Param: ", G_p)
    print("D_Param: ", D_p)

    # train
    train(train_loader=train_loader,
          test_loader=test_dataset,
          generator=generator,
          discriminator=discriminator,
          generator_optimizer=generator_optimizer,
          discriminator_optimizer=discriminator_optimizer,
          criterion=criterion,
          args=args,
          summary=summary)


def train(train_loader, test_loader,
          generator, discriminator,
          generator_optimizer, discriminator_optimizer,
          criterion,
          args, summary):

    data_iter = iter(train_loader)
    test_wavfiles = test_loader.get_batch_test_data(batch_size=4)
    test_wavs = [load_wav(wavfile) for wavfile in test_wavfiles]

    cpsyn_flag = [True, False][0]
    g_lr = args.g_lr
    d_lr = args.d_lr

    start_time = time.time()

    for i in range(args.start_iters, args.num_iters):
        generator.train()
        discriminator.train()

        try:
            mc_real, spk_label_org, spk_c_org = next(data_iter)
        except:
            data_iter = iter(train_loader)
            mc_real, spk_label_org, spk_c_org = next(data_iter)

        mc_real.unsqueeze_(1)  # (B, D, T) -> (B, 1, D, T) for conv2d

        # Generate target domain labels randomly.
        # spk_label_trg: int,   spk_c_trg:one-hot representation
        spk_label_trg, spk_c_trg = sample_spk_c(args, mc_real.size(0))

        mc_real = mc_real.cuda(args.gpu, non_blocking=True)  # Input MCEP
        spk_label_org = spk_label_org.cuda(args.gpu, non_blocking=True)  # Original Spk id
        spk_c_org = spk_c_org.cuda(args.gpu, non_blocking=True)  # Original Spk acc conditioning

        spk_label_trg = spk_label_trg.cuda(args.gpu, non_blocking=True)  # Target Spk id
        spk_c_trg = spk_c_trg.cuda(args.gpu, non_blocking=True)  # target Spk conditioning

        ######################
        # Train Critic
        # WGAN-GP (Wasserstein Gradient Penalty)
        ######################
        out_src, out_cls_spks = discriminator(mc_real)

        # Real MCEP Loss for D
        d_loss_real = - torch.mean(out_src)
        d_loss_cls_spks = criterion(out_cls_spks, spk_label_org)  # StarGAN 참고

        # Fake MCEP Loss for D
        mc_fake = generator(mc_real, spk_c_trg)
        fake_out_src, _ = discriminator(mc_fake.detach())
        d_loss_fake = torch.mean(fake_out_src)

        # Gradient Penalty
        alpha = torch.rand(mc_real.size(0), 1, 1, 1).cuda(args.gpu)
        x_hat = (alpha * mc_real.data + (1 - alpha) * mc_fake.data).requires_grad_(True)
        out_src, _ = discriminator(x_hat)
        d_loss_gp = gradient_penalty(args, out_src, x_hat)

        discriminator_loss = d_loss_real + d_loss_fake + args.lambda_cls * d_loss_cls_spks\
                             + d_loss_gp * args.lambda_gp

        discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        discriminator_optimizer.step()

        if args.gpu == 0:
            summary.add_scalar('Train/D_loss', discriminator_loss.item(), i)
            summary.add_scalar('Train/D_loss_real', d_loss_real.item(), i)
            summary.add_scalar('Train/D_loss_fake', d_loss_fake.item(), i)
            summary.add_scalar('Train/D_loss_cls', (args.lambda_cls * d_loss_cls_spks).item(), i)
            summary.add_scalar('Train/D_loss_gp', (d_loss_gp * args.lambda_gp).item(), i)

        ####################
        # Train Generator
        ###################
        if (i + 1) % args.n_critic == 0:
            # source to target
            mc_fake = generator(mc_real, spk_c_trg)
            out_src, out_cls_spks = discriminator(mc_fake)

            # Generator Loss
            g_loss_fake = - torch.mean(out_src)
            g_loss_cls_spks = criterion(out_cls_spks, spk_label_trg)

            # Reconstruction
            mc_rec = generator(mc_fake, spk_c_org)
            g_loss_rec = torch.mean(torch.abs(mc_real - mc_rec))  # L1 Loss for reconstruction Loss

            generator_loss = g_loss_fake + args.lambda_rec * g_loss_rec + args.lambda_cls * g_loss_cls_spks

            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

            if args.gpu == 0:
                summary.add_scalar('Train/G_loss', generator_loss.item(), i)
                summary.add_scalar('Train/G_loss_rc', (args.lambda_rec * g_loss_rec).item(), i)
                summary.add_scalar('Train/G_loss_cls', (args.lambda_cls * g_loss_cls_spks).item(), i)

        else:
            generator_loss = 0

        ################
        # Print Log
        ################
        if i % args.print_freq == 0:
            elapsed = datetime.timedelta(seconds=time.time() - start_time)
            print(f" Iteration [{i}/{args.num_iters}] | D_loss: {discriminator_loss} | G_loss: {generator_loss} | Elapsed: {elapsed}")

        #####################################
        # Save Model Parameters % Sampling
        #####################################
        if (i + 1) % args.model_save_step == 0:
            torch.save({
                'start_iters': i + 1,
                'G': generator.state_dict(),
                'D': discriminator.state_dict(),
                'G_optimizer': generator_optimizer.state_dict(),
                'D_optimizer': discriminator_optimizer.state_dict(),
            }, "saved_models/checkpoint_%d.pth" % (i + 1))

        if (i + 1) % args.sample_step == 0:
            sampling_rate = 16000
            num_mcep = 36
            frame_period = 5

            generator.eval()
            discriminator.eval()
            with torch.no_grad():
                for idx, wav in tqdm(enumerate(test_wavs)):
                    wav_name = basename(test_wavfiles[idx])
                    f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
                    f0_converted = pitch_conversion(f0=f0,
                                                    mean_log_src=test_loader.logf0s_mean_src,
                                                    std_log_src=test_loader.logf0s_std_src,
                                                    mean_log_target=test_loader.logf0s_mean_trg,
                                                    std_log_target=test_loader.logf0s_std_trg)
                    coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
                    coded_sp_norm = (coded_sp - test_loader.mcep_mean_src) / test_loader.mcep_std_src
                    coded_sp_norm_tensor = torch.FloatTensor(coded_sp_norm.T).unsqueeze_(0).unsqueeze_(1).cuda(args.gpu)
                    conds = torch.FloatTensor(test_loader.spk_c_trg).cuda(args.gpu)
                    coded_sp_converted_norm = generator(coded_sp_norm_tensor, conds).data.cpu().numpy()
                    coded_sp_converted = np.squeeze(coded_sp_converted_norm).T * test_loader.mcep_std_trg + test_loader.mcep_mean_trg
                    coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                    wav_transformed = world_speech_synthesis(f0=f0_converted, coded_sp=coded_sp_converted,
                                                             ap=ap, fs=sampling_rate, frame_period=frame_period)
                    sf.write(
                        join(args.sample_dir, str(i + 1) + '-' + wav_name.split('.')[0] + '-vcto-{}'.format(
                            test_loader.trg_spk) + '.wav'),
                        wav_transformed, sampling_rate)

                    if cpsyn_flag:
                        wav_cpsyn = world_speech_synthesis(f0=f0, coded_sp=coded_sp,
                                                           ap=ap, fs=sampling_rate, frame_period=frame_period)
                        sf.write(join(args.sample_dir, 'cpsyn-' + wav_name), wav_cpsyn, sampling_rate)
                cpsyn_flag = False

        ###########################
        # Decay Learning Rates
        ###########################
        if (i + 1) % args.lr_update_step == 0 and (i + 1) > (args.num_iters - args.num_iters_decay):
            g_lr -= (args.g_lr / float(args.num_iters_decay))
            d_lr -= (args.d_lr / float(args.num_iters_decay))
            update_lr(g_optimizer=generator_optimizer,
                      d_optimizer=discriminator_optimizer,
                      g_lr=g_lr,
                      d_lr=d_lr)
            print(f"[Update Learning Rate] g_lr: {g_lr} | d_lr: {d_lr}")


if __name__ == "__main__":
    main()
