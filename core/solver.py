"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
from adaptiveAugmentation import AdaptiveAugmentation
import os
from os.path import join as ospj
import time
import datetime

import torch
import torch.nn as nn

from core.losses import compute_d_loss, compute_g_loss
from core.model import build_model
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils



class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets, self.nets_ema = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        if args.mode == 'train':
            self.optims = {}
            for net in self.nets.keys():
                if net == 'fan':
                    continue
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.pt'), data_parallel=False, **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.pt'), data_parallel=False, **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.pt'), **self.optims)]
        else:
            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.pt'), data_parallel=False, **self.nets_ema)]

        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders):
        args = self.args
        nets = self.nets
        adaptiveAug = AdaptiveAugmentation(batchSize=args.batch_size) if args.ada else None
        nets_ema = self.nets_ema
        optims = self.optims

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders["src"], loaders["ref"], args.latent_dim, 'train')
        fetcher_val = InputFetcher(loaders["val"], None, args.latent_dim, 'val')
        inputs_val = next(fetcher_val)

        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds

        print('Start training...')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            # fetch images and labels
            inputs = next(fetcher)
            x_real, y_org = inputs["x_src"], inputs["y_src"]
            x_ref, x_ref2, y_trg = inputs["x_ref"], inputs["x_ref2"], inputs["y_ref"]
            z_trg, z_trg2 = inputs["z_trg"], inputs["z_trg2"]



            # train the discriminator
            d_loss, d_losses_latent = compute_d_loss(
                nets, args, x_real, y_org, y_trg, z_trg=z_trg, adaptiveAug=adaptiveAug, layerWiseComposition=args.layerWiseComposition)
            self._reset_grad()
            d_loss.backward()
            optims["discriminator"].step()

            d_loss, d_losses_ref = compute_d_loss(
                nets, args, x_real, y_org, y_trg, x_ref=x_ref, adaptiveAug=adaptiveAug, layerWiseComposition=args.layerWiseComposition)
            self._reset_grad()
            d_loss.backward()
            optims["discriminator"].step()

            # train the generator
            g_loss, g_losses_latent = compute_g_loss(
                nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], attentionGuided=args.attentionGuided, layerWiseComposition=args.layerWiseComposition)
            self._reset_grad()
            g_loss.backward()
            optims["generator"].step()
            optims["mapping_network"].step()
            optims["style_encoder"].step()

            g_loss, g_losses_ref = compute_g_loss(
                nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2],attentionGuided=args.attentionGuided, layerWiseComposition=args.layerWiseComposition)
            self._reset_grad()
            g_loss.backward()
            optims["generator"].step()
            # compute moving average of network parameters
            moving_average(nets["generator"], nets_ema["generator"], beta=0.999)
            moving_average(nets["mapping_network"], nets_ema["mapping_network"], beta=0.999)
            moving_average(nets["style_encoder"], nets_ema["style_encoder"], beta=0.999)

            # decay weight for diversity sensitive loss
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]

                print("================= CURRENT ITERATION : {} / {} ================= ".format(i + 1,
                                                                                                args.total_iters))
                print("Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters))
                if adaptiveAug is not None:
                    print("ADA VALUE : {}".format(adaptiveAug.augmentProb))
                allLosses = {key : {} for key in ["D/latent", "D/ref   ", "G/latent", "G/ref   "]}
                for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                        list(allLosses.keys())):
                    for key, value in loss.items():
                        allLosses[prefix][key] = value
                # allLosses['G/lambda_ds'] = args.lambda_ds
                for key, value in allLosses.items():
                    printOut = "{} | ".format(key)
                    for lossType, lossValue in value.items():
                        printOut += "{} : {:.4f}  ".format(lossType,
                                                     lossValue)
                    print(printOut)

                print("G/lambda_ds : {}".format(args.lambda_ds))
                # log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in allLosses.items()])
                # print(log)

            # generate images for debugging
            if (i+1) % args.sample_every == 0:
                os.makedirs(args.sample_dir, exist_ok=True)
                utils.debug_image(nets_ema, args, inputs=inputs_val, step=i+1)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)

            # # compute FID and LPIPS if necessary
            # if (i+1) % args.eval_every == 0:
            #     calculate_metrics(nets_ema, args, i+1, mode='latent')
            #     calculate_metrics(nets_ema, args, i+1, mode='reference')

    # @torch.no_grad()
    # def sample(self, loaders):
    #     args = self.args
    #     nets_ema = self.nets_ema
    #     os.makedirs(args.result_dir, exist_ok=True)
    #     self._load_checkpoint(args.resume_iter)
    #
    #     src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))
    #     ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test'))
    #
    #     fname = ospj(args.result_dir, 'reference.jpg')
    #     print('Working on {}...'.format(fname))
    #     utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname)
    #
    #     fname = ospj(args.result_dir, 'video_ref.mp4')
    #     print('Working on {}...'.format(fname))
    #     utils.video_ref(nets_ema, args, src.x, ref.x, ref.y, fname)
    #
    # @torch.no_grad()
    # def evaluate(self):
    #     args = self.args
    #     nets_ema = self.nets_ema
    #     resume_iter = args.resume_iter
    #     self._load_checkpoint(args.resume_iter)
    #     calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')
    #     calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


