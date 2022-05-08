import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F


def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None, adaptiveAug = None, layerWiseComposition = False):
    assert (z_trg is None) != (x_ref is None)
    # with real images

    # with fake images
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets["mapping_network"](z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets["style_encoder"](x_ref, y_trg)

        if layerWiseComposition:
            spatialAttentionMap, foreground = nets["generator"](x_real, s_trg, masks=masks)

            x_fake = torch.mul(x_real, 1-spatialAttentionMap) + torch.mul(foreground, spatialAttentionMap)
        else:
            x_fake = nets["generator"](x_real, s_trg, masks=masks)


        if adaptiveAug is not None:
            x_fake_aug = adaptiveAug.forward(x_fake)
        else:
            x_fake_aug = x_fake

    x_real.requires_grad_()
    if adaptiveAug is not None:
        x_real_aug = adaptiveAug.forward(x_real)
    else:
        x_real_aug = x_real



    #### DISCRIMINATOR ON REAL

    # x_real.requires_grad()
    out = nets["discriminator"](x_real_aug, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)
    # DISCRIMINATOR ON FAKE

    if adaptiveAug is not None:
        with torch.no_grad():
            adaptiveAug.update(out.detach().clone())

    out = nets["discriminator"](x_fake_aug, y_trg)
    loss_fake = adv_loss(out, 0)



    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, {"real":loss_real.item(),"fake":loss_fake.item(),"reg" : loss_reg.item()}


def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None, adaptiveAug = None, attentionGuided = False, layerWiseComposition = False):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = nets["mapping_network"](z_trg, y_trg)
    else:
        s_trg = nets["style_encoder"](x_ref, y_trg)



    x_real.requires_grad = True

    if layerWiseComposition:
        spatialAttentionMap, foreground = nets["generator"](x_real, s_trg, masks=masks)
        x_fake = torch.mul(x_real, 1 - spatialAttentionMap) + torch.mul(foreground, spatialAttentionMap)
    else:
        x_fake = nets["generator"](x_real, s_trg, masks=masks)
    # x_fake = nets["generator"](x_real, s_trg)
    if adaptiveAug is not None:
        x_fake_aug = adaptiveAug.forward(x_fake)

    else:
        x_fake_aug = x_fake
    out = nets["discriminator"](x_fake_aug, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    s_pred = nets["style_encoder"](x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets["mapping_network"](z_trg2, y_trg)
    else:
        s_trg2 = nets["style_encoder"](x_ref2, y_trg)
    # x_fake2 = nets["generator"](x_real, s_trg2)
    if layerWiseComposition:
        spatialAttentionMap2, foreground2 = nets["generator"](x_real, s_trg2, masks=masks)
        x_fake2 = torch.mul(x_real, 1 - spatialAttentionMap2) + torch.mul(foreground2, spatialAttentionMap2)
    else:
        x_fake2 = nets["generator"](x_real, s_trg2, masks=masks)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    s_org = nets["style_encoder"](x_real, y_org)

    if args.discGuidedLayerWiseComp:
        with torch.no_grad():
            # print("attention guided layer wise comp")
            featureMap = nets["discriminator"](x_real, y_trg, returnFeatureMap=True)
        featureMap = featureMap.detach()
    # if attentionGuided:
    #     with torch.no_grad():
    #         print("attention guided")
    #         featureMap = nets["discriminator"](x_real, y_trg,returnFeatureMap = True)
    #         x_real = torch.mul(x_real, featureMap)

    if layerWiseComposition:
        reconstructedSpatialAttentionMap, reconstructedForeground = nets["generator"](x_fake, s_org, masks=masks)
        x_rec = torch.mul(x_real, 1 - reconstructedSpatialAttentionMap) + torch.mul(reconstructedForeground, reconstructedSpatialAttentionMap)
    else:
        x_rec = nets["generator"](x_fake, s_org, masks=masks)
    # x_rec = nets["generator"](x_fake, s_org)
    # if attentionGuided:
    #     with torch.no_grad():
    #         # featureMap = nets["discriminator"](x_real, y_trg,returnFeatureMap = True)
    #         x_rec = torch.mul(x_rec, featureMap)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))
    if layerWiseComposition:
        loss_sd_cyc = torch.mean(torch.abs(spatialAttentionMap - reconstructedSpatialAttentionMap))
        loss_sd_con = torch.mean(torch.abs(spatialAttentionMap) + torch.abs(reconstructedSpatialAttentionMap))
    else:
        loss_sd_cyc = torch.FloatTensor([0])
        loss_sd_con = torch.FloatTensor([0])


    if args.discGuidedLayerWiseComp:
        loss_sd_disc = torch.mean(torch.abs(spatialAttentionMap - featureMap))
    else:
        loss_sd_disc = torch.FloatTensor([0]).cuda()



    # CONFIG C USED 2 FOR SD DISC LAMBDA WITH LAYER NUMBER 2
    if layerWiseComposition:
        loss = loss_adv + args.lambda_sty * loss_sty \
            - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc + args.lambda_sd_cyc * loss_sd_cyc + args.lambda_sd_con * loss_sd_con + args.lambda_sd_disc * loss_sd_disc
    else:
        loss = loss_adv + args.lambda_sty * loss_sty \
            - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc
    return loss, {"adv": loss_adv.item(),
                  "sty":loss_sty.item(),
                  "ds":loss_ds.item(),
                  "cyc":loss_cyc.item(),
                  "sd_cyc" : loss_sd_cyc.item(),
                  "sd_con" : loss_sd_con.item(),
                  "loss_sd_disc" : loss_sd_disc.item()}


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg