import math
from math import log, sqrt, cos,sin
import random, torchvision, torch

from matplotlib import pyplot as plt


class AdaptiveAugmentation():

    def scale3D(self, sY, sX, sZ):
        return torch.FloatTensor([[sX, 0,  0,  0],
                                  [0,  sY, 0,  0],
                                  [0,  0,  sZ, 0],
                                  [0,  0,  0,  1]])



    def translate3D(self, tY, tX, tZ):
        return torch.FloatTensor([[1, 0, 0, tX],
                                  [0, 1, 0, tY],
                                  [0, 0, 1, tZ],
                                  [0, 0, 0, 1]])



    def __init__(self, batchSize, augmentProb = 0.0, targetRTVal = 0.6, numberUpdatesToAVG = 32):
        self.randomErase = torchvision.transforms.RandomErasing(p=1.0,
                                                                scale=(0.25,0.25),
                                                                ratio=(1,1),
                                                                value = 0)
        self.augmentProb = augmentProb
        self.targetRTVal = targetRTVal
        self.numberUpdatesToAVG = numberUpdatesToAVG
        self.signedTrainStack = []
        self.ADA_PROB_MOVE_RATE = (batchSize * numberUpdatesToAVG)/250000




    @torch.no_grad()
    def update(self, discriminatorRealResult):
        self.signedTrainStack.append(torch.sign(discriminatorRealResult).mean().item())
        if len(self.signedTrainStack) == self.numberUpdatesToAVG:
            rTValue = sum(self.signedTrainStack) / len(self.signedTrainStack)
            dirToMoveProb = torch.sign(torch.FloatTensor([rTValue - self.targetRTVal])).item()
            self.augmentProb += dirToMoveProb * self.ADA_PROB_MOVE_RATE
            self.augmentProb = min(max(0, self.augmentProb), 1)
            # print("NEW ADA AUGMENT : {}".format(self.augmentProb))
            # logging.info("ADA AUGMENT PROB : {}".format(augmentProb))
            self.signedTrainStack = []

    def forward(self, imageBatch):
        if self.augmentProb == 0.0:
            return imageBatch

        #print("========================================")
        batchSize, channelNo, height, width = imageBatch.size()
        newImages = []
        for imageIndex, eachImage in enumerate(imageBatch):
            # HORIZONTAL FLIP
            #print("H FLIP")
            if random.random() < self.augmentProb:
                eachImage = torchvision.transforms.functional.hflip(eachImage)


            # EACH IMAGE
            #TODO: Get this to work with non square images
            if random.random() < self.augmentProb:
                i = torch.distributions.uniform.Uniform(low = 0,
                                                        high = 3).sample().int().item()
                eachImage = torch.rot90(eachImage,
                                        k = i,
                                        dims=[1,2])
                #print("ROT 90 ")

            # Integer translation
            if random.random() < self.augmentProb:
                #print("INT TRANSLATION ")
                tX, tY = torch.distributions.uniform.Uniform(low=-0.125,
                                                             high = 0.125).sample((2,))
                tX = (tX * width).int().item()
                tY = (tY * height).int().item()
                if tX < 0:
                    padRight = -tX
                    padLeft = 0
                elif tX > 0:
                    padLeft = tX
                    padRight = 0
                else:
                    padLeft = 0
                    padRight = 0

                if tY < 0:
                    padTop = -tY
                    padBot = 0
                elif tY > 0:
                    padBot = tY
                    padTop = 0
                else:
                    padTop = 0
                    padBot = 0
                eachImage = torchvision.transforms.functional.pad(img=eachImage,
                                                    padding=(padLeft, padTop, padRight, padBot),
                                                    padding_mode="reflect")


                eachImage = torchvision.transforms.functional.crop(img = eachImage,
                                                                   left = padRight,
                                                                   top = padBot,
                                                                   height = height,
                                                                   width = width)



            # ISOTROPIC SCALING
            if random.random() < self.augmentProb:
                s = torch.distributions.log_normal.LogNormal(loc = 0,
                                                             scale = (0.2 * log(2))).sample().item()
                #print("ISO SCALING")

                newHeight = int(s * height)
                newWidth = int(s * width)
                eachImage = torchvision.transforms.functional.resize(img = eachImage,
                                                                     size = (newHeight,
                                                                             newWidth))

                if s < 1:
                    verticalPadding = height - newHeight
                    horizontalPadding = width - newWidth
                    padLeft = math.floor(horizontalPadding/2)
                    padRight = padLeft
                    if horizontalPadding % 2 == 1:
                        padRight += 1


                    padTop = math.floor(verticalPadding/2)
                    padBot = padTop
                    if verticalPadding % 2 == 1:
                        padTop += 1
                    eachImage = torchvision.transforms.functional.pad(img = eachImage,
                                            padding = (padLeft, padTop, padRight, padBot),
                                            padding_mode="reflect")

                    if eachImage.size()[-1] != width or eachImage.size()[-2] != height:
                        raise Exception("IMAGE MEANT TO BE OF SIZE H : {} W : {} BUT GOT H : {} W : {}".format(height,
                                                                                                               width,
                                                                                                               eachImage.size()[1],
                                                                                                               eachImage.size()[2]))
                elif s > 1:
                    eachImage = torchvision.transforms.functional.center_crop(img = eachImage,
                                                                              output_size = (height,
                                                                                             width))




            # PRE ROTATION
            rotProb = 1-sqrt((1-self.augmentProb))

            if random.random() < rotProb and False:
                #print("PRE ROT ")
                theta = torch.distributions.uniform.Uniform(low = -180,
                                                            high = 180).sample().item()


                eachImage = torchvision.transforms.functional.pad(img = eachImage,
                                                                  padding = (int(height/2), int(width/2)),
                                                                  padding_mode = "reflect")
                eachImage = torchvision.transforms.functional.rotate(img = eachImage,
                                                                     angle = -theta)
                eachImage = torchvision.transforms.functional.center_crop(img=eachImage,
                                                                          output_size=(height,
                                                                                       width))


            if random.random() < self.augmentProb or False:
                #print("ANO SCALING")
                s = torch.distributions.log_normal.LogNormal(loc = 0,
                                                             scale = (0.2 * log(2))).sample().item()
                newWidth2 = int(s * width)
                newHeight2 = int((1/s) * height)
                eachImage = torchvision.transforms.functional.resize(img = eachImage,
                                                                     size = (newHeight2,
                                                                             newWidth2))

                if newHeight2 < height:
                    verticalPad2 = height - newHeight2
                    padTop2 = math.ceil(verticalPad2 / 2)
                    padBot2 = padTop2

                else:
                    padTop2, padBot2 = 0,0

                if newWidth2 < width:
                    horizontalPad2 = width - newWidth2
                    padLeft2 = math.ceil(horizontalPad2/2)
                    padRight2 = padLeft2
                else:
                    padLeft2, padRight2 = 0,0
                eachImage = torchvision.transforms.functional.pad(img = eachImage,
                                        padding = (padLeft2, padTop2, padRight2, padBot2),
                                        padding_mode="reflect")


                eachImage = torchvision.transforms.functional.center_crop(img=eachImage,
                                                                          output_size=(height,
                                                                                       width))

            #TODO: POST AND PRE ROTATION ARE DISABLED DUE TO PYTORCH ISSUE
            # ISSUE - https://github.com/pytorch/pytorch/issues/34704
            # WHEN FIXED RE ENABLE
            
            # POST ROTATION
            if random.random() < rotProb and False:
                #print("POST ROT ")
                theta = torch.distributions.uniform.Uniform(low = -180,
                                                            high = 180).sample().item()


                eachImage = torchvision.transforms.functional.pad(img = eachImage,
                                                                  padding = (int(height/2), int(width/2)),
                                                                  padding_mode = "reflect")
                eachImage = torchvision.transforms.functional.rotate(img = eachImage,
                                                                     angle = -theta)
                eachImage = torchvision.transforms.functional.center_crop(img=eachImage,
                                                                          output_size=(height,
                                                                                       width))

            # Fractional translation  - Done as int for now
            if random.random() < self.augmentProb:
                #print("FRACTIONAL TRANSLATE")
                tX, tY = torch.distributions.normal.Normal(loc = 0, scale = 0.125).sample((2,))
                tX = (tX * width).int().item()
                tY = (tY * height).int().item()
                if tX < 0:
                    padRight = -tX
                    padLeft = 0
                elif tX > 0:
                    padLeft = tX
                    padRight = 0
                else:
                    padLeft = 0
                    padRight = 0

                if tY < 0:
                    padTop = -tY
                    padBot = 0
                elif tY > 0:
                    padBot = tY
                    padTop = 0
                else:
                    padTop = 0
                    padBot = 0
                eachImage = torchvision.transforms.functional.pad(img=eachImage,
                                                    padding=(padLeft, padTop, padRight, padBot),
                                                    padding_mode="reflect")


                eachImage = torchvision.transforms.functional.crop(img = eachImage,
                                                                   left = padRight,
                                                                   top = padBot,
                                                                   height = height,
                                                                   width = width)


            # ### COLOUR TRANSFORMATIONS
            # if random.random() < self.augmentProb:
            #
            #
            #     #print("COLOR TRANSFORM")
            #     origMax = torch.max(eachImage)
            #     origMin = torch.min(eachImage)
            #     brightness = torch.distributions.log_normal.LogNormal(0, 0.2).sample().item()
            #     eachImage = torch.mul(eachImage, brightness)
            #     eachImage = torch.clamp(eachImage, min=origMin, max=origMax)
            #
            #
            #     # eachImage = torchvision.transforms.functional.adjust_brightness(eachImage,
            #     #                                                                 brightness)
            #
            # if random.random() < self.augmentProb:
            #     #print("INVERT")
            #     eachImage = torch.mul(eachImage, -1)
            #
            # if random.random() < self.augmentProb:
            #     #print("CONTRAST")
            #     contrast = torch.distributions.log_normal.LogNormal(0, (0.5*math.log(2))).sample().item()
            #     origMax = torch.max(eachImage)
            #     origMin = torch.min(eachImage)
            #     imageMean = torch.mean(eachImage, dim=(-1,-2,-3), keepdim=True)
            #     eachImage = torch.mul(eachImage, contrast)
            #     eachImage = torch.add(eachImage,(1-contrast) * imageMean)
            #     eachImage = torch.clamp(eachImage, min=origMin, max=origMax)


            newImages.append(eachImage.view(channelNo, height, width))





        imageBatch = torch.cat(newImages, dim = 0).view(batchSize, channelNo, height, width)

        imageBatch = self.applyColourTransformations(imageBatch)
        imageBatch = self.applyImageCorruptions(imageBatch)


        return imageBatch


    def applyImageCorruptions(self, imageBatch):
        batchSize, channelNo, height, width = imageBatch.size()
        newImages = []
        for eachImage in imageBatch:
        ### IMAGE SPACE CORRUPTIONS
            if random.random() < self.augmentProb:
                #print("CORRUPTION")
                standardDev = torch.distributions.half_normal.HalfNormal(scale=0.1).sample().item()
                eachImage = torch.add(eachImage, torch.normal(mean=0, std = standardDev,size=eachImage.size(), device = eachImage.device))

            if random.random() < self.augmentProb:
                #print("RAND ERASE")
                eachImage = self.randomErase(eachImage)

            newImages.append(eachImage.view(channelNo, height, width))

        imageBatch = torch.cat(newImages, dim = 0).view(batchSize, channelNo, height, width)
        return imageBatch



    def applyBrightnessTranslation(self, inputMatrix,transformationProbs):
        b = torch.distributions.normal.Normal(0,0.2).sample([len(transformationProbs)]) * transformationProbs

        totalTransMatrix = torch.stack([self.translate3D(tY=eachBVal,
                                                         tX = eachBVal,
                                                         tZ = eachBVal) for eachBVal in b])
        return torch.matmul(totalTransMatrix, inputMatrix)

    def applyContrastScaling(self, inputMatrix, transformationProbs):
        c = torch.distributions.log_normal.LogNormal(0,
                                                     0.5 * math.log(2)).sample([len(transformationProbs)]) * transformationProbs
        totalTransMatrix = torch.stack([self.scale3D(sY=eachCVal, sX = eachCVal,sZ = eachCVal) if eachCVal != 0 else self.scale3D(sY=1,sX=1, sZ=1)for eachCVal in c])
        return torch.matmul(totalTransMatrix, inputMatrix)

    def applyLumaFlip(self, inputMatrix,v, transformationProbs):
        i = torch.randint_like(low = 0, high = 2, input = transformationProbs) * transformationProbs
        identityMatrix = torch.eye(n=4).repeat(len(transformationProbs),1,1)
        transformationMatrix = 2 * torch.outer(v, v) * i.view(-1,1,1)
        transformationMatrix = identityMatrix - transformationMatrix
        return torch.matmul(transformationMatrix, inputMatrix)


    def applySaturationScaling(self, inputMatrix, transformationProbs, v):
        s = torch.distributions.log_normal.LogNormal(0,
                                                     1 * math.log(2)).sample([len(transformationProbs)]) * transformationProbs
        identityMatrix = torch.eye(n=4).repeat(len(transformationProbs),1,1)
        transformationMatrix = torch.outer(v,v) + (identityMatrix - torch.outer(v,v)) * s.view(-1,1,1)
        return torch.matmul(transformationMatrix, inputMatrix)



    def applyColourTransformations(self, imageBatch):
        batchSize,channelNumber, imageHeight, imageWidth = imageBatch.size()
        C = torch.eye(n=4).repeat(batchSize,1,1)

        probabilities = torch.rand(batchSize)
        transformationProbs = torch.where(probabilities < self.augmentProb,1,0)
        C = self.applyBrightnessTranslation(C,
                                            transformationProbs=transformationProbs)

        probabilities = torch.rand(batchSize)
        transformationProbs = torch.where(probabilities < self.augmentProb,1,0)
        C = self.applyContrastScaling(C,transformationProbs)
        #

        probabilities = torch.rand(batchSize)
        transformationProbs = torch.where(probabilities < self.augmentProb,1,0)
        v = torch.div(torch.FloatTensor([1,1,1,0]), sqrt(3))
        C = self.applyLumaFlip(C,v,transformationProbs)


        transformationProbs = torch.where(probabilities < self.augmentProb,1,0)
        v = torch.div(torch.FloatTensor([1,1,1,0]), sqrt(3))
        C = self.applySaturationScaling(C,transformationProbs,v)

        C = C.to(imageBatch)

        # FROM NVIDIA STYLEGAN 2 PYTORCH ADA
        imageBatch = imageBatch.reshape([batchSize, channelNumber, imageHeight* imageWidth])
        if channelNumber == 3:
            imageBatch = C[:, :3, :3] @ imageBatch + C[:, :3, 3:]
        elif channelNumber == 1:
            C = C[:, :3, :].mean(dim=1, keepdims=True)
            imageBatch = imageBatch * C[:, :, :3].sum(dim=2, keepdims=True) + C[:, :, 3:]
        else:
            raise ValueError('Image must be RGB (3 channels) or L (1 channel)')
        imageBatch = imageBatch.reshape([batchSize, channelNumber, imageHeight, imageWidth])

        return imageBatch



