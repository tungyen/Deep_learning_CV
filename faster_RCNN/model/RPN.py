import torch
from torch import nn, Tensor
from torch.nn import functional as F

class rpnHead(nn.Module):
    def __init__(self, inputC, anchorNum):
        super(rpnHead, self).__init__()
        self.sw = nn.Conv2d(inputC, inputC, kernel_size=3, stride=1, padding=1)
        self.cls = nn.Conv2d(inputC, anchorNum, kernel_size=1, stride=1)
        self.bboxPred = nn.Conv2d(inputC, anchorNum*4, kernel_size=1, stride=1)
        
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)
                
    def forward(self, x):
        clsLogits = []
        bboxRegression = []
        for _, featureMap in enumerate(x):
            feature = F.relu(self.sw(featureMap))
            clsLogits.append(self.cls(feature))
            bboxRegression.append(self.bboxPred(feature))
        return clsLogits, bboxRegression
    
class anchorGenerator(nn.Module):
    def __init__(self, sizes=(128, 256, 512), ratios=(0.5, 1.0, 2.0)):
        super(anchorGenerator, self).__init__()
        
        self.sizes = sizes
        self.ratios = ratios
        self.anchorCache = {}
        self.anchors = None
        
    def forward(self, imageList, featureMaps):
        # Input:
        #     imageList - The corresponding image to each feature map
        #     featureMaps - List[Tensor], Feature map from different level due to FPN
        featureMapSizes = list([featureMap.shape[-2:] for featureMap in featureMaps])
        imageSize = imageList.data.shape[-2:]
        
        dtype, device = featureMaps[0].dtype, featureMaps[0].device
        strides = [[torch.tensor(imageSize[0] // fSize[0], dtype=torch.int64, device=device),
                    torch.tensor(imageSize[1] // fSize[1], dtype=torch.int64, device=device)] for fSize in featureMapSizes]
        
        self.initializeAnchor(dtype, device)
        anchorsFeatureMaps = self.saveCachedAnchors(featureMapSizes, strides)
        anchors = []
        for i, (imgHeight, imgWidth) in enumerate(imageList.imageSizes):
            anchorImage = []
            for anchorsFeatureMap in anchorsFeatureMaps:
                anchorImage.append(anchorsFeatureMap)
            anchors.append(anchorImage)
            
        anchors = [torch.cat(anchorImage) for anchorImage in anchors]
        self.anchorCache.clear()
        return anchors
        
        
    def initializeAnchor(self, dtype, device):
        if self.anchors is not None:
            return
        anchors = [
            self.anchorGenerate(size, ratio, dtype, device)
            for size, ratio in zip(self.sizes, self.ratios)
        ]
        self.anchors = anchors
        
    def anchorGenerate(self, size, ratio, dtype, device):
        size = torch.as_tensor(size, dtype=dtype, device=device)
        ratio = torch.as_tensor(ratio, dtype=dtype, device=device)
        hRatio = torch.sqrt(ratio)
        wRatio = 1.0 / hRatio
        
        ws = (wRatio[:, None] * size[None, :]).view(-1)
        hs = (hRatio[:, None] * size[None, :]).view(-1)
        bases = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return bases.round()
    
    def saveCachedAnchors(self, featureMapSizes, strides):
        k = str(featureMapSizes) + str(strides)
        
        if k in self.anchorCache:
            return self.anchorCache[k]
        anchors = self.getAnchorPose(featureMapSizes, strides)
        self.anchorCache[k] = anchors
        return anchors
    
    def getAnchorPose(self, featureMapSizes, strides):
        anchors = []
        oriAnchors = self.anchors
        
        for size, stride, bases in zip(featureMapSizes, strides, oriAnchors):
            fHeight, fWidth = size
            sHeight, sWidth = stride
            device = bases.device
            
            shiftX = torch.arange(0, fWidth, dtype=torch.float32, device=device) * sWidth
            shiftY = torch.arange(0, fHeight, dtype=torch.float32, device=device) * sHeight
            
            shiftY, shiftX = torch.meshgrid(shiftY, shiftX)
            shiftX = shiftX.reshape(-1)
            shiftY = shiftY.reshape(-1)
            shifts = torch.stack([shiftX, shiftY, shiftX, shiftY], dim=1)
            shiftedAnchors = shifts.view(-1, 1, 4) + bases.view(1, -1, 4)
            anchors.append(shiftedAnchors(-1, 4))
        return anchors
    
class RPN(nn.Module):
    def __init__(self, anchor_generator, head, 
                 fgIouThres, bgIouThres, 
                 batchSize, posFrac, nmsThres, preNmsTopN, poseNmsTopN):
        self.anchor_generator = anchor_generator
        self.head = self.head
        
        self.boxCoder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.boxSimilarity = box_ops.box_iou
        self.proposalMatcher = det_utils.Matcher(
            fgIouThres, bgIouThres, allowLowQualityMatches=True
        )
        self.fgBgSampler = det_utils.balancedPosNegSampler(
            batchSize, posFrac
        )
        
        self.preNmsTopN = preNmsTopN
        self.postNmsTopN = poseNmsTopN
        self.nmsThres = nmsThres
        self.minSize = 1.0
        
    def forward(self, imgs, feats, annotation=None):
        # Inputs:
        #     imgs - ImageList object
        #     feats - dictionary object that records all feature maps
        #     annotation - ground truth of each image data
        feats = list(feats.values())
        clsLogits, bboxPreds = self.head(feats)
        anchors = self.anchor_generator(imgs, feats)
        imgNum = len(anchors)
        
        anchorNumFeats = [c.shape[1]*c.shape[2]*c.shape[3] for c in clsLogits]
        bboxCls, bboxRegression = concatBoxInfo(clsLogits, bboxPreds)
        
        proposals = self.boxCoder.decode(bboxRegression.detach(), anchors)
        proposals = proposals.view(imgNum, -1, 4)
        
def processing(target, B, anchorNum, classNum, h, w):
    target = target.view(B, -1, classNum, h, w)
    target = target.permute(0, 3, 4, 1, 2)
    target = target.reshape(B, -1, classNum)
    return target
    
def concatBoxInfo(clsLogits, bboxPreds):
    clsLogitsFlatten = []
    bboxPredFlatten = []
    
    for clsLogit, bboxPred in zip(clsLogits, bboxPreds):
        B, scoreNum, h, w = clsLogit.shape
        anchorParamNum = bboxPred.shape[1]
        anchorNum = anchorParamNum // 4
        classNum = scoreNum // anchorNum
        
        clsLogit = processing(clsLogit, B, anchorNum, classNum, h, w)
        clsLogitsFlatten.append(clsLogit)
        bboxPred = processing(bboxPred, B, anchorNum, 4, h, w)
        bboxPredFlatten.append(bboxPred)
    bboxCls = torch.cat(clsLogitsFlatten, dim=1).flatten(0, -2)
    bboxRegression = torch.cat(bboxPredFlatten, dim=1).reshape(-1, 4)
    return bboxCls, bboxRegression
        
        
        
        
        
            
            
            
        
        
        