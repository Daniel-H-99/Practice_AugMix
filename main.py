import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms

from WideResNet_pytorch.wideresnet import WideResNet

from augmentations import *
import time
import matplotlib.pyplot as plt

CK_PATH = "./ckpt/"
RES_PATH = "./res/"
CORRUPTIONS = [
    'clear',
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

def top_n_count(pred, target, n=5, normalized=True):
    questions, options = pred.shape
    choices = np.argsort(-pred, axis=1)
    cnt = 0
    for i in range(questions):
        if target[i] in choices[i,0:n]:
            cnt += 1
    if normalized:
        return cnt / questions
    else:
        return cnt

def JSLoss(p1, p2, p3):
    kldiv = nn.KLDivLoss(reduction='batchmean')
    m = (p1 + p2 + p3) / 3
    log_m = m.log()
    res = (kldiv(log_m, p1) + kldiv(log_m, p2) + kldiv(log_m, p3)) / 3
    return res

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, weight=1):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

class AugMix:
    def __init__(self, k=3, alpha = 1, js_on=True, output_size=(3, 32, 32), pre_aug=transforms.RandomCrop(32), pre_process=transforms.ToTensor()):
        self.k = 3
        self.alpha = 1
        self.js_on = js_on
        self.augmentations = augmentations
        self.output_size = output_size
        self.pre_aug=pre_aug
        self.pre_process=pre_process
        self.draw_dirichlet = lambda : np.random.dirichlet(tuple(self.alpha for _ in range(self.k)))
        self.draw_beta = lambda : np.random.beta(self.alpha, self.alpha)
        self.draw_chainlength = lambda : np.random.randint(1,3)
        self.draw_operation = lambda : np.random.choice(self.augmentations, 3)
        
    def __call__(self, pil_img):
        ori = self.pre_aug(pil_img)
        processed_ori = self.pre_process(ori)
        cnt = 2 if self.js_on else 1
        results = []
        for _ in range(cnt):
            res = torch.zeros(self.output_size)
            w = self.draw_dirichlet()
            for i in range(self.k):
                ops = self.draw_operation()
                chainlength = self.draw_chainlength()
                ops = ops[0:chainlength]
                src = ori
                for op in ops:
                    src = op(src)
                src = self.pre_process(src)
                res += w[i] * src
            m = self.draw_beta()
            res = m * processed_ori + (1 - m) * res
            results.append(res)
        if self.js_on:
            results.append(processed_ori)
        return torch.stack(results).squeeze(dim=0)

def main(augmix_on=True, js_on=True, train_on=True, eval_on=True, use_cifar10=False):
    TAG = ("AugMix" if augmix_on else "Standard") + ("_JSLoss" if augmix_on and js_on else "") 
    torch.manual_seed(2020)
    np.random.seed(2020)
    epochs = 100
    batch_size = 256
    if js_on:
        LAMBDA = 12
    if use_cifar10:
        TESTSET = "CIFAR-10"
    else:
        TESTSET = "CIFAR-100"
    print("[{}_{}] starts".format(TAG, TESTSET))
    # 1. dataload
    # basic augmentation & preprocessing
    train_base_aug = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4)
    ]
    preprocess = [
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)
    ]
    
    # apply augmix at transformation stage
    train_transform = AugMix(js_on=js_on, pre_aug=transforms.Compose(train_base_aug), 
                             pre_process=transforms.Compose(preprocess)) if augmix_on else transforms.Compose(train_base_aug + preprocess)
    test_transform = transforms.Compose(preprocess)
    
    # load data
    if use_cifar10:
        train_data = datasets.CIFAR10('./data/cifar', train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10('./data/cifar', train=False, transform=test_transform, download=True)
    else: 
        train_data = datasets.CIFAR100('./data/cifar', train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100('./data/cifar', train=False, transform=test_transform, download=True)
    
    train_loader = torch.utils.data.DataLoader(
                   train_data,
                   batch_size=batch_size,
                   shuffle=True,
                   num_workers=4,
                   pin_memory=True)
                   
    # 2. model
    # wideresnet 40-2
    model = WideResNet(depth=40, num_classes=100, widen_factor=2, drop_rate=0.0)

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.SGD(
                  model.parameters(),
                  0.1,
                  momentum=0.9,
                  weight_decay=0.0005,
                  nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(train_loader), eta_min=1e-6, last_epoch=-1)

    model = nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    
    # 4. training
    if train_on:
        model.train()
        softmax_train = nn.Softmax(dim=1)
        losses = []
        duration = AverageMeter()
        for epoch in range(epochs):
            elapsed = time.time()
            for i, (images, targets) in enumerate(train_loader):
                images, targets = images.cuda(), targets.cuda()
                optimizer.zero_grad()
                if augmix_on and js_on:
                    # Jensen-Shannon Divergence Loss
                    d = images.shape
                    assert(d[1]==3)
                    images_flatten = images.view(3 * d[0], d[2], d[3], d[4])
                    preds = softmax_train(model(images_flatten))
                    p1 = preds[::3]
                    p2 = preds[1::3]
                    p_ori = preds[2::3]
                    js = JSLoss(p1, p2, p_ori)
                    loss = F.nll_loss(p_ori.log(), targets) + LAMBDA * js
                else:
                    logits = model(images)
                    loss = F.cross_entropy(logits, targets)

                loss.backward()
                optimizer.step()
                scheduler.step()

                losses.append(loss.item())
                if i % 100 == 0 or i+1 == len(train_loader):
                    print("Train Loss: {:.4f}".format(loss.item()))
                    if augmix_on and js_on:
                        print("JSLoss: {:.4f}".format(js))
            
            elapsed = time.time() - elapsed
            duration.update(elapsed)
            torch.save({
                "epoch": epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses
            }, CK_PATH + TAG + "/{}/epoch_{}.pt".format(TESTSET, epoch + 1))
        # report duration taken for an epoch
        with open('{}_{}_duration.txt'.format(RES_PATH + TAG, TESTSET),'w') as f:
            f.write("Last Value: {}\nAverage: {}".format(duration.val, duration.avg))
            
    # 4. evaluation
    if eval_on:
        chpk = torch.load(CK_PATH + TAG + "/{}/epoch_100.pt".format(TESTSET))
        model.load_state_dict(chpk['model_state_dict'])
        model.eval()
        softmax_eval = nn.Softmax(dim=1)
        errors = []
        with torch.no_grad():
            # evaluate on cirfar100, cifar100-c
            for corruption in CORRUPTIONS:
                print("CORRUPTIONS: {}".format(corruption))
                if corruption != 'clear':
                    # use corrupted testset
                    test_data.data = np.load('./data/cifar/{}-C/{}.npy'.format(TESTSET, corruption))
                    test_data.targets = np.load('./data/cifar/{}-C/labels.npy'.format(TESTSET))
                test_loader = torch.utils.data.DataLoader(
                               test_data,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=4,
                               pin_memory=True)
                corrects = 0
                for _, (inputs, targets) in enumerate(test_loader):
                    logits = model(inputs)
                    preds =  softmax_eval(logits).detach().cpu().numpy()
                    corrects += top_n_count(preds, targets.numpy(), n=1, normalized=False)
                error = 1 - corrects / len(test_data)
                print("[{}] \"{}-{}\" error rate: {:.6}".format(TAG, TESTSET, corruption, error))
                errors.append((corruption, error))
            # report errors for each testset
            np.savetxt('{}_{}.csv'.format(RES_PATH + TAG, TESTSET), np.array(errors), delimiter=',', fmt="%s")

if __name__=="__main__":
    # cifar-100
    main(augmix_on=True, js_on=True)
    main(augmix_on=True, js_on=False)
    main(augmix_on=False, js_on=False)
    # cifar-10
    main(augmix_on=True, js_on=True, use_cifar10=True)
    main(augmix_on=True, js_on=False, use_cifar10=True)
    main(augmix_on=False, js_on=False, use_cifar10=True)