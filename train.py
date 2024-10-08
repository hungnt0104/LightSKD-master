import os
import time
import gdown
import pickle
import numpy as np
import torch
torch.set_printoptions(threshold=np.inf)
from torch.nn import DataParallel
from torch import nn, optim
from datasets import get_trainloader as tl, get_testloader as tel
from backbone.adapter import adapter_2
from backbone.mobilenetv2 import *
import torch.nn.functional as F
import argparse
from utils import output_process,get_model


parser = argparse.ArgumentParser(description="DRG & DSR")
parser.add_argument("--epoch", default=200, type=int)
parser.add_argument("--lr_drop_percent", default=0.2, type=float)
parser.add_argument("--lr_drop_epoch", default=[60,120,160])
parser.add_argument("--visible_device_single", default=0, type=int, help="if use_parallel=True, this item will not work.")
parser.add_argument("--visible_device_list", default=[0], type=list, help="if use_parallel=False, this item will not work.")
parser.add_argument("--use_parallel", default=False)
parser.add_argument("--datasets", default='FER2013', type=str)
parser.add_argument("--num_classes", default=7, type=int)
parser.add_argument("--temperature", default=4.0, type=float)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--alpha", default=0.2, help="RG-Loss")
parser.add_argument("--beta", default=1.2, help="SW-Loss")
parser.add_argument("--large_trans", default=True)
parser.add_argument("--model", default='ResNet50')
parser.add_argument("--model_path", default="./save/")
parser.add_argument("--initial_lr", default=0.001, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--DRG", default=False)
parser.add_argument("--DSR", default=True)
parser.add_argument("--pretrained", default=True)
args = parser.parse_args()

def download_pretrained_weights():
    # Google Drive file ID
    drive_url = "https://drive.google.com/uc?id=1_ig22lgZCMpHdlP2B18RbIAhtGVI6yen"
    
    # Define the output file path where the model will be saved
    output = "./pretrained/resnet50_scratch_weight.pkl"
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output), exist_ok=True)

    # Check if the file already exists to avoid redundant downloads
    if not os.path.exists(output):
        # Download the file from Google Drive
        gdown.download(drive_url, output, quiet=False)
        print(f"Pretrained weights downloaded to {output}")
    else:
        print(f"File already exists at {output}")


if args.pretrained:
    download_pretrained_weights()
    pretrained_weights_path = "./pretrained/resnet50_scratch_weight.pkl"
    print(f"Loading pretrained weights from {pretrained_weights_path}")

    # Load the state dict using pickle
    with open(pretrained_weights_path, 'rb') as f:
        state_dict = pickle.load(f)
    # Convert numpy arrays to torch tensors in state_dict
    for key in state_dict.keys():
        if isinstance(state_dict[key], np.ndarray):
            state_dict[key] = torch.from_numpy(state_dict[key])

    net = get_model(args)
    
    # Load the state dict into the model
    net.load_state_dict(state_dict)
    net.fc = nn.Linear(net.fc.in_features, 10)

else:
    net = get_model(args)


print(args)

cur_time = time.time()
trainloader = tl(params=args.datasets,isDense=args.large_trans,bs=args.batch_size)
testloader = tel(params=args.datasets,isDense=args.large_trans,bs=args.batch_size)
# svloader = svl()

device = torch.device("cuda:"+str(args.visible_device_single) if torch.cuda.is_available() else "cpu")

net = get_model(args)
ada_net = adapter_2(num_classes=args.num_classes)
if args.use_parallel:
    net = DataParallel(net,device_ids=args.visible_device_list)
    ada_net = DataParallel(ada_net,device_ids=args.visible_device_list)
else:
    net.to(device)
    ada_net = ada_net.to(device)

criterion = nn.CrossEntropyLoss()
if args.datasets == "FER2013":
    optimizer = optim.AdamW([{'params': net.parameters()}, {'params': ada_net.parameters()}],
                        lr=0.001, weight_decay=args.weight_decay)
else:
    optimizer = optim.SGD([{'params': net.parameters()}, {'params': ada_net.parameters()}],
                        lr=args.initial_lr, momentum=args.momentum, weight_decay=args.weight_decay)


load_dir = args.model_path
load_name = load_dir+str(args.model)+'.pth'
if not os.path.isdir(load_dir):
    os.makedirs(load_dir)
if os.path.isfile(load_name):
    checkpoint = torch.load(load_name)
    net.load_state_dict(checkpoint['net'])
    ada_net.load_state_dict(checkpoint['net_d'])

bs = args.batch_size

def train(epoch):
    global init, trainloader, testloader, bs
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_i_loss = 0
    train_r_loss = 0 
    last_output_processed= torch.rand((bs, args.num_classes))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # learn labels
        outputs,mids = net(inputs)
        outputs_d = ada_net(mids[1])
        
        # train the model
        loss = torch.mean(criterion(outputs, targets))
        # +torch.mean(criterion(outputs_d, targets))
        loss += 100
        T=4
        # DRG
        if args.DRG:
            ist = args.alpha*nn.KLDivLoss(reduction="batchmean")(F.log_softmax(outputs/T,dim=1),
                                                       F.softmax((outputs_d.detach())/T,dim=1))* (T * T)
            loss += ist
            train_i_loss += ist.item()

        if args.DSR:
            output_processed = output_process(outputs)
            if batch_idx  !=0:
                # DSR
                adv_loss = args.beta*nn.KLDivLoss(reduction="batchmean")(F.log_softmax(output_processed/T,dim=1),
                                                     F.softmax((last_output_processed.detach())/T,dim=1))* (T * T)
                loss += adv_loss
                train_r_loss += adv_loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).sum().float().cpu()
        train_loss += loss.item()
        if args.DSR:
            last_output_processed = output_processed

        if (batch_idx + 1) % 40 == 0:
            print(batch_idx + 1, len(trainloader),
                    'Loss: %.3f ---------- Accuracy: %.3f%% (%d/%d) ------- istr: %.6f ------- sort: %.6f'
                  % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total,
                    train_i_loss / (batch_idx + 1),train_r_loss / (batch_idx + 1)))

    filename = args.model_path+str(args.model)+'.pth'
    state = None
    if args.use_parallel:
        state = {'net': net.module.state_dict(), 'net_d': ada_net.module.state_dict()}
    else:
        state = {'net':net.state_dict(),'net_d':ada_net.state_dict()}
    torch.save(state, filename)

def full_test():
    print("*****************************************TESTING*****************************************")
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs,_ = net(inputs)
            correct += outputs.max(1)[1].eq(targets).sum()
            test_loss += torch.mean(criterion(outputs, targets))
            total += outputs.size(0)

        print(len(testloader),
              'Total Loss: %.3f | Average Accuracy: %.3f%% (%d/%d)'
              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        #f.close()
        #f.write(str(correct/total*100.)+'\n')

# def single_test():
#     net.eval()
#     f = open("./logs/single-logs.txt", "w")
#     with torch.no_grad():
#         for batch_idx, (inputs, _) in enumerate(svloader):
#             inputs = inputs.to(device)
#             outputs,_ = net(inputs)
#             f.write(str(outputs[0]))
#             f.write("\n")
#         print("single test finished.")
#         f.close()

if __name__ == "__main__":
    for epoch in range(args.epoch):
        if epoch in args.lr_drop_epoch:
            for params in optimizer.param_groups:
                params["lr"] *= args.lr_drop_percent
        train(epoch)
        full_test()
        #single_test()
        #break

    print("FINISHED! Total time expenditure: {}".format(str(time.time()-cur_time)))







