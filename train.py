import csv
import datetime
import os
import shutil
import time

import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from config import opt
from object_place_dataset import get_test_dataloader, get_train_dataloader
from object_place_net import ObjectPlaceNet

global test_results, best_acc, best_f1
test_results = []
best_acc = 0
best_f1 = 0

## F1-score and balanced accuracy
def F1(preds, gts):
    tp = sum(list(map(lambda a, b: a == 1 and b == 1, preds, gts)))
    fp = sum(list(map(lambda a, b: a == 1 and b == 0, preds, gts)))
    fn = sum(list(map(lambda a, b: a == 0 and b == 1, preds, gts)))
    tn = sum(list(map(lambda a, b: a == 0 and b == 0, preds, gts)))
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    f1 = (2 * tp) / (2 * tp + fp + fn)
    bal_acc = (tpr + tnr) / 2
    return f1, bal_acc


def train(train_loader, net, criterion, optimizer, epoch, device, writer):
    start = time.time()
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    print("\n===  Epoch: [{}/{}]  === ".format(epoch + 1, opt.epochs))
    fetchdata_time = time.time()
    forward_time = time.time()
    batch_time = time.time()

    for batch_index, (img_cat, label, target_box) in enumerate(train_loader):
        fetchdata_time = time.time() - fetchdata_time
        img_cat, label, target_box = img_cat.to(device), label.to(device), target_box.to(device)

        forward_time = time.time()
        logits = net(img_cat)
        forward_time = time.time() - forward_time

        loss = criterion(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, pre_label = logits.max(1)
        total += label.size(0)
        correct += pre_label.eq(label).sum().item()
        iteration = epoch * len(train_loader) + batch_index

        if (batch_index + 1) % opt.display_freq == 0:
            cur_lr = optimizer.param_groups[0]['lr']
            avg_acc = correct / total
            avg_loss = train_loss / (batch_index + 1)
            writer.add_scalar('Train/loss', avg_loss, iteration)
            writer.add_scalar('Train/accuracy',  avg_acc,  iteration)
            writer.add_scalar('Train/learning_rate', cur_lr, iteration)

            time_per_batch = (time.time() - start) / (batch_index + 1.)
            last_batches = (opt.epochs - epoch - 1) * \
                len(train_loader) + (len(train_loader) - batch_index - 1)
            last_time = int(last_batches * time_per_batch)
            time_str = str(datetime.timedelta(seconds=last_time))

            print(
                "===  step: [{:3}/{}], loss: {:.3f} | acc: {:6.3f} | lr: {:.6f} | estimated last time: {} ===".format(
                    batch_index + 1, len(train_loader), avg_loss, avg_acc, cur_lr, time_str))

        batch_time = time.time() - batch_time

        batch_time = time.time()
        fetchdata_time = time.time()


def test(test_loader, net, criterion, optimizer, epoch, device, writer):
    global best_acc, best_f1
    net.eval()
    test_loss = 0
    total = 0
    preds = []
    gts = []

    print("===  Validate [{}/{}] ===".format(epoch + 1, opt.epochs))
    with torch.no_grad():
        for batch_index, (img_cat, label, target_box) in enumerate(tqdm(test_loader)):
            img_cat, label, target_box = img_cat.to(device), label.to(device), target_box.to(device)

            logits = net(img_cat)

            preds.extend(logits.max(1)[1].cpu().numpy())
            gts.extend(label.cpu().numpy())
            total += label.size(0)

    f1, bal_acc = F1(preds, gts)
    print("Test on {} images, local:f1={:.3f},bal_acc={:.3f}".format(
        total, f1, bal_acc))
    writer.add_scalar('Test/F1', f1, epoch)
    writer.add_scalar('Test/balanced_accuracy', bal_acc, epoch)

    if bal_acc > best_acc:
        best_acc = bal_acc
        checkpoint_path = os.path.join(opt.checkpoint_dir, 'best-acc.pth')
        torch.save(net.state_dict(), checkpoint_path)
        print('Update best accuracy checkpoint, best_acc={:.3f}'.format(best_acc))

    if f1 > best_f1:
        best_f1 = f1
        checkpoint_path = os.path.join(opt.checkpoint_dir, 'best-f1.pth')
        torch.save(net.state_dict(), checkpoint_path)

    if epoch % opt.save_freq == 0:
        checkpoint_path = os.path.join(
            opt.checkpoint_dir, f'model-{epoch}.pth')
        torch.save(net.state_dict(), checkpoint_path)
        print('Update best F1 checkpoint, best_F1={:.3f}'.format(best_f1))

    writer.add_scalar('Test/best_acc', best_acc, epoch)
    writer.add_scalar('Test/best_f1', best_f1, epoch)
    global test_results
    test_results.append([epoch, f1, bal_acc])


def write_test_results():
    global test_results
    csv_path = os.path.join(opt.exp_path, '..', '{}.csv'.format(opt.exp_name))
    header = ['epoch', 'F1', 'balanced_accuracy']
    epoches = list(range(len(test_results)))
    rows = [header] + test_results
    metrics = [[] for i in header]
    for result in test_results:
        for i, r in enumerate(result):
            metrics[i].append(r)
    for name, m in zip(header, metrics):
        if name == 'epoch':
            continue
        index = m.index(max(m))
        title = 'best {}(epoch-{})'.format(name, index)
        row = [l[index] for l in metrics]
        row[0] = title
        rows.append(row)
    with open(csv_path, 'w') as f:
        cw = csv.writer(f)
        cw.writerows(rows)
    print('Save result to ', csv_path)


if __name__ == "__main__":
    device = torch.device("cuda:{}".format(opt.gpu_id))
    opt.create_path()
    print('Experiment name {} \n'.format(os.path.basename(opt.exp_path)))
    for file in ['config.py', 'object_place_dataset.py', 'object_place_net.py', 'train.py']:
        shutil.copy(file, opt.exp_path)
        print('backup ', file)

    net = ObjectPlaceNet().to(device)
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(net.parameters(), opt.base_lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=opt.lr_milestones, gamma=opt.lr_gamma)

    train_loader = get_train_dataloader()

    print(("=======  Training  ======="))
    writer = SummaryWriter(log_dir=opt.log_dir)
    for epoch in range(opt.epochs):
        train(train_loader, net, criterion, optimizer, epoch, device, writer)
        if epoch == 0 or (epoch + 1) % opt.eval_freq == 0 or epoch == opt.epochs - 1:
            test_loader = get_test_dataloader()
            test(test_loader, net, criterion, optimizer, epoch, device, writer)
            write_test_results()
        lr_scheduler.step()
    print(("=======  Training Finished.Best F1={:.3f}, best balanced accuracy={:.1%}========".format(
        best_f1, best_acc)))
