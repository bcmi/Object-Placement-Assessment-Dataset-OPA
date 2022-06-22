import os

import torch
from tqdm import tqdm

from config import opt
from object_place_dataset import get_test_dataloader
from object_place_net import ObjectPlaceNet


def F1(preds, gts):
    tp = sum(list(map(lambda a, b: a == 1 and b == 1, preds, gts)))
    fp = sum(list(map(lambda a, b: a == 1 and b == 0, preds, gts)))
    fn = sum(list(map(lambda a, b: a == 0 and b == 1, preds, gts)))
    tn = sum(list(map(lambda a, b: a == 0 and b == 0, preds, gts)))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    bal_acc = (tpr + tnr) / 2
    return f1, bal_acc


def evaluate_model(device, checkpoint_path='./best-acc.pth'):
    opt.without_mask = False
    assert os.path.exists(checkpoint_path), checkpoint_path
    net = ObjectPlaceNet(backbone_pretrained=False)
    print('load pretrained weights from ', checkpoint_path)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net = net.to(device).eval()

    total = 0
    pred_labels = []
    gts = []

    test_loader = get_test_dataloader()

    with torch.no_grad():
        for batch_index, (img_cat, label, target_box) in enumerate(tqdm(test_loader)):
            img_cat, label, target_box = img_cat.to(
                device), label.to(device), target_box.to(device)

            logits = net(img_cat)

            pred_labels.extend(logits.max(1)[1].cpu().numpy())
            gts.extend(label.cpu().numpy())
            total += label.size(0)

    total_f1, total_bal_acc = F1(pred_labels, gts)
    print("Baseline model evaluate on {} images, local:f1={:.4f},bal_acc={:.4f}".format(
        total, total_f1, total_bal_acc))

    return total_f1, total_bal_acc


if __name__ == '__main__':
    device = "cuda:0"
    f1, balanced_acc = evaluate_model(device, checkpoint_path='./best-acc.pth')
