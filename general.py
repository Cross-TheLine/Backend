import torch
import time
import numpy as np
import torch.nn as nn
import cv2
from scipy.spatial import distance

def train(model, train_loader, optimizer, device, epoch, max_iters=200):
    start_time = time.time()
    losses = []
    criterion = nn.CrossEntropyLoss()
    for iter_id, batch in enumerate(train_loader):
        optimizer.zero_grad()
        model.train()
        out = model(batch[0].float().to(device))
        gt = torch.tensor(batch[1], dtype=torch.long, device=device)
        loss = criterion(out, gt)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        end_time = time.time()
        duration = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
        print('train | epoch = {}, iter = [{}|{}], loss = {}, time = {}'.format(epoch, iter_id, max_iters,
                                                                                round(loss.item(), 6), duration))
        losses.append(loss.item())
        
        if iter_id > max_iters - 1:
            break
        
    return np.mean(losses)

def validate(model, val_loader, device, epoch, min_dist=5):
    losses = []
    tp = [0, 0, 0, 0]
    fp = [0, 0, 0, 0]
    tn = [0, 0, 0, 0]
    fn = [0, 0, 0, 0]
    criterion = nn.CrossEntropyLoss()
    model.eval()
    for iter_id, batch in enumerate(val_loader):
        with torch.no_grad():
            out = model(batch[0].float().to(device))
            gt = torch.tensor(batch[1], dtype=torch.long, device=device)
            loss = criterion(out, gt)
            losses.append(loss.item())
            # metrics
            output = out.argmax(dim=1).detach().cpu().numpy()
            for i in range(len(output)):
                x_pred, y_pred = postprocess(output[i])
                x_gt = batch[2][i]
                y_gt = batch[3][i]
                vis = batch[4][i]
                if x_pred:
                    if vis != 0:
                        dst = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                        if dst < min_dist:
                            tp[vis] += 1
                        else:
                            fp[vis] += 1
                    else:        
                        fp[vis] += 1
                if not x_pred:
                    if vis != 0:
                        fn[vis] += 1
                    else:
                        tn[vis] += 1
            print('val | epoch = {}, iter = [{}|{}], loss = {}, tp = {}, tn = {}, fp = {}, fn = {} '.format(epoch,
                                                                                                            iter_id,
                                                                                                            len(val_loader),
                                                                                                            round(np.mean(losses), 6),
                                                                                                            sum(tp),
                                                                                                            sum(tn),
                                                                                                            sum(fp),
                                                                                                            sum(fn)))
    eps = 1e-15
    precision = sum(tp) / (sum(tp) + sum(fp) + eps)
    vc1 = tp[1] + fp[1] + tn[1] + fn[1]
    vc2 = tp[2] + fp[2] + tn[2] + fn[2]
    vc3 = tp[3] + fp[3] + tn[3] + fn[3]
    recall = sum(tp) / (vc1 + vc2 + vc3 + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    print('precision = {}'.format(precision))
    print('recall = {}'.format(recall))
    print('f1 = {}'.format(f1))

    return np.mean(losses), precision, recall, f1


def postprocess(feature_map, scale=2, peak_threshold=80, blur_size=5, centroid_radius=2):
    feature_map = feature_map.reshape((360, 640)).astype(np.float32)
    heatmap = cv2.GaussianBlur(feature_map, (blur_size, blur_size), 0)
    max_value = float(heatmap.max())
    if max_value < peak_threshold:
        return None, None

    y_idx, x_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)

    x_min = max(0, x_idx - centroid_radius)
    x_max = min(heatmap.shape[1], x_idx + centroid_radius + 1)
    y_min = max(0, y_idx - centroid_radius)
    y_max = min(heatmap.shape[0], y_idx + centroid_radius + 1)
    patch = heatmap[y_min:y_max, x_min:x_max]

    if patch.sum() > 0:
        ys, xs = np.mgrid[y_min:y_max, x_min:x_max]
        x_idx = float((patch * xs).sum() / patch.sum())
        y_idx = float((patch * ys).sum() / patch.sum())

    return x_idx * scale, y_idx * scale



