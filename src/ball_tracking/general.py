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


def postprocess_with_confidence(feature_map, scale=2, peak_threshold=80, blur_size=5,
                                centroid_radius=8, relative_threshold=0.6):
    candidates = postprocess_candidates(
        feature_map,
        scale=scale,
        peak_threshold=peak_threshold,
        blur_size=blur_size,
        centroid_radius=centroid_radius,
        relative_threshold=relative_threshold,
        max_candidates=1,
    )
    if not candidates:
        feature_map = feature_map.reshape((360, 640)).astype(np.float32)
        heatmap = cv2.GaussianBlur(feature_map, (blur_size, blur_size), 0)
        return None, None, float(heatmap.max())
    x_idx, y_idx, score = candidates[0]
    return x_idx, y_idx, score


def _weighted_component_center(heatmap, x_idx, y_idx, peak_threshold,
                               centroid_radius, relative_threshold, scale):
    x_min = max(0, x_idx - centroid_radius)
    x_max = min(heatmap.shape[1], x_idx + centroid_radius + 1)
    y_min = max(0, y_idx - centroid_radius)
    y_max = min(heatmap.shape[0], y_idx + centroid_radius + 1)
    patch = heatmap[y_min:y_max, x_min:x_max]
    max_value = float(patch.max())

    threshold = max(peak_threshold, max_value * relative_threshold)
    mask = patch >= threshold
    if np.any(mask):
        peak_y = y_idx - y_min
        peak_x = x_idx - x_min
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        peak_label = labels[peak_y, peak_x]
        mask = labels == peak_label if peak_label > 0 else mask
        weights = np.where(mask, patch, 0)
    else:
        weights = patch

    if weights.sum() > 0:
        ys, xs = np.mgrid[y_min:y_max, x_min:x_max]
        x_idx = float((weights * xs).sum() / weights.sum())
        y_idx = float((weights * ys).sum() / weights.sum())

    return x_idx * scale, y_idx * scale, max_value


def postprocess_candidates(feature_map, scale=2, peak_threshold=80, blur_size=5,
                           centroid_radius=8, relative_threshold=0.6,
                           max_candidates=5):
    feature_map = feature_map.reshape((360, 640)).astype(np.float32)
    heatmap = cv2.GaussianBlur(feature_map, (blur_size, blur_size), 0)
    if float(heatmap.max()) < peak_threshold:
        return []

    mask = heatmap >= peak_threshold
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    candidates = []
    for label in range(1, num_labels):
        x, y, width, height, area = stats[label]
        if area <= 0:
            continue

        component = labels[y:y + height, x:x + width] == label
        component_values = np.where(component, heatmap[y:y + height, x:x + width], 0)
        local_y, local_x = np.unravel_index(np.argmax(component_values), component_values.shape)
        peak_x = int(x + local_x)
        peak_y = int(y + local_y)
        candidates.append(
            _weighted_component_center(
                heatmap,
                peak_x,
                peak_y,
                peak_threshold,
                centroid_radius,
                relative_threshold,
                scale,
            )
        )

    candidates.sort(key=lambda item: item[2], reverse=True)
    return candidates[:max_candidates]


def postprocess(feature_map, scale=2, peak_threshold=80, blur_size=5, centroid_radius=2):
    x_idx, y_idx, _ = postprocess_with_confidence(
        feature_map,
        scale=scale,
        peak_threshold=peak_threshold,
        blur_size=blur_size,
        centroid_radius=centroid_radius,
    )
    return x_idx, y_idx



