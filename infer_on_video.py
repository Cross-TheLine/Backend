from model import BallTrackerNet
import torch
import cv2
from general import postprocess_candidates
from tqdm import tqdm
import numpy as np
import argparse
from itertools import groupby
from scipy.spatial import distance
from scipy.signal import savgol_filter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
TRACK_WIDTH = 1280
TRACK_HEIGHT = 720
MIN_TRACK_CONFIDENCE = 80


def estimate_next_position(track, max_history_gap=4):
    valid = []
    for idx in range(len(track) - 1, -1, -1):
        x, y = track[idx]
        if x is not None and y is not None:
            valid.append((idx, float(x), float(y)))
            if len(valid) == 2:
                break
    if len(valid) < 2:
        return None

    last_idx, last_x, last_y = valid[0]
    prev_idx, prev_x, prev_y = valid[1]
    last_gap = len(track) - last_idx
    prev_gap = last_idx - prev_idx
    if last_gap > max_history_gap or prev_gap <= 0 or prev_gap > max_history_gap:
        return None

    vx = (last_x - prev_x) / prev_gap
    vy = (last_y - prev_y) / prev_gap
    return last_x + vx * last_gap, last_y + vy * last_gap, last_gap


def estimate_recent_motion(track, max_history_gap=4):
    valid = []
    for idx in range(len(track) - 1, -1, -1):
        x, y = track[idx]
        if x is not None and y is not None:
            valid.append((idx, float(x), float(y)))
            if len(valid) == 2:
                break
    if len(valid) < 2:
        return None

    last_idx, last_x, last_y = valid[0]
    prev_idx, prev_x, prev_y = valid[1]
    gap = last_idx - prev_idx
    if gap <= 0 or gap > max_history_gap:
        return None

    vx = (last_x - prev_x) / gap
    vy = (last_y - prev_y) / gap
    speed = float(np.linalg.norm(np.array([vx, vy])))
    return vx, vy, speed, gap


def choose_temporal_candidate(candidates, track, max_candidate_dist=260,
                              score_distance_tradeoff=0.35):
    if not candidates:
        return None, None, 0.0

    prediction = estimate_next_position(track)
    if prediction is None:
        return candidates[0]

    pred_x, pred_y, gap = prediction
    gate = max_candidate_dist * max(1, gap)
    best_score = max(candidate[2] for candidate in candidates)
    ranked = []
    for candidate in candidates:
        x, y, score = candidate
        dist = float(np.linalg.norm(np.array([x - pred_x, y - pred_y])))
        if dist <= gate:
            score_penalty = max(best_score - score, 0.0) * score_distance_tradeoff
            ranked.append((dist + score_penalty, candidate))

    if not ranked:
        return candidates[0]
    ranked.sort(key=lambda item: item[0])
    return ranked[0][1]


def refine_visual_center(frame, x, y, roi_radius=30, min_area=3, max_area=1100,
                         prev_frame=None):
    if x is None or y is None:
        return x, y

    height, width = frame.shape[:2]
    frame_x = float(x) * width / TRACK_WIDTH
    frame_y = float(y) * height / TRACK_HEIGHT
    cx = int(round(frame_x))
    cy = int(round(frame_y))

    x_min = max(0, cx - roi_radius)
    x_max = min(width, cx + roi_radius + 1)
    y_min = max(0, cy - roi_radius)
    y_max = min(height, cy + roi_radius + 1)
    if x_max <= x_min or y_max <= y_min:
        return x, y

    roi = frame[y_min:y_max, x_min:x_max]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    value_floor = max(95.0, float(np.percentile(val, 70)))
    sat_floor = max(35.0, float(np.percentile(sat, 45)))
    color_mask = (
        (hue >= 18) & (hue <= 58) &
        (sat >= sat_floor) &
        (val >= value_floor)
    )
    bright_ball_mask = (
        (hue >= 15) & (hue <= 65) &
        (sat >= 25) &
        (val >= max(85.0, float(np.percentile(val, 55))))
    )
    mask = color_mask | bright_ball_mask

    if prev_frame is not None:
        prev_roi = prev_frame[y_min:y_max, x_min:x_max]
        if prev_roi.shape == roi.shape:
            diff = cv2.absdiff(roi, prev_roi)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            diff_floor = max(18.0, float(np.percentile(diff_gray, 85)))
            motion_mask = diff_gray >= diff_floor
            mask = mask | (motion_mask & bright_ball_mask)

    mask = mask.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return x, y

    local_pred = np.array([frame_x - x_min, frame_y - y_min], dtype=np.float32)
    best = None
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        ys, xs = np.where(labels == label)
        weights = sat[ys, xs].astype(np.float32) + val[ys, xs].astype(np.float32)
        if float(weights.sum()) <= 0:
            continue

        center_x = float((weights * xs).sum() / weights.sum())
        center_y = float((weights * ys).sum() / weights.sum())
        dist = float(np.linalg.norm(np.array([center_x, center_y]) - local_pred))
        if dist > roi_radius * 0.95:
            continue

        x_span = float(xs.max() - xs.min() + 1)
        y_span = float(ys.max() - ys.min() + 1)
        elongation = max(x_span, y_span) / max(min(x_span, y_span), 1.0)
        area_penalty = 0.0 if area <= max_area * 0.45 else area / max_area
        streak_bonus = min(elongation, 5.0) * 0.75
        rank = dist + area_penalty * 8.0 - streak_bonus
        if best is None or rank < best[0]:
            best = (rank, center_x + x_min, center_y + y_min)

    if best is None:
        return x, y

    _, refined_frame_x, refined_frame_y = best
    return refined_frame_x * TRACK_WIDTH / width, refined_frame_y * TRACK_HEIGHT / height

def read_video(path_video):
    """ Read video file    
    :params
        path_video: path to video file
    :return
        frames: list of video frames
        fps: frames per second
    """
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps

def infer_model(frames, model, return_scores=False, min_confidence=MIN_TRACK_CONFIDENCE,
                centroid_radius=8, relative_threshold=0.6, max_candidates=5,
                max_candidate_dist=260, score_distance_tradeoff=0.35,
                visual_refine=True, visual_roi_radius=30, fast_ball=True,
                fast_speed_threshold=70, fast_min_confidence=55):
    """ Run pretrained model on a consecutive list of frames    
    :params
        frames: list of consecutive video frames
        model: pretrained model
    :return    
        ball_track: list of detected ball points
        dists: list of euclidean distances between two neighbouring ball points
    """
    height = 360
    width = 640
    dists = [-1]*2
    ball_track = [(None,None)]*2
    scores = [0.0]*2
    for num in tqdm(range(2, len(frames))):
        img = cv2.resize(frames[num], (width, height))
        img_prev = cv2.resize(frames[num-1], (width, height))
        img_preprev = cv2.resize(frames[num-2], (width, height))
        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32)/255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)

        with torch.no_grad():
            out = model(torch.from_numpy(inp).float().to(device))
        output = out.argmax(dim=1).detach().cpu().numpy()
        motion = estimate_recent_motion(ball_track)
        speed = motion[2] if motion is not None else 0.0
        candidate_threshold = min_confidence
        adaptive_max_dist = max_candidate_dist
        adaptive_roi_radius = visual_roi_radius
        adaptive_candidates = max_candidates
        if fast_ball and speed >= fast_speed_threshold:
            candidate_threshold = min(min_confidence, fast_min_confidence)
            adaptive_max_dist = max(max_candidate_dist, min(620.0, max_candidate_dist + speed * 2.0))
            adaptive_roi_radius = int(max(visual_roi_radius, min(95.0, visual_roi_radius + speed * 0.45)))
            adaptive_candidates = max(max_candidates, 8)

        candidates = postprocess_candidates(
            output,
            peak_threshold=candidate_threshold,
            centroid_radius=centroid_radius,
            relative_threshold=relative_threshold,
            max_candidates=adaptive_candidates,
        )
        if fast_ball and not candidates and min_confidence > fast_min_confidence:
            candidates = postprocess_candidates(
                output,
                peak_threshold=fast_min_confidence,
                centroid_radius=max(centroid_radius, 10),
                relative_threshold=max(0.45, relative_threshold - 0.15),
                max_candidates=adaptive_candidates,
            )
        x_pred, y_pred, score = choose_temporal_candidate(
            candidates,
            ball_track,
            max_candidate_dist=adaptive_max_dist,
            score_distance_tradeoff=score_distance_tradeoff,
        )
        if visual_refine:
            x_pred, y_pred = refine_visual_center(
                frames[num],
                x_pred,
                y_pred,
                roi_radius=adaptive_roi_radius,
                prev_frame=frames[num - 1] if num > 0 else None,
            )
        if fast_ball and candidate_threshold < min_confidence and score >= candidate_threshold:
            scores.append(max(score, min_confidence))
        else:
            scores.append(score)
        ball_track.append((x_pred, y_pred))

        if ball_track[-1][0] and ball_track[-2][0]:
            dist = distance.euclidean(ball_track[-1], ball_track[-2])
        else:
            dist = -1
        dists.append(dist)  
    if return_scores:
        return ball_track, dists, scores
    return ball_track, dists

def remove_outliers(ball_track, dists, max_dist = 100):
    """ Remove outliers from model prediction    
    :params
        ball_track: list of detected ball points
        dists: list of euclidean distances between two neighbouring ball points
        max_dist: maximum distance between two neighbouring ball points
    :return
        ball_track: list of ball points
    """
    outliers = list(np.where(np.array(dists) > max_dist)[0])
    for i in outliers:
        keep_fast_motion = False
        if 0 < i < len(ball_track) - 1:
            prev_point = ball_track[i - 1]
            curr_point = ball_track[i]
            next_point = ball_track[i + 1]
            if (
                prev_point[0] is not None and curr_point[0] is not None and
                next_point[0] is not None
            ):
                prev_vec = np.array([curr_point[0] - prev_point[0], curr_point[1] - prev_point[1]])
                next_vec = np.array([next_point[0] - curr_point[0], next_point[1] - curr_point[1]])
                prev_norm = float(np.linalg.norm(prev_vec))
                next_norm = float(np.linalg.norm(next_vec))
                if prev_norm > 0 and next_norm > 0:
                    cosine = float(np.dot(prev_vec, next_vec) / (prev_norm * next_norm))
                    keep_fast_motion = cosine > 0.25 and next_norm <= max_dist * 2.8
        if keep_fast_motion:
            continue
        if i + 1 >= len(dists):
            ball_track[i] = (None, None)
        elif (dists[i+1] > max_dist) | (dists[i+1] == -1):       
            ball_track[i] = (None, None)
            outliers.remove(i)
        elif dists[i-1] == -1:
            ball_track[i-1] = (None, None)
    return ball_track  

def split_track(ball_track, max_gap=4, max_dist_gap=80, min_track=5):
    """ Split ball track into several subtracks in each of which we will perform
    ball interpolation.    
    :params
        ball_track: list of detected ball points
        max_gap: maximun number of coherent None values for interpolation  
        max_dist_gap: maximum distance at which neighboring points remain in one subtrack
        min_track: minimum number of frames in each subtrack    
    :return
        result: list of subtrack indexes    
    """
    list_det = [0 if x[0] else 1 for x in ball_track]
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

    cursor = 0
    min_value = 0
    result = []
    for i, (k, l) in enumerate(groups):
        if (k == 1) & (i > 0) & (i < len(groups) - 1):
            dist = distance.euclidean(ball_track[cursor-1], ball_track[cursor+l])
            if (l >=max_gap) | (dist/l > max_dist_gap):
                if cursor - min_value > min_track:
                    result.append([min_value, cursor])
                    min_value = cursor + l - 1        
        cursor += l
    if len(list_det) - min_value > min_track: 
        result.append([min_value, len(list_det)]) 
    return result    

def interpolation(coords):
    """ Run ball interpolation in one subtrack    
    :params
        coords: list of ball coordinates of one subtrack    
    :return
        track: list of interpolated ball coordinates of one subtrack
    """
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    x = np.array([x[0] if x[0] is not None else np.nan for x in coords])
    y = np.array([x[1] if x[1] is not None else np.nan for x in coords])

    nons, yy = nan_helper(x)
    if np.count_nonzero(~nons) == 0:
        return coords
    x[nons]= np.interp(yy(nons), yy(~nons), x[~nons])
    nans, xx = nan_helper(y)
    if np.count_nonzero(~nans) == 0:
        return coords
    y[nans]= np.interp(xx(nans), xx(~nans), y[~nans])

    track = [*zip(x,y)]
    return track

def smooth_track(ball_track, window_size=3):
    """Smooth contiguous valid track segments with a centered moving average."""
    if window_size < 3 or window_size % 2 == 0:
        return ball_track

    smoothed_track = list(ball_track)
    segment_start = None
    for idx, point in enumerate(ball_track + [(None, None)]):
        is_valid = point[0] is not None and point[1] is not None
        if is_valid and segment_start is None:
            segment_start = idx
        if not is_valid and segment_start is not None:
            segment = ball_track[segment_start:idx]
            if len(segment) >= window_size:
                x_vals = np.array([p[0] for p in segment], dtype=np.float32)
                y_vals = np.array([p[1] for p in segment], dtype=np.float32)
                kernel = np.ones(window_size, dtype=np.float32) / window_size
                x_pad = np.pad(x_vals, (window_size // 2, window_size // 2), mode='edge')
                y_pad = np.pad(y_vals, (window_size // 2, window_size // 2), mode='edge')
                x_smooth = np.convolve(x_pad, kernel, mode='valid')
                y_smooth = np.convolve(y_pad, kernel, mode='valid')
                for offset in range(len(segment)):
                    smoothed_track[segment_start + offset] = (float(x_smooth[offset]), float(y_smooth[offset]))
            segment_start = None
    return smoothed_track

def kalman_smooth_track(ball_track, scores=None, min_confidence=MIN_TRACK_CONFIDENCE,
                        max_prediction_gap=8, max_gate_dist=320,
                        min_gate_detections=4,
                        return_statuses=False):
    """Smooth track with a constant-velocity Kalman filter and fill short gaps."""
    if scores is None:
        scores = [min_confidence] * len(ball_track)

    state = None
    covariance = None
    transition = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1]],
        dtype=np.float32,
    )
    measurement = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]],
        dtype=np.float32,
    )
    process_noise = np.diag([0.5, 0.5, 4.0, 4.0]).astype(np.float32)
    identity = np.eye(4, dtype=np.float32)

    result = []
    statuses = []
    missed = 0
    accepted_detections = 0
    for idx, (x, y) in enumerate(ball_track):
        score = scores[idx] if idx < len(scores) else min_confidence
        has_measurement = x is not None and y is not None and score >= min_confidence

        if state is None:
            if has_measurement:
                state = np.array([[x], [y], [0], [0]], dtype=np.float32)
                covariance = np.diag([25.0, 25.0, 100.0, 100.0]).astype(np.float32)
                result.append((float(x), float(y)))
                statuses.append('detected')
                accepted_detections += 1
            else:
                result.append((None, None))
                statuses.append('missing')
            continue

        state = transition @ state
        covariance = transition @ covariance @ transition.T + process_noise

        if has_measurement:
            z = np.array([[x], [y]], dtype=np.float32)
            predicted_xy = measurement @ state
            gate_dist = float(np.linalg.norm(z - predicted_xy))
            gate_ready = accepted_detections >= min_gate_detections and missed == 0
            if gate_ready and gate_dist > max_gate_dist:
                if missed < max_prediction_gap:
                    missed += 1
                    result.append((float(state[0, 0]), float(state[1, 0])))
                    statuses.append('rejected')
                else:
                    missed += 1
                    result.append((None, None))
                    statuses.append('missing')
                continue

            score_margin = max(float(score - min_confidence), 0.0)
            noise = max(4.0, 64.0 / (1.0 + score_margin / 20.0))
            measurement_noise = np.eye(2, dtype=np.float32) * noise
            innovation = z - measurement @ state
            innovation_covariance = measurement @ covariance @ measurement.T + measurement_noise
            kalman_gain = covariance @ measurement.T @ np.linalg.inv(innovation_covariance)
            state = state + kalman_gain @ innovation
            covariance = (identity - kalman_gain @ measurement) @ covariance
            missed = 0
            result.append((float(state[0, 0]), float(state[1, 0])))
            statuses.append('detected')
            accepted_detections += 1
        elif missed < max_prediction_gap:
            missed += 1
            result.append((float(state[0, 0]), float(state[1, 0])))
            statuses.append('predicted')
        else:
            missed += 1
            result.append((None, None))
            statuses.append('missing')

    if return_statuses:
        return result, statuses
    return result

def offline_smooth_track(ball_track, window_size=7, polyorder=2):
    """Use nearby past and future points to reduce frame-to-frame jitter."""
    if window_size < 3 or window_size % 2 == 0:
        return ball_track

    smoothed_track = list(ball_track)
    segment_start = None
    for idx, point in enumerate(ball_track + [(None, None)]):
        is_valid = point[0] is not None and point[1] is not None
        if is_valid and segment_start is None:
            segment_start = idx
        if not is_valid and segment_start is not None:
            segment = ball_track[segment_start:idx]
            if len(segment) >= window_size and len(segment) > polyorder:
                x_vals = np.array([p[0] for p in segment], dtype=np.float32)
                y_vals = np.array([p[1] for p in segment], dtype=np.float32)
                local_window = min(window_size, len(segment))
                if local_window % 2 == 0:
                    local_window -= 1
                if local_window > polyorder:
                    x_smooth = savgol_filter(x_vals, local_window, polyorder, mode='interp')
                    y_smooth = savgol_filter(y_vals, local_window, polyorder, mode='interp')
                    for offset in range(len(segment)):
                        smoothed_track[segment_start + offset] = (
                            float(x_smooth[offset]),
                            float(y_smooth[offset]),
                        )
            segment_start = None
    return smoothed_track


def bridge_short_gaps(ball_track, statuses=None, max_gap=8):
    """Fill short remaining holes between two valid points with a straight segment."""
    bridged_track = list(ball_track)
    bridged_statuses = list(statuses) if statuses is not None else None
    idx = 0
    while idx < len(bridged_track):
        if bridged_track[idx][0] is not None:
            idx += 1
            continue

        gap_start = idx
        while idx < len(bridged_track) and bridged_track[idx][0] is None:
            idx += 1
        gap_end = idx
        gap_len = gap_end - gap_start
        left = gap_start - 1
        right = gap_end
        if gap_len > max_gap or left < 0 or right >= len(bridged_track):
            continue

        left_x, left_y = bridged_track[left]
        right_x, right_y = bridged_track[right]
        if None in (left_x, left_y, right_x, right_y):
            continue

        step_x = (right_x - left_x) / (gap_len + 1)
        step_y = (right_y - left_y) / (gap_len + 1)
        for offset in range(1, gap_len + 1):
            fill_idx = left + offset
            bridged_track[fill_idx] = (
                float(left_x + step_x * offset),
                float(left_y + step_y * offset),
            )
            if bridged_statuses is not None:
                bridged_statuses[fill_idx] = 'predicted'

    if bridged_statuses is not None:
        return bridged_track, bridged_statuses
    return bridged_track


def suppress_isolated_detections(ball_track, statuses, window=3, max_link_dist=140):
    """Mark direct detections as isolated when no nearby detection follows the same track."""
    if statuses is None:
        return statuses

    filtered_statuses = list(statuses)
    for idx, status in enumerate(statuses):
        if status != 'detected':
            continue

        x, y = ball_track[idx]
        if x is None or y is None:
            filtered_statuses[idx] = 'isolated'
            continue

        linked = False
        start = max(0, idx - window)
        end = min(len(ball_track), idx + window + 1)
        for other_idx in range(start, end):
            if other_idx == idx or statuses[other_idx] != 'detected':
                continue

            other_x, other_y = ball_track[other_idx]
            if other_x is None or other_y is None:
                continue

            gap = abs(other_idx - idx)
            dist = float(np.linalg.norm(np.array([x - other_x, y - other_y])))
            if dist <= max_link_dist * max(1, gap):
                linked = True
                break

        if not linked:
            filtered_statuses[idx] = 'isolated'

    return filtered_statuses


def suppress_jump_detections(ball_track, statuses, window=6, max_interp_error=90,
                             max_step_dist=190):
    """Mark detections as jumps when they break the local trajectory."""
    if statuses is None:
        return statuses

    filtered_statuses = list(statuses)
    detected_indexes = [
        idx for idx, status in enumerate(statuses)
        if status == 'detected' and ball_track[idx][0] is not None
    ]
    detected_set = set(detected_indexes)

    for idx in detected_indexes:
        x, y = ball_track[idx]
        prev_idx = None
        next_idx = None

        for candidate in range(idx - 1, max(-1, idx - window - 1), -1):
            if candidate in detected_set:
                prev_idx = candidate
                break
        for candidate in range(idx + 1, min(len(ball_track), idx + window + 1)):
            if candidate in detected_set:
                next_idx = candidate
                break

        if prev_idx is None and next_idx is None:
            filtered_statuses[idx] = 'isolated'
            continue

        if prev_idx is not None:
            prev_x, prev_y = ball_track[prev_idx]
            prev_gap = idx - prev_idx
            prev_dist = float(np.linalg.norm(np.array([x - prev_x, y - prev_y])))
            if next_idx is None and prev_dist > max_step_dist * max(1, prev_gap):
                filtered_statuses[idx] = 'jump'
                continue

        if next_idx is not None:
            next_x, next_y = ball_track[next_idx]
            next_gap = next_idx - idx
            next_dist = float(np.linalg.norm(np.array([x - next_x, y - next_y])))
            if prev_idx is None and next_dist > max_step_dist * max(1, next_gap):
                filtered_statuses[idx] = 'jump'
                continue

        if prev_idx is None or next_idx is None:
            continue

        prev_x, prev_y = ball_track[prev_idx]
        next_x, next_y = ball_track[next_idx]
        total_gap = next_idx - prev_idx
        if total_gap <= 0:
            continue

        ratio = (idx - prev_idx) / total_gap
        expected_x = prev_x + (next_x - prev_x) * ratio
        expected_y = prev_y + (next_y - prev_y) * ratio
        interp_error = float(np.linalg.norm(np.array([x - expected_x, y - expected_y])))

        prev_dist = float(np.linalg.norm(np.array([x - prev_x, y - prev_y])))
        next_dist = float(np.linalg.norm(np.array([x - next_x, y - next_y])))
        local_step_limit = max_step_dist * max(1, min(idx - prev_idx, next_idx - idx))
        if interp_error > max_interp_error and min(prev_dist, next_dist) > local_step_limit:
            filtered_statuses[idx] = 'jump'

    return filtered_statuses

def write_track(frames, ball_track, path_output_video, fps, trace=7, statuses=None,
                scores=None, debug=False, detected_only=False):
    """ Write .avi file with detected ball tracks
    :params
        frames: list of original video frames
        ball_track: list of ball coordinates
        path_output_video: path to output video
        fps: frames per second
        trace: number of frames with detected trace
    """
    height, width = frames[0].shape[:2]
    scale_x = width / TRACK_WIDTH
    scale_y = height / TRACK_HEIGHT
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'DIVX'), 
                          fps, (width, height))
    for num in range(len(frames)):
        frame = frames[num].copy()
        for i in range(trace):
            if (num-i > 0):
                if ball_track[num-i][0]:
                    x = int(ball_track[num-i][0] * scale_x)
                    y = int(ball_track[num-i][1] * scale_y)
                    status = statuses[num-i] if statuses and num-i < len(statuses) else 'detected'
                    if detected_only and not debug and status != 'detected':
                        continue
                    if debug:
                        if status == 'predicted':
                            color = (0, 255, 255)
                        elif status == 'rejected':
                            color = (255, 0, 255)
                        elif status == 'missing':
                            color = (128, 128, 128)
                        elif status == 'isolated':
                            color = (160, 160, 160)
                        elif status == 'jump':
                            color = (255, 128, 0)
                        else:
                            color = (0, 0, 255)
                    else:
                        color = (0, 0, 255)
                    frame = cv2.circle(frame, (x,y), radius=0, color=color, thickness=10-i)
                else:
                    break
        if debug:
            status = statuses[num] if statuses and num < len(statuses) else 'missing'
            score = scores[num] if scores and num < len(scores) else 0.0
            label = f'{num} {status} conf={score:.1f}'
            cv2.putText(frame, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, 'red=detected yellow=kalman magenta=rejected gray=isolated blue=jump', (20, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                        cv2.LINE_AA)
        out.write(frame) 
    out.release()    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--video_path', type=str, help='path to input video')
    parser.add_argument('--video_out_path', type=str, help='path to output video')
    parser.add_argument('--extrapolation', action='store_true', help='whether to use ball track extrapolation')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='device to use for inference')
    parser.add_argument('--min_confidence', type=float, default=MIN_TRACK_CONFIDENCE,
                        help='minimum heatmap confidence for a detection')
    parser.add_argument('--max_prediction_gap', type=int, default=8,
                        help='maximum consecutive frames to fill with Kalman prediction')
    parser.add_argument('--max_dist', type=float, default=100,
                        help='maximum plausible distance between neighboring track points')
    parser.add_argument('--max_gate_dist', type=float, default=320,
                        help='maximum distance between Kalman prediction and accepted detection')
    parser.add_argument('--min_gate_detections', type=int, default=4,
                        help='minimum accepted detections before enabling Kalman gate')
    parser.add_argument('--offline_window', type=int, default=7,
                        help='odd Savitzky-Golay window for offline smoothing')
    parser.add_argument('--offline_polyorder', type=int, default=2,
                        help='Savitzky-Golay polynomial order for offline smoothing')
    parser.add_argument('--centroid_radius', type=int, default=8,
                        help='local heatmap radius used to estimate ball center')
    parser.add_argument('--relative_threshold', type=float, default=0.6,
                        help='relative heatmap threshold used for local blob center')
    parser.add_argument('--max_candidates', type=int, default=5,
                        help='number of heatmap candidates to compare with temporal motion')
    parser.add_argument('--max_candidate_dist', type=float, default=260,
                        help='maximum plausible candidate distance from temporal prediction')
    parser.add_argument('--score_distance_tradeoff', type=float, default=0.35,
                        help='candidate score penalty when choosing the closest temporal point')
    parser.add_argument('--no_visual_refine', action='store_true',
                        help='disable local color/blob refinement around TrackNet detections')
    parser.add_argument('--visual_roi_radius', type=int, default=30,
                        help='original-frame pixel radius for local ball-center refinement')
    parser.add_argument('--no_fast_ball', action='store_true',
                        help='disable adaptive high-speed ball tracking')
    parser.add_argument('--fast_speed_threshold', type=float, default=70,
                        help='track-space speed threshold for high-speed mode')
    parser.add_argument('--fast_min_confidence', type=float, default=55,
                        help='lower heatmap threshold used only in high-speed mode')
    parser.add_argument('--max_bridge_gap', type=int, default=8,
                        help='maximum remaining internal gap to bridge after Kalman filtering')
    parser.add_argument('--debug', action='store_true',
                        help='draw detected and Kalman-predicted points differently')
    parser.add_argument('--detected_only', action='store_true',
                        help='draw only frames with direct model detections in non-debug output')
    parser.add_argument('--suppress_isolated', action='store_true',
                        help='hide direct detections that are not connected to nearby detections')
    parser.add_argument('--isolation_window', type=int, default=3,
                        help='neighboring frame window used to suppress isolated detections')
    parser.add_argument('--isolation_max_dist', type=float, default=140,
                        help='maximum per-frame distance used to link detections')
    parser.add_argument('--suppress_jumps', action='store_true',
                        help='hide direct detections that break the local trajectory')
    parser.add_argument('--jump_window', type=int, default=6,
                        help='neighboring detection window used to suppress jumps')
    parser.add_argument('--jump_max_interp_error', type=float, default=90,
                        help='maximum interpolation error before marking a detection as a jump')
    parser.add_argument('--jump_max_step_dist', type=float, default=190,
                        help='maximum per-frame step distance for jump suppression')
    args = parser.parse_args()
    
    model = BallTrackerNet()
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    frames, fps = read_video(args.video_path)
    ball_track, dists, scores = infer_model(
        frames,
        model,
        return_scores=True,
        min_confidence=args.min_confidence,
        centroid_radius=args.centroid_radius,
        relative_threshold=args.relative_threshold,
        max_candidates=args.max_candidates,
        max_candidate_dist=args.max_candidate_dist,
        score_distance_tradeoff=args.score_distance_tradeoff,
        visual_refine=not args.no_visual_refine,
        visual_roi_radius=args.visual_roi_radius,
        fast_ball=not args.no_fast_ball,
        fast_speed_threshold=args.fast_speed_threshold,
        fast_min_confidence=args.fast_min_confidence,
    )
    ball_track = remove_outliers(ball_track, dists, max_dist=args.max_dist)    
    
    if args.extrapolation:
        subtracks = split_track(ball_track)
        for r in subtracks:
            ball_subtrack = ball_track[r[0]:r[1]]
            ball_subtrack = interpolation(ball_subtrack)
            ball_track[r[0]:r[1]] = ball_subtrack
    ball_track, statuses = kalman_smooth_track(
        ball_track,
        scores,
        min_confidence=args.min_confidence,
        max_prediction_gap=args.max_prediction_gap,
        max_gate_dist=args.max_gate_dist,
        min_gate_detections=args.min_gate_detections,
        return_statuses=True,
    )
    ball_track, statuses = bridge_short_gaps(
        ball_track,
        statuses,
        max_gap=args.max_bridge_gap,
    )
    ball_track = offline_smooth_track(
        ball_track,
        window_size=args.offline_window,
        polyorder=args.offline_polyorder,
    )
    ball_track = smooth_track(ball_track)
    if args.suppress_isolated:
        statuses = suppress_isolated_detections(
            ball_track,
            statuses,
            window=args.isolation_window,
            max_link_dist=args.isolation_max_dist,
        )
    if args.suppress_jumps:
        statuses = suppress_jump_detections(
            ball_track,
            statuses,
            window=args.jump_window,
            max_interp_error=args.jump_max_interp_error,
            max_step_dist=args.jump_max_step_dist,
        )
        
    write_track(
        frames,
        ball_track,
        args.video_out_path,
        fps,
        statuses=statuses,
        scores=scores,
        debug=args.debug,
        detected_only=args.detected_only,
    )    
    
    
    
    
    
