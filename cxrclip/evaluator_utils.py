
import logging
import numpy as np
from sklearn import metrics
import torch
from tqdm import tqdm
from statistics import mean
from collections import Counter
import cv2
import os
import torch.nn.functional as F
from torchmetrics.segmentation import DiceScore
from sklearn.metrics import roc_auc_score
import pickle
from cxrclip.prompt import constants
import matplotlib.pyplot as plt
import matplotlib.cm as cm
log = logging.getLogger(__name__)

def get_class_list(dataset_name):
    return getattr(constants, dataset_name.upper())

def draw_adaptive_title_safe(
    img,
    text,
    align="left",
    margin_ratio=0.03,
    max_font_ratio=0.17,
    thickness_ratio=0.005,
    text_color=(0, 200, 0), # draw the text in green to make it apparent
    min_font_scale=1
):
    """
    Fully safe adaptive text rendering.
    Automatically shrinks font until it fits width and height.

    img,
    text,
    align="left",              # "left" or "center"
    margin_ratio=0.03,         # margin as % of image width
    font_ratio=0.2,           # font scale relative to width
    thickness_ratio=0.004,     # thickness relative to width
    text_color=(255, 255, 255)
    """

    img_out = img.copy()
    h, w = img_out.shape[:2]

    text = text.title()
    font = cv2.FONT_HERSHEY_SIMPLEX

    margin = int(w * margin_ratio)
    thickness = max(1, int(w * thickness_ratio))

    # Start from maximum font scale
    font_scale = w * max_font_ratio / 100

    # Reduce font size until it fits
    while True:
        (text_w, text_h), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        fits_width = text_w <= (w - 2 * margin)
        fits_height = (text_h + baseline + margin) <= h

        if fits_width and fits_height:
            break

        font_scale *= 0.95  # shrink gradually

        if font_scale < min_font_scale:
            break

    # Compute final position
    if align == "center":
        x = (w - text_w) // 2
    else:
        x = margin

    y = text_h + baseline + margin

    cv2.putText(
        img_out,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )

    return img_out

def draw_adaptive_title(
    img,
    text,
    align="left",              # "left" or "center"
    margin_ratio=0.03,         # margin as % of image width
    font_ratio=0.2,           # font scale relative to width
    thickness_ratio=0.004,     # thickness relative to width
    text_color=(255, 255, 255)
):
    """
    Draw adaptive text that scales with image size.
    No shadow version.
    """

    img_out = img.copy()
    h, w = img_out.shape[:2]

    text = text.title()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ---- Scale parameters based on image width ----
    margin = int(w * margin_ratio)
    font_scale = w * font_ratio / 100
    thickness = max(1, int(w * thickness_ratio))

    # ---- Get text size ----
    (text_w, text_h), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    # ---- Compute safe position ----
    if align == "center":
        x = (w - text_w) // 2
    else:
        x = margin

    y = text_h + baseline + margin

    # Clamp vertically if needed
    if y >= h:
        y = h - baseline - 5

    # ---- Draw text ----
    cv2.putText(
        img_out,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )

    return img_out

def compute_font_scale(img, target_ratio=0.03, thickness=2):
    H = img.shape[0]
    target_height = H * target_ratio
    
    # get base height at scale=1
    (_, base_h), _ = cv2.getTextSize(
        "Test",
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.0,
        thickness=thickness
    )
    
    return target_height / base_h

def define_fname_by_path(img_path):

    # Normalize path (handles Windows/Linux)
    img_path = os.path.normpath(img_path)
    parts = img_path.split(os.sep)
    
    if 'chexlocalize' in parts:
        idx = parts.index('chexlocalize')
        
        # Exclude filename (last element)
        relevant_parts = parts[idx:-1]
        # Add filename without extension
        file_stem = os.path.splitext(parts[-1])[0]
        fname = "_".join(relevant_parts + [file_stem])
    else:
        fname = '.'.join(os.path.basename(img_path).split('.')[:-1])
    return fname

def compute_specificity(negative_probs, threshold):
    """
    Calculate image-level specificity.

    Specificity is the proportion of true negatives (images correctly identified as negative)
    out of the total number of negatives.

    Args:
        negative_probs (torch.Tensor): Tensor of predicted probabilities for negative images.
        threshold (float): Threshold to classify the predictions.

    Returns:
        float: Specificity value.
    """
    # Check if all pixels in the negative images are below the threshold
    true_negatives = (
        (negative_probs.squeeze(1) > threshold).long().sum(-1).sum(-1) == 0
    ).sum()

    # Calculate specificity as the ratio of true negatives to the total number of negative images
    spec = (true_negatives / len(negative_probs)).item()
    return spec

def get_grounding_type(dataset_name):
    return 'pointing_game'

def gather_pointinggame_statistics(
        class_list, 
        attention_maps, 
        bbox_labels, 
        label_names, 
        image_paths, 
        grounding_type='pointing_game',
        input_res=None,
        visual_save_dir=None,
        visual_name_suffix='',
        draw_box_on_overlay=True
    ):
    assert grounding_type in ['pointing_game', 'contour', 'refer_grounding', 'mask']
    """
    pointinggame_predictions is shape [number of images, number of class, prediction of each class]
    """
    assert len(bbox_labels) == len(label_names) and len(label_names) == len(image_paths), "size does not match."
    bbox_labels = list(zip(bbox_labels, label_names))
    B = len(bbox_labels)
    Q = len(class_list)

    bbox_tensor = [[[] for _ in range(Q)] for _ in range(B)]
    for i, (bboxes, names) in enumerate(bbox_labels):
        for bbox, disease_name in zip(bboxes, names):
            if disease_name in class_list: # exact match case insensitive.
                q = class_list.index(disease_name)
                bbox_tensor[i][q].append(list(bbox))  # APPEND, don't overwrite
    
    if isinstance(attention_maps, np.ndarray):
        attention_maps = torch.from_numpy(attention_maps).clone()
    else:
        attention_maps = attention_maps.clone()

    if grounding_type in {'pointing_game', 'contour', 'mask'}:
        results = pointing_game(
            attention_maps, 
            bbox_tensor, 
            image_paths, 
            label_names, 
            class_list, 
            grounding_type=grounding_type,
            input_res=input_res,
            save_dir=visual_save_dir,
            visual_fname_suffix=visual_name_suffix,
            draw_box_on_overlay=draw_box_on_overlay
        )
    else:
        results = pointing_game_flat(
            attention_maps, 
            bbox_tensor, 
            image_paths, 
            class_list,
            input_res=input_res,
            save_dir=visual_save_dir,
            visual_fname_suffix=visual_name_suffix,
            draw_box_on_overlay=draw_box_on_overlay
            )
    return results

def pointing_game(
        attn, 
        bbox, 
        image_paths, 
        label_names, 
        class_list, 
        grounding_type,
        input_res=None,
        save_dir=None,
        visual_fname_suffix='',
        draw_box_on_overlay=True
    ):
    """
    Pointing Game metric in the same *style* as chestXDet10_eval_grounding:
    - For each (image, query), get a spatial score map over patches.
    - Convert patch grid -> original image resolution (H0, W0) by bilinear upsampling.
    - Take argmax point in ORIGINAL coords.
    - Hit if point is inside ANY GT box for that (image, query).
    - Skip samples where no GT box exists for that (image, query).
    
    Args:
        attn:  [B, Q, T] attention/similarity over tokens (may include CLS)
        bbox:  nested boxes. bbox[i][q] should be:
            - [] or None if absent, OR
            - list of boxes [[xmin,ymin,xmax,ymax], ...] in ORIGINAL image coords
        image_paths: list length B, used to read original sizes
        class_list: list length Q

    Returns:
        results: dict[class_name]['pointing_game'] = hit-rate over images where GT present
    """
    results = {}
    B, Q, T = attn.shape
    assert T > 1
    assert T in {1370, 1369, 257, 256, 325, 324, 197, 196, 49}, \
        "Unexpected token count. Ensure attn includes/excludes CLS consistently."

    # ---- remove CLS if present ----
    if T in {1370, 257, 325, 197}:
        spatial = attn[:, :, 1:]   # [B, Q, HW]
    else:
        spatial = attn             # [B, Q, HW]
    
    if input_res is not None:
        S = input_res
    elif T == 1370 or T == 1369: S = 518
    elif T == 325 or T == 324: S = 256
    elif T == 257 or T == 256: S = 224
    else: S = 224
    # NOTE: for biovil-t 196 corresponds to 448 input size

    HW = spatial.shape[-1]
    Hg = Wg = int(HW ** 0.5)
    assert Hg * Wg == HW, "Token count (without CLS) must form a square grid."

    # [B, Q, Hg, Wg]
    spatial = spatial.reshape(B, Q, Hg, Wg)

    # ---- load original sizes once ----
    img_data = []
    for path in tqdm(image_paths, desc="Loading image dimensions"):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"cv2 cannot read image: {path}")
        img_data.append(img)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    def _boxes_for(i, q):
        """Return list of boxes for sample (i,q) or empty list."""
        b = bbox[i][q]
        if b is None:
            return []
        # already list-of-boxes
        if isinstance(b, (list, tuple)) and len(b) > 0 and isinstance(b[0], (list, tuple, np.ndarray)):
            return b
        # single box
        if isinstance(b, (list, tuple, np.ndarray)) and len(b) == 4:
            return [b]
        return []

    def _point_in_any_box(box_list, x, y):
        for box in box_list:
            x_min, y_min, x_max, y_max = box
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return True
        return False

    def _pointing_game_hit_by_mask(mask, x, y):
        """
        mainly for siim and other datasets that provides the segmentation masks only
        mainly for pointing game for segmentation mask instead of bounding box.
        heat: (H, W) float heatmap
        mask: (H, W) binary mask for the class
        """
        y, x = np.unravel_index(np.argmax(heat), heat.shape)
        hit = bool(mask[y, x] > 0) # TODO: make sure the axis is done correctly
        return hit
    
    def _hit_by_contours(contours_xy, x, y):
        """
        mainly for chestXlocalize
        point_xy: (x, y) in image pixel coordinates
        contours_xy: list of polygons, each polygon is [(x,y), (x,y), ...], essentially a list of list.
        returns: True if inside ANY polygon
        """
        pt = (float(x), float(y))
        for poly in contours_xy:
            poly_np = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)  # (N,1,2)
            inside = cv2.pointPolygonTest(poly_np, pt, measureDist=False)  # 1 inside, 0 on edge, -1 outside
            if inside >= 0:
                return True
        return False

    assert grounding_type in ['pointing_game', 'contour', 'mask']
    if grounding_type == 'pointing_game':
        hit_func = _point_in_any_box
    elif grounding_type == 'contour':
        hit_func = _hit_by_contours
    elif grounding_type == 'mask':
        hit_func = _pointing_game_hit_by_mask

    for q in range(Q):
        hits_q, count_q = 0, 0
        cls_name = class_list[q]

        for i in range(B):
            box_list = _boxes_for(i, q)
            if len(box_list) == 0:
                continue  # skip if no GT for this (image, query)

            orig_img = img_data[i].copy()
            H0, W0 = orig_img.shape[:2]
            # assert H0 == W0,\
            # 'The following implementation assume the original image is in squared size. Otherwise it need to handle the Resize() operation part.'
            
            heat = spatial[i, q].unsqueeze(0).unsqueeze(0)  # [1,1,Hg,Wg]
            if H0 == W0:
                # ---- upsample patch map to ORIGINAL image size ----
                heat = F.interpolate(
                    heat, size=(H0, W0),
                    mode="bilinear", 
                    align_corners=False # TODO:
                ).squeeze(0).squeeze(0)  # [H0, W0]
            else:
                # scale so that min(H0, W0) -> S
                scale = S / float(min(H0, W0))

                # resized size
                Hr = int(round(H0 * scale))
                Wr = int(round(W0 * scale))

                # center crop offsets in resized image
                top  = max((Hr - S) // 2, 0)
                left = max((Wr - S) // 2, 0)

                # 1) upsample patch heatmap to crop size (S,S), NOT (H0,W0)
                heat = F.interpolate(heat, size=(S, S), mode="bilinear", align_corners=False)[0, 0]

            # argmax in ORIGINAL coords
            idx = int(torch.argmax(heat).item())
            if H0 == W0:
                y_peak = idx // W0
                x_peak = idx % W0
            else:
                y_c = idx // S
                x_c = idx % S

                # 3) crop coords -> resized coords
                x_r = x_c + left
                y_r = y_c + top
                # 4) resized coords -> original coords
                x_peak = int(round(x_r / scale))
                y_peak = int(round(y_r / scale))

            # route to different method based on the dataset: bounding box, contour, and segmentation
            hit = hit_func(box_list, x_peak, y_peak)
            hits_q += int(hit)
            count_q += 1

            # === PRETTY VISUALIZATION BLOCK ===
            if save_dir and hit >= 1:
                pretty_visualize_grounding(
                    heat, 
                    orig_img, 
                    box_list, 
                    disease_name_mapping(cls_name), 
                    hit, 
                    x_peak, 
                    y_peak, 
                    save_dir, 
                    i, 
                    fname_suffix=visual_fname_suffix,
                    draw_box_on_overlay=draw_box_on_overlay,
                    box_thickness=4
                )
            if save_dir and hit < 1:
                pretty_visualize_grounding(
                    heat, 
                    orig_img, 
                    box_list, 
                    disease_name_mapping(cls_name), 
                    hit, 
                    x_peak, 
                    y_peak, 
                    save_dir+'_fail_cases', 
                    i, 
                    fname_suffix=visual_fname_suffix,
                    draw_box_on_overlay=draw_box_on_overlay,
                    box_thickness=4
                )
        if count_q > 0:
            results.setdefault(cls_name, {})
            results[cls_name]["pointing_game"] = hits_q / count_q

    return results

def pretty_visualize_grounding(
    heat,
    orig_img,
    box_list,
    cls_name,
    hit,
    x_peak,
    y_peak,
    save_dir,
    i,
    fname_suffix='',
    draw_box_on_overlay=True,
    use_gradcam_style_overlay=False,
    # ---- Visualization controls ----
    thickness_ratio=0.005,
    alpha_strength=0.60,
    blur_ksize=31,
    clip_percentiles=(2, 98),
    box_color=(0, 160, 0),
    box_thickness=6,
    box_shadow=False,
    draw_peak=True,
    peak_color_hit=(0, 255, 255),
    peak_color_miss=(255, 255, 0), # BGR
):
    """
    JET-style heatmap with NO blue background.
    Only strong activations are overlaid.
    """
    # -------------------------
    # 1) heat -> numpy
    # -------------------------
    if isinstance(heat, np.ndarray):
        heat_np = heat.astype(np.float32, copy=True)
    else:
        heat_np = heat.detach().cpu().numpy().astype(np.float32)

    # -------------------------
    # 2) orig_img -> BGR
    # -------------------------
    if orig_img.ndim == 2:
        orig_bgr = cv2.cvtColor(orig_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        orig_bgr = orig_img.astype(np.uint8)

    H, W = orig_bgr.shape[:2]

    if heat_np.shape[:2] != (H, W):
        heat_np = cv2.resize(heat_np, (W, H), cv2.INTER_LINEAR)

    # -------------------------
    # 3) Robust normalization
    # -------------------------
    lo, hi = np.percentile(heat_np, clip_percentiles)
    heat_clip = np.clip(heat_np, lo, hi)

    denom = heat_clip.max() - heat_clip.min()
    if denom < 1e-8:
        heat01 = np.zeros_like(heat_clip)
    else:
        heat01 = (heat_clip - heat_clip.min()) / denom

    heat_u8 = (heat01 * 255).astype(np.uint8)

    # -------------------------
    # 4) Smooth blobs
    # -------------------------
    if blur_ksize and blur_ksize > 0:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        heat_u8 = cv2.GaussianBlur(heat_u8, (blur_ksize, blur_ksize), 0)

    # -------------------------
    # 5) Kill weak activations (removes blue background)
    # -------------------------
    # Anything below ~20% is treated as background
    if not use_gradcam_style_overlay:
        thr = int(0.20 * heat_u8.max()) # NOTE
        heat_u8[heat_u8 < thr] = 0 # NOTE

    # -------------------------
    # 6) JET colormap
    # -------------------------
    if not use_gradcam_style_overlay:
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET) #NOTE
    else:
        ## sesmic color map alternative
        heat_norm = heat_u8.astype(np.float32) / 255.0
        cmap = plt.get_cmap('jet')
        heat_color_rgba = cmap(heat_norm)
        heat_color = (heat_color_rgba[:, :, :3][:, :, ::-1] * 255).astype(np.uint8)

    # -------------------------
    # 7) Heat-based alpha (KEY PART)
    # -------------------------
    if not use_gradcam_style_overlay:
        alpha = heat_u8.astype(np.float32) / 255.0
    else:
        alpha_map = np.full((H, W, 1), alpha_strength, dtype=np.float32) # NOTE

    # Sharpen hotspot and Remove faint regions
    if not use_gradcam_style_overlay:
        alpha = np.power(alpha, 2.0) # NOTE
        alpha[alpha < 0.15] = 0.0 # NOTE
        alpha = alpha * alpha_strength # NOTE
        alpha = alpha[..., None] # NOTE

    # -------------------------
    # 8) Blend (only where alpha > 0)
    # -------------------------
    if not use_gradcam_style_overlay:
        overlay = (
            heat_color.astype(np.float32) * alpha +
            orig_bgr.astype(np.float32) * (1.0 - alpha)
        ).astype(np.uint8) # NOTE
    else:
        overlay = (
            heat_color.astype(np.float32) * alpha_map +
            orig_bgr.astype(np.float32) * (1.0 - alpha_map)
        ).astype(np.uint8)

    # -------------------------
    # 9) Draw boxes
    # -------------------------
    if box_list is None:
        box_list = []

    for box in box_list:
        x1, y1, x2, y2 = map(int, box)

        if draw_box_on_overlay and box_shadow:
            cv2.rectangle(
                overlay,
                (x1, y1), (x2, y2),
                (0, 0, 0),
                box_thickness + 2
            )

        if draw_box_on_overlay:
            cv2.rectangle(
                overlay,
                (x1, y1), (x2, y2),
                box_color,
                box_thickness
            )

        # ALWAYS draw the box in the original image

        if box_shadow:
            cv2.rectangle(
                orig_bgr,
                (x1, y1), (x2, y2),
                (0, 0, 0),
                box_thickness + 2
            )

        cv2.rectangle(
            orig_bgr,
            (x1, y1), (x2, y2),
            box_color,
            box_thickness
        )
    # -------------------------
    # 10) Peak point
    # -------------------------
    if draw_peak and x_peak is not None and y_peak is not None:
        peak_color = peak_color_hit if hit else peak_color_miss

        cv2.circle(
            overlay,
            (int(x_peak), int(y_peak)),
            7,
            (255, 255, 255),
            -1
        )

        cv2.circle(
            overlay,
            (int(x_peak), int(y_peak)),
            4,
            peak_color,
            -1
        )

    # -------------------------
    # 11) Save
    # -------------------------

    # all the image variant saved under the same directory
    instance_dir = os.path.join(save_dir, f'{cls_name}_{i}')
    os.makedirs(instance_dir, exist_ok=True)

    # save the raw overlay image
    fname = f"pretty_overlay_{cls_name}_{i}.jpg"
    if len(fname_suffix) > 0:
        fname = fname.replace('.jpg', f'_{fname_suffix}.jpg')
    out_path = os.path.join(instance_dir, fname)
    cv2.imwrite(out_path, overlay)

    # save text augmented overlay images
    overlay_text = draw_adaptive_title_safe(
        overlay,
        cls_name,
        align="left",
        text_color=box_color,
        thickness_ratio=thickness_ratio
    ) 
    fname = f"pretty_annotated_overlay_{cls_name}_{i}.jpg"
    if len(fname_suffix) > 0:
        fname = fname.replace('.jpg', f'_{fname_suffix}.jpg')
    out_path = os.path.join(instance_dir, fname)
    cv2.imwrite(out_path, overlay_text)

    # save the raw original image with box (no suffix needed)
    fname = f"pretty_origin_{cls_name}_{i}.jpg"
    out_path = os.path.join(instance_dir, fname)
    cv2.imwrite(out_path, orig_bgr)

    # save the text augmented original image with box (no suffix needed)
    origin_img_text = draw_adaptive_title_safe( 
        orig_bgr,
        cls_name,
        align="left",
        text_color=box_color,
        thickness_ratio=thickness_ratio
    )
    fname = f"pretty_annotated_origin_{cls_name}_{i}.jpg"
    out_path = os.path.join(instance_dir, fname)
    cv2.imwrite(out_path, origin_img_text)
    return 

def decide_best_model_overall(per_dataset_eval, task_name='zeroshot_binary', metric_of_interest='AUROC(Avg)'):
    """
    gather auroc stats across evaluation dataset and get the average number for each checkpoint
    then output the best model.
    """
    per_ckpt_stats = {} # key is the ckpt path value is a list of AUROC(Avg) (and other metrics like pointing game if enabled)
    winners = []         # list of per-dataset winning ckpts (one entry per dataset)

    for ckpts in per_dataset_eval:
        # evals: dict[ckpt_path] -> stats
        per_dataset_classification_scores = {}
        for ckpt_instance in ckpts: # ckpt_instance = /cluster/projects/mcintoshgroup/CXR-CLIP/hydra_outputs/2026-01-16/18-50-53/checkpoints/model-5.tar
            stats = ckpts[ckpt_instance]
            if task_name in stats:
                cls_scores = stats[task_name][metric_of_interest]
                per_ckpt_stats.setdefault(ckpt_instance, []).append(cls_scores)
                per_dataset_classification_scores[ckpt_instance] = cls_scores
        # majority-win: choose the best ckpt for this dataset (if any scores exist)
        if per_dataset_classification_scores:
            dataset_winner = max(per_dataset_classification_scores, key=per_dataset_classification_scores.get)
            winners.append(dataset_winner)

    # (A) overall average win
    key_means = {k: mean(v) for k, v in per_ckpt_stats.items()}
    log.info(f'per checkpoint overall score in {task_name} with metric {metric_of_interest}:')

    if len(key_means) > 0:
        for k, m in sorted(key_means.items(), key=lambda x: x[1], reverse=True):
            log.info(f"    {k}: mean={m:.6f}")
        best_key = max(key_means, key=key_means.get)
        log.info(f'best checkpoint overall: {best_key} with overall score of {key_means[best_key]}')

    # (B) Majority-win best (wins across datasets)
    win_counts = Counter(winners)
    log.info(f"per checkpoint majority wins (#datasets won) in {task_name} with metric {metric_of_interest}:")
    for k, c in win_counts.most_common():
        log.info(f"    {k}: wins={c}")
    if not win_counts:
        log.info("no majority-winner could be computed (no per-dataset scores found).")
        return

    # tie-break: (1) more wins, then (2) higher overall mean, then (3) ckpt path stable sort
    max_wins = max(win_counts.values())
    tied = [k for k, c in win_counts.items() if c == max_wins]
    best_key_majority = max(tied, key=lambda k: (key_means.get(k, float("-inf")), k))

    log.info(
        f"best checkpoint majority-win: {best_key_majority} "
        f"with wins={win_counts[best_key_majority]} and mean={key_means[best_key_majority]:.6f}\n"
    )
    return key_means

def average_and_select_best_across_tasks(dict_list):
    if not dict_list:
        raise ValueError("empty")
    keys = dict_list[0].keys()
    if any(d.keys() != keys for d in dict_list):
        raise ValueError("key mismatch")
    avg = {k: sum(d[k] for d in dict_list) / len(dict_list) for k in keys}

    # Sort by performance (high → low)
    sorted_avg = sorted(
        avg.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Log all checkpoints
    log.info("Checkpoint ranking (high → low):")
    for rank, (ckpt, score) in enumerate(sorted_avg, 1):
        log.info(f"{rank:02d}. {ckpt}: {score:.6f}")

    # Best model
    best, best_score = sorted_avg[0]
    log.info(
        f"Best model across the tasks is {best} "
        f"with average performance of {best_score:.6f}"
    )

def text_guided_similarities(image_embeddings, image_patch_embeddings, text_embeddings):

    similarities = []
    for i, text in enumerate(text_embeddings):
        enriched_image_embedddings_for_i = flair_attention_numpy(
            text[None, None, :], 
            np.concatenate([image_embeddings[:, None, :], image_patch_embeddings], axis=1), # combine the cls with patch tokens
            None
        ) # => in shape [3000x512]

        similarity_scores = enriched_image_embedddings_for_i @ text.reshape(-1, 1)
        # similarity_scores = metrics.pairwise.cosine_similarity(enriched_image_embedddings_for_i, text.reshape(1, -1))
        similarities.append(similarity_scores)

    similarities = np.concatenate(similarities, axis=-1)
    return similarities

def multilabel_classification(preds: np.ndarray, labels: np.ndarray, class_list: list):
    log.info("evaluate multi-label classification")

    result = {}
    for idx, class_name in enumerate(class_list):
        result[class_name] = {}
        fpr, tpr, thresholds = metrics.roc_curve(labels[:, idx], preds[:, idx])
        result[class_name]["AUROC"] = metrics.auc(fpr, tpr)
        result[class_name]["PR_AUROC"] = metrics.average_precision_score(labels[:, idx], preds[:, idx])

        result[class_name]["Accuracy"] = metrics.accuracy_score(labels[:, idx], preds[:, idx] > 0.5)
        result[class_name]["F1"] = metrics.f1_score(labels[:, idx], preds[:, idx] > 0.5)

    return classification_score(result)

def pointing_game_score(result: dict, _print=True):
    pg = np.mean([value["pointing_game"] for value in result.values()])
    result["Pointing_Game(Avg)"] = pg
    if _print:
        s = "\n".join(f"{k}: {v}" for k, v in result.items())
        log.info(s)
    return result

def classification_score(result: dict, class_counts: dict = {}, _print=True):

    # macro average
    auroc = np.mean([value["AUROC"] for value in result.values()])
    # f1 = np.mean([value["F1"] for value in result.values()])
    acc = np.mean([value["Accuracy"] for value in result.values()])
    pr_auc = np.mean([value["PR_AUROC"] for value in result.values()])

    result["AUROC(Avg)"] = auroc
    result["PR_AUROC(Avg)"] = pr_auc
    result["Accuracy(Avg)"] = acc
    # ---------- weighted average ----------
    if class_counts:
        weights = np.array([class_counts[k] for k in result.keys() if k in class_counts])
        weights = weights / weights.sum()

        def weighted_metric(metric_name):
            vals = []
            wts = []
            for k, v in result.items():
                if k in class_counts and metric_name in v:
                    vals.append(v[metric_name])
                    wts.append(class_counts[k])
            if not vals:
                return None
            wts = np.array(wts) / np.sum(wts)
            return np.sum(np.array(vals) * wts)

        w_auroc = weighted_metric("AUROC")
        w_pr_auc = weighted_metric("PR_AUROC")
        w_acc = weighted_metric("Accuracy")
        # w_f1 = weighted_metric("F1")
        # w_precision = weighted_metric("Precision")
        # w_recall = weighted_metric("Recall")

        if w_auroc is not None:
            result["AUROC(Weighted)"] = w_auroc
        if w_pr_auc is not None:
            result["PR_AUROC(Weighted)"] = w_pr_auc
        if w_acc is not None:
            result["Accuracy(Weighted)"] = w_acc

    if _print:
        s = "\n".join(f"{k}: {v}" for k, v in result.items())
        log.info(s)

    return result

def multiclass_classification(preds: np.ndarray, labels: np.ndarray, class_list: list):
    log.info("evaluate multi-class classification")
    preds_args = np.argmax(preds, axis=1)

    class_dict = {class_name: {"total_num": 0, "correct_num": 0} for class_name in class_list}
    for idx, class_name in enumerate(class_list):
        class_dict[class_name]["total_num"] = labels[:, idx].sum()
        class_dict[class_name]["correct_num"] = (labels[:, idx] * (preds_args == idx)).sum()

    total_num = len(labels)
    correct_num = sum([v["correct_num"] for v in class_dict.values()])

    result = {k: v["correct_num"] / v["total_num"] for k, v in class_dict.items()}
    result["Accuracy(Macro)"] = np.mean(list(result.values()))
    result["Accuracy(Micro)"] = correct_num / total_num  # same with macro due to same total_num
    s = " / ".join([f"{c}: {v:.3f}" for c, v in result.items()])
    log.info(s)

    return result

def cosine_retrieval(
    image_embeddings: np.ndarray, 
    text_embeddings: np.ndarray, 
    text_list: list = [], 
    mode: str = "i2t"):
    """
    Compute retrieval metrics between images and texts. (a general version of retrieval_image_text above.)

    Args:
        image_embeddings (np.ndarray): Embeddings of images, shape (N, D)
        text_embeddings (np.ndarray): Embeddings of texts, shape (M, D)
        text_list (list): Original list of texts (to identify identicals)
        mode (str): 'i2t' for image-to-text or 't2i' for text-to-image retrieval

    Returns:
        dict: Recall@1, Recall@5, Recall@10, MeanRank
    """
    assert mode in ["i2t", "t2i"], "Mode must be 'i2t' or 't2i'"

    log.info(f"Evaluate consine retrieval: mode = {mode}")

    identical_text_set = []
    imgIdx2txt = {}
    identical_indexes = []

    for i, text in enumerate(text_list):
        if text not in identical_text_set:
            identical_text_set.append(text)
            identical_indexes.append(i)
            imgIdx2txt[i] = len(identical_text_set) - 1
        else:
            imgIdx2txt[i] = identical_text_set.index(text)

    identical_text_embedding = text_embeddings[identical_indexes]
    n_images = image_embeddings.shape[0] # each image is unique

    if mode == "i2t":
        n_texts = len(identical_text_set)
        similarities = metrics.pairwise.cosine_similarity(
            image_embeddings, 
            identical_text_embedding
        )  # [n_images, n_texts]
        outer_loop = range(n_images)
        get_target = lambda idx: imgIdx2txt[idx]
        total = n_images
        search_space = n_texts
    else:  # t2i
        n_texts = len(text_embeddings)
        similarities = metrics.pairwise.cosine_similarity(
            text_embeddings, 
            image_embeddings
        )  # [n_texts, n_images]

        assert similarities.shape[0] == similarities.shape[1]
        outer_loop = range(n_texts)
        get_target = lambda idx: idx
        total = n_texts
        search_space = n_images

    return get_recall_ranking(outer_loop, get_target, similarities, search_space, total)

def get_recall_ranking(outer_loop, get_target, similarities, search_space, total):

    recall_dict = {1: 0, 5: 0, 10: 0}
    mean_rank = 0

    for idx in outer_loop:
        target = get_target(idx)
        similarity = similarities[idx]
        similarity_args = similarity.argsort()  # ascending
        rank = search_space - np.argwhere(similarity_args == target).ravel()[0]

        mean_rank += rank
        for k in recall_dict:
            if rank <= k:
                recall_dict[k] += 1

    log.info(
        "\n".join([f"Recall@{k}: {v / total:.3f}" for k, v in recall_dict.items()]) + f"\nMean Rank: {mean_rank / total:.3f}"
    )

    result = {f"Recall@{k}": v / total for k, v in recall_dict.items()}
    result["MeanRank"] = mean_rank / total
    return result

def cosine_retrieval_feasible(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    text_list: list = [],
    mode: str = "i2t",
    block_size: int = 256,
):
    assert mode in ["i2t", "t2i"]

    # build identical/unique texts exactly like your code
    identical_text_set = []
    imgIdx2txt = {}
    identical_indexes = []

    for i, text in enumerate(text_list):
        if text not in identical_text_set:
            identical_text_set.append(text)
            identical_indexes.append(i)
            imgIdx2txt[i] = len(identical_text_set) - 1
        else:
            imgIdx2txt[i] = identical_text_set.index(text)

    identical_text_embedding = text_embeddings[identical_indexes]
    n_images = image_embeddings.shape[0]

    if mode == "i2t":
        n_texts = len(identical_text_set)
        outer_loop = range(n_images)
        get_target = lambda idx: imgIdx2txt[idx]
        total = n_images
        search_space = n_texts

        Q = image_embeddings
        C = identical_text_embedding

    else:  # t2i
        n_texts = len(text_embeddings)
        # you had this assumption:
        assert n_texts == n_images, "t2i here assumes #texts == #images and index-aligned pairs"

        outer_loop = range(n_texts)
        get_target = lambda idx: idx
        total = n_texts
        search_space = n_images

        Q = text_embeddings
        C = image_embeddings

    return get_recall_ranking_streaming(
        outer_loop=outer_loop,
        get_target=get_target,
        query_embeddings=Q,
        candidate_embeddings=C,
        search_space=search_space,
        total=total,
        block_size=block_size,
        tie_break="strict",
    )

def get_recall_ranking_streaming(
    outer_loop,
    get_target,
    query_embeddings: np.ndarray,      # Q: shape (Nq, D)
    candidate_embeddings: np.ndarray,  # C: shape (Nc, D)
    search_space: int,
    total: int,
    block_size: int = 256,
    tie_break: str = "strict",  # "strict" uses > ; "inclusive" uses >=
):
    """
    Exact Recall@K and MeanRank without allocating similarities and without sorting.

    Matches your rank definition:
      rank = search_space - position_in_ascending_sort(target)
    which is equivalent to:
      rank = 1 + count(scores > score_target)  (strict)

    If you want pessimistic handling of ties, use tie_break="inclusive".
    """
    recall_dict = {1: 0, 5: 0, 10: 0}
    mean_rank = 0.0

    C = candidate_embeddings
    assert C.shape[0] == search_space, "search_space must equal #candidates"

    for idx in outer_loop:
        target = int(get_target(idx))
        q = query_embeddings[idx]                 # (D,)

        s_target = float(q @ C[target])

        greater = 0
        if tie_break == "strict":
            for start in range(0, search_space, block_size):
                end = min(start + block_size, search_space)
                scores = C[start:end] @ q         # (block,)
                greater += int(np.sum(scores > s_target))
            rank = 1 + greater

        elif tie_break == "inclusive":
            for start in range(0, search_space, block_size):
                end = min(start + block_size, search_space)
                scores = C[start:end] @ q
                greater += int(np.sum(scores >= s_target))
            # subtract the target itself (it is always >= itself)
            rank = 1 + max(greater - 1, 0)
        else:
            raise ValueError("tie_break must be 'strict' or 'inclusive'")

        mean_rank += rank
        for k in recall_dict:
            if rank <= k:
                recall_dict[k] += 1

    # keep your logging behavior if you want
    # log.info("\n".join([f"Recall@{k}: {v / total:.3f}" for k, v in recall_dict.items()]) +
    #          f"\nMean Rank: {mean_rank / total:.3f}")

    result = {f"Recall@{k}": v / total for k, v in recall_dict.items()}
    result["MeanRank"] = mean_rank / total
    return result

def disease_list_mapping(names):    
    return [disease_name_mapping(name) for name in names]

def disease_name_mapping(name):
    if name.lower() == 'ILD'.lower() or 'interstitial lung disease' in name.lower():
        return 'interstitial lung disease'
    if name.lower() == 'Enlarged PA'.lower():
        return 'enlarged pulmonary artery'
    if name.lower() == 'COPD'.lower():
        return 'chronic obstructive pulmonary disease'
    if name.lower() == 'Pleural_Thickening'.lower():
        return 'pleural thickening'
    if name.lower() == 'No Finding'.lower() or name.lower() == 'Normal'.lower() or name.lower() == 'No Findings'.lower() or 'no ' in name.lower():
        return 'no findings'
    if name.lower() == 'Fracture'.lower():
        return 'rib fracture'
    if name.lower() == 'Nodule'.lower():
        return 'lung nodule'
    if name.lower() == 'Mass'.lower():
        return 'lung mass'
    if name.lower() == 'Effusion'.lower():
        return 'pleural effusion'
    if name.lower() == 'Fibrosis'.lower():
        return 'pulmonary fibrosis'
    if name.lower() == 'cpam'.lower():
        return 'congenital pulmonary airway malformation'
    if name.lower() == 'COPD'.lower():
        return 'chronic obstructive pulmonary disease'
    if name.lower() == 'tb':
        return 'tuberculosis'
    if name.lower() == 'fracture old' or name.lower() == 'fracture_old':
        return 'fracture'
    if name.lower() == 'nodule/mass':
        return 'nodule or mass'
    covid_des = 'ground-glass opacities, consolidation, pleural thickening commonly appear in infection'
    if name.lower() == 'covid'.lower():
        return covid_des
    return name.lower()

def baseline_model_gather_statistics(label_names, class_list, perclass_predictions):
    if len(label_names) > 0 and type(label_names[0]) is not list:
        label_names = [[label] for label in label_names]

    # gather statistics, auroc, prauc, and acc
    zs_results, class_counts = {}, {}
    for i, class_name in enumerate(class_list):
        class_preds = perclass_predictions[class_name]
        y_true = [1 if disease_name_mapping(class_name) in disease_list_mapping(row_labels) else 0 for row_labels in label_names] # flatten the internal list
        assert len(y_true) == len(class_preds), f'label size does not match the number of image instances. {len(y_true)}, {len(class_preds)} for {class_name}'
        zs_results[class_name] = {}
        fpr, tpr, thresholds = metrics.roc_curve(y_true, class_preds) # get the positive class probability
        zs_results[class_name]["AUROC"] = metrics.auc(fpr, tpr) # micro AUC for each class then average it to get MACRO auc.
        zs_results[class_name]["PR_AUROC"] = metrics.average_precision_score(y_true, class_preds)

        class_preds = np.array(class_preds)
        zs_results[class_name]["Accuracy"] = metrics.accuracy_score(y_true, (class_preds > 0.5).astype(int))
        zs_results[class_name]["F1"] = metrics.f1_score(y_true, (class_preds > 0.5).astype(int))
        y_pred_bin = (class_preds > 0.5).astype(int)
        zs_results[class_name]["Precision"] = metrics.precision_score(y_true, y_pred_bin, zero_division=0)
        zs_results[class_name]["Recall"] = metrics.recall_score(y_true, y_pred_bin, zero_division=0)
        class_counts[class_name] = sum(y_true) # for weighted average
    
    summarised_results = {}
    summarised_results['zeroshot_binary'] = classification_score(zs_results, class_counts)
    return summarised_results