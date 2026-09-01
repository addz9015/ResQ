"""
Cyclone satellite image processing pipeline - VISIBLE / TRUE-COLOR variant
(MODIS / VIIRS true-color imagery)

Steps: load -> preprocess -> enhance -> segment -> feature extraction ->
classification/interpretation -> annotated output + report

Usage: python3 pipeline_visible.py <image_path> <json_path> <output_dir>
"""
import sys, os, json, math
import numpy as np
import cv2
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def run(image_path, json_path, out_dir, extra_context=None):
    os.makedirs(out_dir, exist_ok=True)
    report = {"pipeline": "visible_true_color", "steps": []}

    # ---------- STEP 0: LOAD & INSPECT ----------
    log("Step 0: Load & inspect")
    with open(json_path) as f:
        meta = json.load(f)
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    h0, w0 = img_bgr.shape[:2]

    # Resize to a manageable working resolution (keep aspect ratio)
    MAX_DIM = 1400
    scale = min(1.0, MAX_DIM / max(h0, w0))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w0*scale), int(h0*scale)), interpolation=cv2.INTER_AREA)
    h, w = img_bgr.shape[:2]

    step0_path = os.path.join(out_dir, "step0_original.jpg")
    cv2.imwrite(step0_path, img_bgr)
    report["steps"].append({
        "step": 0, "name": "Load & Inspect",
        "output_image": step0_path,
        "findings": {
            "storm_name": meta.get("storm_name"),
            "satellite": meta.get("satellite"), "sensor": meta.get("sensor"),
            "channel_band": meta.get("channel_band"), "image_type": meta.get("image_type"),
            "datetime_utc": meta.get("image_datetime_utc"),
            "original_resolution_px": [w0, h0],
            "working_resolution_px": [w, h],
            "geographic_description": meta.get("geographic_description"),
        }
    })

    # ---------- STEP 1: PREPROCESSING ----------
    log("Step 1: Preprocessing (denoise + white-balance normalize)")
    denoised = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=50, sigmaSpace=50)
    # simple gray-world white balance to normalize color cast
    b, g, r = cv2.split(denoised.astype(np.float32))
    mb, mg, mr = b.mean(), g.mean(), r.mean()
    mgray = (mb + mg + mr) / 3
    b = np.clip(b * (mgray / (mb + 1e-6)), 0, 255)
    g = np.clip(g * (mgray / (mg + 1e-6)), 0, 255)
    r = np.clip(r * (mgray / (mr + 1e-6)), 0, 255)
    preprocessed = cv2.merge([b, g, r]).astype(np.uint8)

    step1_path = os.path.join(out_dir, "step1_preprocessed.jpg")
    cv2.imwrite(step1_path, preprocessed)
    noise_reduction_pct = 100 * (1 - np.std(cv2.cvtColor(denoised,cv2.COLOR_BGR2GRAY).astype(np.float32) -
                                              cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)) /
                                  (np.std(cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)) + 1e-6))
    report["steps"].append({
        "step": 1, "name": "Preprocessing",
        "output_image": step1_path,
        "operations": ["bilateral denoise (edge-preserving)", "gray-world color normalization"],
        "findings": {"mean_channel_before_BGR": [round(float(x),1) for x in [img_bgr[:,:,0].mean(), img_bgr[:,:,1].mean(), img_bgr[:,:,2].mean()]],
                     "mean_channel_after_BGR": [round(float(mb),1), round(float(mg),1), round(float(mr),1)]}
    })

    # ---------- STEP 2: ENHANCEMENT ----------
    log("Step 2: Enhancement (CLAHE contrast + unsharp mask)")
    lab = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2LAB)
    l, a, bb = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    enhanced_lab = cv2.merge([l2, a, bb])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    # unsharp mask
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
    enhanced = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    step2_path = os.path.join(out_dir, "step2_enhanced.jpg")
    cv2.imwrite(step2_path, enhanced)
    report["steps"].append({
        "step": 2, "name": "Enhancement",
        "output_image": step2_path,
        "operations": ["CLAHE local contrast on L channel (clip=2.0, tiles=8x8)", "unsharp mask sharpening"],
        "findings": {"contrast_std_before": round(float(l.std()),2), "contrast_std_after": round(float(l2.std()),2)}
    })

    # ---------- STEP 3: SEGMENTATION ----------
    log("Step 3: Segmentation (cloud / ocean / land masks)")
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    Hh, Ss, Vv = cv2.split(hsv)

    # Cloud: high brightness, low saturation (white/gray)
    cloud_mask = ((Vv > 150) & (Ss < 70)).astype(np.uint8) * 255
    # Ocean: low-mid brightness, blue hue, moderate saturation
    ocean_mask = ((Hh > 90) & (Hh < 140) & (Vv < 140)).astype(np.uint8) * 255
    # Land/dust: tan/brown hue range
    land_mask = ((Hh > 5) & (Hh < 40) & (Ss > 30)).astype(np.uint8) * 255

    # Clean cloud mask
    kernel = np.ones((5, 5), np.uint8)
    cloud_clean = cv2.morphologyEx(cloud_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cloud_clean = cv2.morphologyEx(cloud_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Connected components -> pick the largest bright blob as candidate storm
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cloud_clean, connectivity=8)
    storm_label = None
    if n_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        storm_label = 1 + int(np.argmax(areas))
    storm_mask = (labels == storm_label).astype(np.uint8) * 255 if storm_label else np.zeros_like(cloud_clean)

    seg_vis = enhanced.copy()
    overlay = np.zeros_like(seg_vis)
    overlay[cloud_mask > 0] = (255, 255, 255)   # white = all cloud
    overlay[ocean_mask > 0] = (140, 60, 0)      # dark blue-ish (BGR) = ocean
    overlay[land_mask > 0] = (40, 120, 180)     # tan/brown (BGR) = land/dust
    seg_vis = cv2.addWeighted(enhanced, 0.55, overlay, 0.45, 0)
    contours_storm, _ = cv2.findContours(storm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(seg_vis, contours_storm, -1, (0, 0, 255), 3)  # red outline = detected storm blob

    step3_path = os.path.join(out_dir, "step3_segmented.jpg")
    cv2.imwrite(step3_path, seg_vis)

    cloud_pct = round(100 * float((cloud_mask > 0).sum()) / (h * w), 2)
    ocean_pct = round(100 * float((ocean_mask > 0).sum()) / (h * w), 2)
    land_pct = round(100 * float((land_mask > 0).sum()) / (h * w), 2)
    storm_area_px = int((storm_mask > 0).sum())

    report["steps"].append({
        "step": 3, "name": "Segmentation",
        "output_image": step3_path,
        "operations": ["HSV thresholding (cloud/ocean/land)", "morphological open+close", "connected-component largest-blob selection"],
        "findings": {
            "cloud_cover_pct_of_frame": cloud_pct,
            "ocean_pct_of_frame": ocean_pct,
            "land_pct_of_frame": land_pct,
            "largest_cloud_blob_area_px": storm_area_px,
            "largest_cloud_blob_pct_of_frame": round(100 * storm_area_px / (h * w), 2)
        }
    })

    # ---------- STEP 4: FEATURE EXTRACTION ----------
    log("Step 4: Feature extraction (shape, size, eye search)")
    feat_vis = enhanced.copy()
    findings4 = {}
    if contours_storm:
        c_raw = max(contours_storm, key=cv2.contourArea)
        area_raw = cv2.contourArea(c_raw)
        perim_raw = cv2.arcLength(c_raw, True)
        equiv_diam_raw_px = 2 * math.sqrt(area_raw / math.pi)

        # "Core" mask: strip thin filaments/outflow streamers via opening with a larger
        # kernel, so shape metrics reflect the dense cloud shield rather than wispy edges.
        core_k = max(9, int(round(0.018 * min(h, w))) | 1)  # odd kernel, ~1.8% of frame
        core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_k, core_k))
        core_mask = cv2.morphologyEx(storm_mask, cv2.MORPH_OPEN, core_kernel)
        core_contours, _ = cv2.findContours(core_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if core_contours:
            c = max(core_contours, key=cv2.contourArea)
        else:
            c = c_raw
        area = cv2.contourArea(c)
        # Smooth the boundary before measuring perimeter/circularity: raw pixel-mask
        # contours are jagged at pixel scale, which artificially inflates perimeter.
        eps = 0.004 * cv2.arcLength(c, True)
        c_smooth = cv2.approxPolyDP(c, eps, True)
        perim = cv2.arcLength(c_smooth, True)
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        equiv_diam_px = 2 * math.sqrt(area / math.pi)
        circularity = 4 * math.pi * area / (perim ** 2) if perim > 0 else 0
        M = cv2.moments(c)
        centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] else (int(cx), int(cy))
        core_equiv_radius = math.sqrt(area / math.pi)

        # Eye search: look for a dark/low-brightness enclosed hole WITHIN the dense core,
        # required to be roughly round, small relative to the core, and near the storm
        # center to avoid flagging cloud-gap noise.
        inner_mask = np.zeros_like(storm_mask)
        cv2.drawContours(inner_mask, [c], -1, 255, -1)

        # Reference center for "is this near the storm center" uses a distance transform
        # (the point deepest inside the blob), not the area centroid -- an asymmetric band
        # or extension can drag the plain centroid well away from the true eye/center.
        dist_xform = cv2.distanceTransform(inner_mask, cv2.DIST_L2, 5)
        _, _, _, dt_loc = cv2.minMaxLoc(dist_xform)
        storm_center = dt_loc  # (x, y)

        cv2.drawContours(feat_vis, [c_raw], -1, (0, 140, 255), 2)   # orange = full cloud shield incl. filaments
        cv2.drawContours(feat_vis, [c], -1, (0, 0, 255), 3)          # red = dense core (filaments stripped)
        cv2.circle(feat_vis, (int(cx), int(cy)), int(radius), (0, 255, 255), 2)
        cv2.drawMarker(feat_vis, centroid, (0, 255, 0), cv2.MARKER_CROSS, 25, 3)
        cv2.drawMarker(feat_vis, storm_center, (255, 255, 0), cv2.MARKER_DIAMOND, 25, 3)

        interior_v = Vv.copy()
        interior_v[inner_mask == 0] = 255  # ignore outside core
        dark_spot = ((interior_v < 120) & (inner_mask > 0)).astype(np.uint8) * 255
        dark_spot = cv2.morphologyEx(dark_spot, cv2.MORPH_OPEN, kernel)
        eye_found = False
        eye_info = None
        n2, lbl2, stats2, cent2 = cv2.connectedComponentsWithStats(dark_spot, connectivity=8)
        if n2 > 1:
            for lbl in range(1, n2):
                cand_area = stats2[lbl, cv2.CC_STAT_AREA]
                ex, ey = cent2[lbl]
                dist_from_centroid = math.hypot(ex - storm_center[0], ey - storm_center[1])
                area_ratio = cand_area / area
                # candidate contour roundness
                cand_mask = (lbl2 == lbl).astype(np.uint8) * 255
                cc, _ = cv2.findContours(cand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cc:
                    continue
                cand_perim = cv2.arcLength(cc[0], True)
                cand_round = 4 * math.pi * cand_area / (cand_perim ** 2) if cand_perim > 0 else 0
                is_central = dist_from_centroid < 0.75 * core_equiv_radius
                is_eye_sized = 0.0003 < area_ratio < 0.06
                is_round = cand_round > 0.45
                if is_central and is_eye_sized and is_round:
                    eye_found = True
                    eye_diam_px = 2 * math.sqrt(cand_area / math.pi)
                    eye_info = {"center_px": [round(float(ex),1), round(float(ey),1)],
                                 "equivalent_diameter_px": round(float(eye_diam_px),1),
                                 "roundness_0to1": round(float(cand_round), 2),
                                 "distance_from_core_centroid_px": round(float(dist_from_centroid), 1)}
                    cv2.circle(feat_vis, (int(ex), int(ey)), int(eye_diam_px/2), (255, 0, 255), 3)
                    cv2.putText(feat_vis, "EYE", (int(ex)-15, int(ey)-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
                    break

        findings4 = {
            "full_cloud_shield_area_px": round(area_raw, 1),
            "full_cloud_shield_equivalent_diameter_px": round(equiv_diam_raw_px, 1),
            "dense_core_area_px": round(area, 1),
            "dense_core_perimeter_px_smoothed": round(perim, 1),
            "dense_core_equivalent_diameter_px": round(equiv_diam_px, 1),
            "min_enclosing_circle_diameter_px": round(radius * 2, 1),
            "dense_core_circularity_0to1": round(circularity, 3),
            "area_centroid_px": centroid,
            "storm_center_px_distance_transform": list(storm_center),
            "eye_detected": eye_found,
            "eye_details": eye_info,
            "note_on_scale": "Metadata does not include a geographic bounding box / pixel-to-km scale, so measurements are reported in pixels (working resolution) rather than converted to km, to avoid false precision.",
            "note_on_shape_metrics": "Orange outline = full cloud shield (includes thin outflow filaments); red outline = dense core after stripping thin filaments via morphological opening, used for circularity/eye search. Yellow diamond = distance-transform storm center (used for eye centrality check); green cross = plain area centroid (can be pulled off-center by asymmetric bands)."
        }
    else:
        findings4 = {"error": "No storm contour detected by segmentation step."}

    step4_path = os.path.join(out_dir, "step4_features.jpg")
    cv2.imwrite(step4_path, feat_vis)
    report["steps"].append({
        "step": 4, "name": "Feature Extraction",
        "output_image": step4_path,
        "operations": ["contour geometry (area, perimeter, circularity)", "min enclosing circle", "centroid", "interior dark-spot eye search"],
        "findings": findings4
    })

    # ---------- STEP 5: CLASSIFICATION / INTERPRETATION ----------
    log("Step 5: Classification / interpretation")
    circularity = findings4.get("dense_core_circularity_0to1", 0)
    eye_found = findings4.get("eye_detected", False)
    if eye_found and circularity > 0.7:
        structure_assessment = "Well-organized system with a detectable eye and near-circular cloud mass — consistent with a mature/intense cyclone."
    elif circularity > 0.6:
        structure_assessment = "Fairly organized, rounded cloud mass but no clear eye detected — consistent with a developing or moderately intense system."
    else:
        structure_assessment = "Irregular / asymmetric cloud shape, no eye detected — consistent with a weaker, developing, or dissipating system."

    classification = {
        "structure_assessment": structure_assessment,
        "circularity_0to1": circularity,
        "eye_detected_by_pipeline": eye_found,
    }
    if extra_context:
        classification["reference_context_from_source_article"] = extra_context

    step5_path = os.path.join(out_dir, "step5_annotated_summary.jpg")
    summary_vis = feat_vis.copy()
    label = f"{meta.get('storm_name','?')} | {meta.get('image_datetime_utc','?')} | circularity={circularity}"
    cv2.rectangle(summary_vis, (0, h-40), (w, h), (0,0,0), -1)
    cv2.putText(summary_vis, label, (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.imwrite(step5_path, summary_vis)

    report["steps"].append({
        "step": 5, "name": "Classification / Interpretation",
        "output_image": step5_path,
        "findings": classification
    })

    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"Done. Report: {report_path}")
    return report

if __name__ == "__main__":
    image_path, json_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    run(image_path, json_path, out_dir)
