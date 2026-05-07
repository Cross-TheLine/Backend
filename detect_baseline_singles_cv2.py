from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2

from court_geometry import LineCluster, serialize_line
from hough_lines import cluster_segments, hough_segments
from line_mask import line_region_mask, make_white_mask
from line_selection import classify_lines, classify_near_three_lines
from roboflow_roi import (
    apply_bbox_roi,
    bbox_from_prediction,
    roboflow_detect,
    select_court_prediction,
)
from visualize import (
    draw_all_clusters,
    draw_colored_region_overlay,
    draw_region_overlay,
    draw_roi_box,
    draw_selected,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect the near-side tennis court lines used for camera-side in/out checks."
    )
    parser.add_argument("--input", required=True, help="Path to the input image.")
    parser.add_argument("--out-dir", default="line_detection_output", help="Directory for result images and JSON.")
    parser.add_argument("--save-debug", action="store_true", help="Save intermediate masks and line-cluster images.")
    parser.add_argument("--use-roboflow-roi", action="store_true")
    parser.add_argument("--roboflow-model", default="tennis-vhrs9/9", help="Roboflow model id.")
    parser.add_argument("--roboflow-api-key", default=None, help="Roboflow API key. Prefer ROBOFLOW_API_KEY env var.")
    parser.add_argument("--roboflow-api-url", default="https://serverless.roboflow.com")
    parser.add_argument("--roboflow-client", choices=["sdk", "legacy"], default="sdk")
    parser.add_argument("--roboflow-workspace", default=None)
    parser.add_argument("--roboflow-workflow-id", default=None)
    parser.add_argument("--roboflow-workflow-image-input", default="image")
    parser.add_argument("--roboflow-margin", type=int, default=30)
    parser.add_argument("--roboflow-confidence", type=int, default=25)
    parser.add_argument(
        "--line-region-grow-px",
        type=float,
        default=0.0,
        help="Pixel radius used to recover the full visible line region around each Hough line. 0 chooses automatically.",
    )
    args = parser.parse_args()

    if args.use_roboflow_roi and not args.roboflow_model and not args.roboflow_workflow_id:
        parser.error("--use-roboflow-roi requires --roboflow-model or --roboflow-workflow-id.")
    return args


def resolve_out_dir(out_dir_arg: str) -> Path:
    out_dir = Path(out_dir_arg)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def apply_optional_roboflow_roi(
    input_path: Path,
    image_shape: tuple[int, int, int],
    mask,
    args: argparse.Namespace,
) -> tuple[object, tuple[int, int, int, int] | None, str | None, object]:
    if not args.use_roboflow_roi:
        return mask, None, None, None

    api_key = args.roboflow_api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("Roboflow ROI를 쓰려면 --roboflow-api-key 또는 ROBOFLOW_API_KEY 환경변수가 필요합니다.")

    roboflow_result = roboflow_detect(
        input_path,
        model_id=args.roboflow_model,
        api_key=api_key,
        api_url=args.roboflow_api_url,
        client_mode=args.roboflow_client,
        workspace_name=args.roboflow_workspace,
        workflow_id=args.roboflow_workflow_id,
        workflow_image_input=args.roboflow_workflow_image_input,
        confidence=args.roboflow_confidence,
    )
    court_prediction = select_court_prediction(roboflow_result)
    if court_prediction is None:
        warning = "Roboflow 결과에서 class='court' bbox를 찾지 못해 ROI 없이 전체 이미지로 진행했습니다."
        return mask, None, warning, roboflow_result

    court_bbox = bbox_from_prediction(court_prediction, image_shape, args.roboflow_margin)
    return apply_bbox_roi(mask, court_bbox), court_bbox, None, roboflow_result


def try_classify_near_three(
    clusters: list[LineCluster], image_shape: tuple[int, int, int]
) -> tuple[dict[str, LineCluster], str | None]:
    try:
        return classify_near_three_lines(clusters, image_shape), None
    except RuntimeError as exc:
        return {}, str(exc)


def try_classify_full_court(
    clusters: list[LineCluster], image_shape: tuple[int, int, int]
) -> tuple[dict[str, LineCluster], str | None]:
    try:
        return classify_lines(clusters, image_shape), None
    except RuntimeError as exc:
        return {}, str(exc)


def save_outputs(
    out_dir: Path,
    image,
    mask,
    court_bbox: tuple[int, int, int, int] | None,
    clusters: list[LineCluster],
    near_three: dict[str, LineCluster],
    selected: dict[str, LineCluster],
    line_region_grow_px: float,
    save_debug: bool,
) -> None:
    if near_three:
        cv2.imwrite(str(out_dir / "result_lines.png"), draw_selected(image, near_three))
        cv2.imwrite(str(out_dir / "near_three_lines.png"), draw_selected(image, near_three))
        cv2.imwrite(
            str(out_dir / "near_three_line_regions.png"),
            draw_colored_region_overlay(image, mask, near_three, line_region_grow_px),
        )
    elif selected:
        cv2.imwrite(str(out_dir / "result_lines.png"), draw_selected(image, selected))
        cv2.imwrite(str(out_dir / "full_court_lines.png"), draw_selected(image, selected))
    else:
        cv2.imwrite(str(out_dir / "result_lines.png"), draw_all_clusters(image, clusters))

    if not save_debug:
        return

    all_region_mask = line_region_mask(mask, clusters, line_region_grow_px)
    cv2.imwrite(str(out_dir / "debug_00_roboflow_roi.png"), draw_roi_box(image, court_bbox))
    cv2.imwrite(str(out_dir / "debug_01_white_mask.png"), mask)
    cv2.imwrite(str(out_dir / "debug_02_all_line_clusters.png"), draw_all_clusters(image, clusters))
    cv2.imwrite(str(out_dir / "debug_03_line_region_candidates.png"), draw_region_overlay(image, all_region_mask))

    if selected:
        cv2.imwrite(str(out_dir / "debug_04_baseline_and_single_lines.png"), draw_selected(image, selected))
        selected_region_mask = line_region_mask(mask, list(selected.values()), line_region_grow_px)
        cv2.imwrite(str(out_dir / "debug_05_selected_line_region_mask.png"), selected_region_mask)
        cv2.imwrite(
            str(out_dir / "debug_06_selected_line_region_overlay.png"),
            draw_region_overlay(image, selected_region_mask),
        )


def build_result_json(
    input_path: Path,
    image_shape: tuple[int, int, int],
    segments: list[tuple[int, int, int, int]],
    clusters: list[LineCluster],
    args: argparse.Namespace,
    court_bbox: tuple[int, int, int, int] | None,
    roboflow_warning: str | None,
    roboflow_result,
    line_region_grow_px: float,
    near_three: dict[str, LineCluster],
    near_three_error: str | None,
    selected: dict[str, LineCluster],
    selected_error: str | None,
) -> dict:
    return {
        "input": input_path.name,
        "image_size": [int(image_shape[1]), int(image_shape[0])],
        "raw_hough_segments": len(segments),
        "line_clusters": len(clusters),
        "roboflow_model": args.roboflow_model if args.use_roboflow_roi else None,
        "roboflow_api_url": args.roboflow_api_url if args.use_roboflow_roi else None,
        "roboflow_client": args.roboflow_client if args.use_roboflow_roi else None,
        "roboflow_workspace": args.roboflow_workspace if args.use_roboflow_roi else None,
        "roboflow_workflow_id": args.roboflow_workflow_id if args.use_roboflow_roi else None,
        "roboflow_court_bbox": list(court_bbox) if court_bbox is not None else None,
        "roboflow_roi_applied": court_bbox is not None,
        "roboflow_warning": roboflow_warning,
        "roboflow_predictions": roboflow_result.get("predictions", []) if roboflow_result else None,
        "line_region_grow_px": round(float(line_region_grow_px), 3),
        "near_three_error": near_three_error,
        "near_three_selected": {name: serialize_line(line) for name, line in near_three.items()},
        "full_court_error": selected_error,
        "full_court_selected": {name: serialize_line(line) for name, line in selected.items()},
        "clusters": [serialize_line(line) for line in clusters],
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = resolve_out_dir(args.out_dir)

    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(input_path)

    mask = make_white_mask(image)
    mask, court_bbox, roboflow_warning, roboflow_result = apply_optional_roboflow_roi(
        input_path, image.shape, mask, args
    )

    segments = hough_segments(mask)
    clusters = cluster_segments(segments, mask)
    line_region_grow_px = (
        args.line_region_grow_px
        if args.line_region_grow_px > 0
        else max(16.0, min(image.shape[:2]) * 0.015)
    )

    near_three, near_three_error = try_classify_near_three(clusters, image.shape)
    selected, selected_error = try_classify_full_court(clusters, image.shape)
    save_outputs(
        out_dir,
        image,
        mask,
        court_bbox,
        clusters,
        near_three,
        selected,
        line_region_grow_px,
        args.save_debug,
    )

    data = build_result_json(
        input_path,
        image.shape,
        segments,
        clusters,
        args,
        court_bbox,
        roboflow_warning,
        roboflow_result,
        line_region_grow_px,
        near_three,
        near_three_error,
        selected,
        selected_error,
    )
    (out_dir / "detected_lines.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved outputs to: {out_dir}")
    if near_three_error:
        print(f"Near-three warning: {near_three_error}")
    if roboflow_warning:
        print(f"Roboflow warning: {roboflow_warning}")


if __name__ == "__main__":
    main()
