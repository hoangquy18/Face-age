#!/usr/bin/env python3
"""
Gradio Web UI: verification, top-K gallery, age, and FAS (face age synthesis).

  conda activate moe
  cd /path/to/MTLFace
  pip install gradio
  python demo_web_ui.py

FR weights (default): weights_1/backbone-* and estimation_network-*.
FAS weights (default): two_task_weights/backbone-* and generator-* (joint training).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
import torch

from mtlface_face_engine import FaceEngine, group_labels, repo_root
from mtlface_fas_engine import FASEngine, discover_fas_iteration

_web_root = repo_root()
if _web_root not in sys.path:
    sys.path.insert(0, _web_root)
from backbone.aifr import BACKBONE_CHOICES


def default_ckpt_dir(root: str) -> str:
    w = os.path.join(root, "weights_1")
    if os.path.isdir(w) and os.path.isfile(os.path.join(w, "backbone-20000")):
        return w
    return os.path.join(root, "output", "save_models")


def resolve_ckpt(base_dir: str, name: str, iteration: int) -> str | None:
    path = os.path.join(base_dir, f"{name}-{iteration}")
    return path if os.path.isfile(path) else None


def collect_gallery_paths(
    list_path: str,
    data_root: str,
    max_images: int,
    seed: int = 0,
) -> list[str]:
    """Reservoir sample up to max_images existing file paths (streaming whole list)."""
    rng = np.random.default_rng(seed)
    reservoir: list[str] = []
    seen = 0
    with open(list_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            rel = parts[1]
            p = rel if os.path.isabs(rel) else os.path.join(data_root, rel)
            if not os.path.isfile(p):
                continue
            seen += 1
            if len(reservoir) < max_images:
                reservoir.append(p)
            else:
                j = int(rng.integers(0, seen))
                if j < max_images:
                    reservoir[j] = p
    return reservoir


def collect_gallery_paths_sequential(
    list_path: str,
    data_root: str,
    max_images: int,
) -> list[str]:
    """First max_images existing paths in file order (fast; biased to early identities)."""
    out: list[str] = []
    with open(list_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if len(out) >= max_images:
                break
            parts = line.split()
            if len(parts) < 2:
                continue
            rel = parts[1]
            p = rel if os.path.isabs(rel) else os.path.join(data_root, rel)
            if os.path.isfile(p):
                out.append(p)
    return out


def build_gallery_matrix(
    engine: FaceEngine,
    paths: list[str],
    batch_size: int = 32,
    progress=None,
) -> tuple[list[str], np.ndarray]:
    from PIL import Image

    valid_paths: list[str] = []
    embs: list[np.ndarray] = []
    n = len(paths)
    for start in range(0, n, batch_size):
        if progress is not None:
            progress((start + batch_size) / max(n, 1), desc=f"Gallery {start}/{n}")
        batch_paths = paths[start : start + batch_size]
        tensors = []
        ok_paths = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(engine.tfm(img))
                ok_paths.append(p)
            except OSError:
                continue
        if not tensors:
            continue
        x = torch.stack(tensors, dim=0).to(engine.device)
        with torch.inference_mode():
            e = engine.backbone(x)
            e = torch.nn.functional.normalize(e, dim=1)
        embs.append(e.float().cpu().numpy())
        valid_paths.extend(ok_paths)
    if not embs:
        return [], np.zeros((0, 512), dtype=np.float32)
    mat = np.concatenate(embs, axis=0).astype(np.float32)
    return valid_paths, mat


def _backbone_cache_tag(backbone_ckpt: str) -> str:
    """Stable tag for cache key: basename + file size (not mtime — mtime changes on copy/touch)."""
    if not backbone_ckpt or not os.path.isfile(backbone_ckpt):
        return "none"
    try:
        sz = os.path.getsize(backbone_ckpt)
    except OSError:
        return "none"
    return f"{os.path.basename(backbone_ckpt)}:{sz}"


def cache_path(
    root: str,
    list_path: str,
    max_images: int,
    backbone_ckpt: str,
    gallery_mode: str,
) -> str:
    list_key = os.path.abspath(list_path)
    tag = _backbone_cache_tag(backbone_ckpt)
    h = hashlib.md5(
        f"{list_key}|{max_images}|{tag}|{gallery_mode}".encode(),
        usedforsecurity=False,
    ).hexdigest()[:12]
    d = os.path.join(root, ".cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"gallery_emb_{h}.npz")


def load_or_build_gallery(
    engine: FaceEngine,
    list_path: str,
    data_root: str,
    max_images: int,
    backbone_ckpt: str,
    use_cache: bool,
    gallery_fast: bool,
    progress=None,
) -> tuple[list[str], np.ndarray, bool, str]:
    """
    Returns (paths, embedding_matrix, loaded_from_cache, cache_file_path).
    """
    root = repo_root()
    gallery_mode = "sequential" if gallery_fast else "reservoir"
    cache_file = cache_path(
        root, list_path, max_images, backbone_ckpt or "", gallery_mode
    )
    if use_cache and os.path.isfile(cache_file):
        z = np.load(cache_file, allow_pickle=True)
        paths = z["paths"].tolist()
        mat = z["emb"].astype(np.float32)
        return paths, mat, True, cache_file

    if gallery_fast:
        paths = collect_gallery_paths_sequential(list_path, data_root, max_images)
    else:
        paths = collect_gallery_paths(list_path, data_root, max_images)
    paths, mat = build_gallery_matrix(engine, paths, progress=progress)
    if use_cache and mat.shape[0] > 0:
        np.savez_compressed(cache_file, emb=mat, paths=np.array(paths, dtype=object))
    return paths, mat, False, cache_file


def format_age(pred) -> str:
    if pred is None:
        return "Age: N/A (load estimation_network checkpoint)"
    return (
        f"Predicted age: **{pred.age_years:.1f}** years\n"
        f"Age group: **{pred.group_label}** (id={pred.group_id}, conf={pred.group_confidence:.2f})"
    )


def make_app(
    engine: FaceEngine,
    gallery_paths: list[str],
    gallery_mat: np.ndarray,
    threshold: float,
    fas_engine: FASEngine | None,
    fas_group_labels: list[str],
):
    import gradio as gr

    def tab_verify(img_a, img_b, thr: float):
        if img_a is None or img_b is None:
            return "Upload both images."
        sim, mode = engine.verify_pair(img_a, img_b)
        pa = engine.predict_age(img_a)
        pb = engine.predict_age(img_b)
        if mode != "ok":
            verdict = "**Verdict:** unreliable (backbone not loaded or random weights)"
        else:
            verdict = "**SAME person**" if sim >= thr else "**DIFFERENT persons**"
        md = (
            f"**Cosine similarity:** `{sim:.4f}`\n\n"
            f"{verdict} (threshold `{thr:.2f}`)\n\n"
            "---\n\n"
            "**Image A**\n\n"
            f"{format_age(pa)}\n\n"
            "**Image B**\n\n"
            f"{format_age(pb)}"
        )
        return md

    def tab_topk(query, k: int):
        if query is None:
            return None, "Upload a query image."
        if gallery_mat.shape[0] == 0:
            return None, "Gallery is empty. Check list file / data_root / max_images."
        k = int(max(1, min(k, gallery_mat.shape[0])))
        q = engine.encode_embedding(query).float().cpu().numpy()
        sims = gallery_mat @ q
        top_idx = np.argsort(-sims)[:k]
        items = []
        captions = []
        for rank, i in enumerate(top_idx, 1):
            p = gallery_paths[int(i)]
            s = float(sims[int(i)])
            items.append(p)
            captions.append(f"#{rank} sim={s:.4f}\n{p}")
        pred = engine.predict_age(query)
        info = format_age(pred).replace("**", "")
        info += f"\n\nTop-{k} paths:\n" + "\n".join(captions)
        gallery_data = list(zip(items, captions))
        return gallery_data, info

    fas_radio_choices = [f"{i}: {lbl}" for i, lbl in enumerate(fas_group_labels)]

    def tab_fas_single(src, group_choice: str, res_scale: float, train_bn: bool):
        if fas_engine is None or not fas_engine.ready:
            return None, (
                "**FAS chưa load.** Đặt checkpoint `backbone-<iter>` và `generator-<iter>` "
                "vào thư mục `two_task_weights/` (hoặc chỉnh `--fas_weights_dir`, `--fas_iter`)."
            )
        if src is None:
            return None, "Upload một ảnh mặt."
        gid = int(str(group_choice).split(":", 1)[0].strip())
        out = fas_engine.synthesize(
            src,
            gid,
            residual_scale=float(res_scale),
            generator_train_mode=bool(train_bn),
        )
        ref_gid = 0 if gid != 0 else fas_engine.age_group - 1
        try:
            pix_diff = fas_engine.mean_abs_pixel_diff_between_groups(
                src,
                gid,
                ref_gid,
                residual_scale=float(res_scale),
                generator_train_mode=bool(train_bn),
            )
        except Exception:
            pix_diff = float("nan")
        md = ""
        return out, md

    def tab_fas_all(src, res_scale: float, train_bn: bool):
        if fas_engine is None or not fas_engine.ready:
            return None, (
                "**FAS chưa load.** Cần `two_task_weights/` với backbone + generator cùng iteration."
            )
        if src is None:
            return None, "Upload một ảnh mặt."
        imgs = fas_engine.synthesize_all_groups(
            src,
            residual_scale=float(res_scale),
            generator_train_mode=bool(train_bn),
        )
        caps = [f"{i}: {fas_group_labels[i]}" for i in range(len(imgs))]
        return list(zip(imgs, caps)), (
            f"All groups với residual_scale={res_scale:.2f}, train_bn={train_bn}."
        )

    with gr.Blocks(title="MTLFace Demo") as demo:
        gr.Markdown(
            "## MTLFace — Verification, Top-K, Age, **FAS synthesis**\n"
            "Upload aligned **112×112** face crops (or similar). "
            "Verification dùng **cosine** trên embedding. Tab FAS cần weight joint **backbone + generator**."
        )
        with gr.Tab("Verify (2 images)"):
            with gr.Row():
                img_a = gr.Image(type="pil", label="Image A")
                img_b = gr.Image(type="pil", label="Image B")
            thr_in = gr.Slider(
                0.0, 1.0, value=threshold, step=0.01, label="Same-person threshold"
            )
            out_md = gr.Markdown()
            btn = gr.Button("Compare")
            btn.click(tab_verify, inputs=[img_a, img_b, thr_in], outputs=[out_md])

        with gr.Tab("Top-K similar (gallery)"):
            gr.Markdown(
                f"Gallery size: **{len(gallery_paths)}** images. "
                "Matches are from the indexed subset (not full CASIA)."
            )
            qimg = gr.Image(type="pil", label="Query face")
            gmax = max(1, len(gallery_paths))
            k_max = min(50, gmax)
            k_slider = gr.Slider(1, k_max, value=min(10, k_max), step=1, label="Top-K")
            g_out = gr.Gallery(label="Top matches", columns=5, height="auto")
            info = gr.Textbox(label="Query age + details", lines=12)
            btn2 = gr.Button("Search")
            btn2.click(tab_topk, inputs=[qimg, k_slider], outputs=[g_out, info])

        with gr.Tab("Face age synthesis (FAS)"):
            if fas_engine is not None and fas_engine.ready:
                gr.Markdown("")
            else:
                gr.Markdown(
                    "⚠️ **FAS tắt hoặc thiếu weight.** Thêm `two_task_weights/backbone-<N>` và "
                    "`two_task_weights/generator-<N>` (cùng `N`), hoặc chạy với "
                    "`--fas_weights_dir` / `--fas_iter`."
                )
            src_fas = gr.Image(type="pil", label="Ảnh nguồn (mặt)")
            fas_res = gr.Slider(
                0.25,
                4.0,
                value=2.0,
                step=0.25,
                label="residual_scale (nhân phần thay đổi; 1.0 = đúng paper)",
            )
            fas_train_bn = gr.Checkbox(
                value=False,
                label="Generator.train() khi infer (BN theo batch 1 ảnh — thử nghiệm, có thể artifact)",
            )
            grp = gr.Radio(
                choices=fas_radio_choices,
                value=fas_radio_choices[min(3, len(fas_radio_choices) - 1)],
                label="Nhóm tuổi đích",
            )
            out_img = gr.Image(type="pil", label="Ảnh sau synthesis")
            out_md_fas = gr.Markdown()
            b_fas = gr.Button("Sinh ảnh (1 nhóm)")
            b_fas.click(
                tab_fas_single,
                inputs=[src_fas, grp, fas_res, fas_train_bn],
                outputs=[out_img, out_md_fas],
            )

            gr.Markdown("---")
            _nc = min(7, max(1, len(fas_group_labels)))
            gal_all = gr.Gallery(
                label="Tất cả nhóm tuổi (0→K−1)", columns=_nc, height="auto"
            )
            info_all = gr.Textbox(label="Ghi chú", lines=2)
            b_all = gr.Button("Sinh cả 7 nhóm tuổi")
            b_all.click(
                tab_fas_all,
                inputs=[src_fas, fas_res, fas_train_bn],
                outputs=[gal_all, info_all],
            )

    return demo


def main():
    try:
        import gradio  # noqa: F401
    except ImportError:
        print("Install Gradio: pip install gradio", file=sys.stderr)
        sys.exit(1)

    root = repo_root()
    parser = argparse.ArgumentParser(description="MTLFace Gradio Web UI")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Gradio public share link")
    parser.add_argument(
        "--backbone_name",
        type=str,
        default="ir50",
        help=f"Registered backbones: {', '.join(BACKBONE_CHOICES)}",
    )
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--age_group", type=int, default=7)
    parser.add_argument("--ckpt_dir", type=str, default="weights_1")
    parser.add_argument(
        "--iter", type=int, default=20000, help="Checkpoint iteration suffix"
    )
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--list_path", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--gallery_max", type=int, default=3000)
    parser.add_argument(
        "--gallery_fast",
        action="store_true",
        help="Take first N existing images in list order (fast startup; not a random subset of CASIA)",
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="Disable gallery embedding cache"
    )
    parser.add_argument(
        "--fas_weights_dir",
        type=str,
        default="two_task_weights",
        help="Thư mục chứa backbone-* và generator-* (joint FR+FAS). Đường dẫn tương đối = trong repo.",
    )
    parser.add_argument(
        "--fas_iter",
        type=int,
        default=None,
        help="Iteration checkpoint FAS; mặc định: tự tìm N lớn nhất có cả backbone-N và generator-N.",
    )
    parser.add_argument(
        "--no_fas",
        action="store_true",
        help="Không load FAS (ẩn chức năng synthesis; tab vẫn hiện hướng dẫn).",
    )
    args = parser.parse_args()

    _ckpt = args.ckpt_dir
    if _ckpt and not os.path.isabs(_ckpt):
        _ckpt = os.path.join(root, _ckpt)
    ckpt_dir = _ckpt or default_ckpt_dir(root)
    backbone_ckpt = resolve_ckpt(ckpt_dir, "backbone", args.iter)
    age_ckpt = resolve_ckpt(ckpt_dir, "estimation_network", args.iter)

    if not backbone_ckpt:
        print(
            f"Warning: backbone checkpoint not found under {ckpt_dir}", file=sys.stderr
        )
    if not age_ckpt:
        print(
            f"Warning: estimation_network checkpoint not found under {ckpt_dir}",
            file=sys.stderr,
        )

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")

    engine = FaceEngine(
        backbone_name=args.backbone_name,
        image_size=args.image_size,
        age_group=args.age_group,
        backbone_ckpt=backbone_ckpt,
        age_ckpt=age_ckpt,
        device=device,
    )

    list_path = os.path.abspath(
        args.list_path or os.path.join(root, "dataset", "casia-webface.txt")
    )
    data_root = args.data_root or root

    print(f"Device: {device}")
    print(f"Backbone: {backbone_ckpt}")
    print(f"Age head: {age_ckpt}")
    gallery_mode = "sequential" if args.gallery_fast else "reservoir"
    _cf = cache_path(
        root, list_path, args.gallery_max, backbone_ckpt or "", gallery_mode
    )
    if not args.no_cache and os.path.isfile(_cf):
        print(
            f"Gallery: loading from cache ({_cf}) — {args.gallery_max} max, mode={gallery_mode}"
        )
    else:
        print(
            f"Gallery: computing embeddings from {list_path} (max {args.gallery_max}, "
            f"mode={gallery_mode}) ..."
        )
        if not args.no_cache:
            print(f"Gallery: cache will be saved to {_cf}")

    def prog(t, desc=""):
        pass

    gallery_paths, gallery_mat, from_cache, cache_file = load_or_build_gallery(
        engine,
        list_path=list_path,
        data_root=data_root,
        max_images=args.gallery_max,
        backbone_ckpt=backbone_ckpt or "",
        use_cache=not args.no_cache,
        gallery_fast=args.gallery_fast,
        progress=prog,
    )
    if from_cache:
        print(f"Gallery ready: {len(gallery_paths)} images (loaded from cache).")
    elif args.no_cache:
        print(f"Gallery ready: {len(gallery_paths)} images (--no_cache, not saved).")
    else:
        print(
            f"Gallery ready: {len(gallery_paths)} images (computed; saved {cache_file})."
        )

    fas_engine: FASEngine | None = None
    if not args.no_fas:
        fas_dir = args.fas_weights_dir
        if not os.path.isabs(fas_dir):
            fas_dir = os.path.join(root, fas_dir)
        fas_iter = args.fas_iter
        if fas_iter is None:
            fas_iter = discover_fas_iteration(fas_dir)
        if fas_iter is not None:
            bb_fas = resolve_ckpt(fas_dir, "backbone", fas_iter)
            gen_fas = resolve_ckpt(fas_dir, "generator", fas_iter)
            if bb_fas and gen_fas:
                fas_engine = FASEngine(
                    backbone_name=args.backbone_name,
                    image_size=args.image_size,
                    age_group=args.age_group,
                    backbone_ckpt=bb_fas,
                    generator_ckpt=gen_fas,
                    device=device,
                )
                print(
                    f"FAS: loaded backbone + generator iter {fas_iter} from {fas_dir}"
                )
            else:
                print(
                    f"FAS: thiếu backbone-{fas_iter} hoặc generator-{fas_iter} trong {fas_dir}",
                    file=sys.stderr,
                )
        else:
            print(
                f"FAS: không tìm thấy cặp backbone-*/generator-* trong {fas_dir}",
                file=sys.stderr,
            )

    fas_labels = group_labels(args.age_group)
    demo = make_app(
        engine,
        gallery_paths,
        gallery_mat,
        args.threshold,
        fas_engine,
        fas_labels,
    )
    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
