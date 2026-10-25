"""
InspectAI | Visual Quality Lab  (dark-blue "zen" theme)
Autoencoder-based anomaly detection for MVTec AD (bottle).

Run:  streamlit run app.py
Keep the .streamlit/config.toml folder next to this file so tables and widgets match the theme.

Expected layout:
    app.py
    .streamlit/config.toml
    bottle_autoencoder.pth
    mvtec_ad/bottle/train/good/*.png
    mvtec_ad/bottle/test/<good|broken_large|broken_small|contamination>/*.png
    mvtec_ad/bottle/ground_truth/<defect>/*_mask.png      (optional)
"""
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ------------------------------ CONFIG ------------------------------
st.set_page_config(
    page_title="InspectAI | Visual Quality Lab",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "mvtec_ad" / "bottle"
MODEL_PATH = ROOT / "bottle_autoencoder.pth"
PREVIOUS_AUROC = 0.8365  # Reported by the previous 10-epoch run; shown for reference only.
PREVIOUS_EPOCHS = 10
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Zen dark-blue palette
BG, PANEL, PANEL2, LINE = "#0B1530", "#121F42", "#0E1A38", "#25376A"
TEXT, MUTED = "#E6ECFF", "#8FA2D0"
ACCENT, ACCENT2 = "#4FD1C5", "#6C8CFF"
BAD, GOOD = "#FF6B7A", "#3DDC97"

# ------------------------------ STYLE ------------------------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Sora:wght@500;600;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family:'Manrope',sans-serif; }}
.stApp {{ background:radial-gradient(ellipse at 85% -10%, #1A2F66 0%, {BG} 45%, #081027 100%); color:{TEXT}; }}
[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ max-width:1400px; padding-top:1.6rem; padding-bottom:3rem; }}
h1,h2,h3,h4 {{ font-family:'Sora',sans-serif !important; color:{TEXT} !important; letter-spacing:-.02em; }}
p, li, label, .stMarkdown, [data-testid="stCaptionContainer"] {{ color:{TEXT}; }}
[data-testid="stCaptionContainer"] {{ color:{MUTED}; }}
[data-testid="stSidebar"] {{ background:{PANEL2}; border-right:1px solid {LINE}; }}
[data-testid="stMetric"] {{ background:{PANEL}; border:1px solid {LINE}; border-radius:12px; padding:14px 18px; }}
[data-testid="stMetricLabel"] p {{ color:{MUTED} !important; }}
[data-testid="stMetricValue"] {{ font-family:'Sora',sans-serif; color:{TEXT}; }}
[data-testid="stFileUploader"] section {{ background:{PANEL}; border:1.5px dashed {ACCENT2}; border-radius:14px; padding:18px; }}
[data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span {{ color:{MUTED}; }}
.stButton>button, .stDownloadButton>button {{ background:linear-gradient(100deg,{ACCENT2},{ACCENT}); color:#06122B; border:0; border-radius:10px; font-weight:700; padding:.55rem 1.1rem; }}
.stButton>button:hover, .stDownloadButton>button:hover {{ color:#06122B; border:0; filter:brightness(1.1); }}
.stTabs [data-baseweb="tab-list"] {{ gap:8px; border-bottom:1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ color:{MUTED}; font-weight:600; padding:10px 18px; }}
.stTabs [aria-selected="true"] {{ color:{ACCENT}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{ACCENT}; }}
[data-testid="stDataFrame"] {{ border:1px solid {LINE}; border-radius:10px; }}
hr {{ border-color:{LINE}; }}
.masthead {{ padding:24px 28px; border:1px solid {LINE}; border-radius:18px; margin-bottom:20px;
  background:linear-gradient(115deg,rgba(26,47,102,.9),rgba(14,26,56,.9) 65%,rgba(21,76,98,.55)); }}
.masthead h1 {{ font-size:2.3rem; margin:0 0 6px 0; }}
.masthead p {{ margin:0; color:{MUTED}; max-width:760px; line-height:1.6; }}
.verdict {{ border-radius:12px; padding:16px 20px; border:1px solid; margin-bottom:14px; }}
.verdict.fail {{ background:rgba(255,107,122,.10); border-color:{BAD}; }}
.verdict.ok {{ background:rgba(61,220,151,.10); border-color:{GOOD}; }}
.verdict b {{ font-family:'Sora',sans-serif; font-size:1.4rem; }}
.verdict.fail b {{ color:{BAD}; }}
.verdict.ok b {{ color:{GOOD}; }}
.verdict p {{ margin:4px 0 0 0; color:{TEXT}; }}
.note {{ border-left:3px solid {ACCENT}; background:{PANEL}; padding:12px 16px; border-radius:0 10px 10px 0; color:{TEXT}; font-size:.92rem; line-height:1.6; }}
footer {{ visibility:hidden; }}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------ MODEL ------------------------------
class LightAE(nn.Module):
    """Same convolutional autoencoder architecture as the training script."""

    def __init__(self):
        super().__init__()

        def down(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU())

        def up(i, o):
            return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU())

        self.enc = nn.Sequential(down(3, 16), down(16, 32), down(32, 64), down(64, 64))
        self.dec = nn.Sequential(
            up(64, 64), up(64, 32), up(32, 16), nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.dec(self.enc(x))


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    model = LightAE().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


preprocess = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])


# ------------------------------ DATA HELPERS ------------------------------
def image_files(folder: Path):
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def class_folders(root: Path):
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []


def dataset_stats():
    train_good = len(image_files(DATA_ROOT / "train" / "good"))
    test_counts = {p.name: len(image_files(p)) for p in class_folders(DATA_ROOT / "test")}
    mask_counts = {p.name: len(image_files(p)) for p in class_folders(DATA_ROOT / "ground_truth")}
    return train_good, test_counts, mask_counts


def find_mask(image_path: Path):
    mask = DATA_ROOT / "ground_truth" / image_path.parent.name / f"{image_path.stem}_mask.png"
    return mask if mask.exists() else None


def open_rgb(path_or_file) -> Image.Image:
    return Image.open(path_or_file).convert("RGB")


# ------------------------------ INFERENCE ------------------------------
@torch.no_grad()
def run_batch(model, tensors: torch.Tensor):
    """Returns reconstructions (B,3,H,W) and smoothed per-pixel squared error (B,H,W)."""
    x = tensors.to(DEVICE)
    recon = model(x)
    err = ((x - recon) ** 2).mean(dim=1, keepdim=True)
    err = F.avg_pool2d(err, kernel_size=5, stride=1, padding=2)
    return recon.cpu(), err[:, 0].cpu()


def image_scores(err: np.ndarray):
    flat = err.reshape(err.shape[0], -1)
    k = max(1, int(flat.shape[1] * 0.01))
    peak = np.sort(flat, axis=1)[:, -k:].mean(axis=1)  # mean of the worst 1% of pixels
    return {"Mean error": flat.mean(axis=1), "Peak error (top 1%)": peak}


@st.cache_data(show_spinner=False)
def score_paths(paths: tuple):
    model = load_model()
    out = {"Mean error": [], "Peak error (top 1%)": []}
    pixel_sample = []
    for i in range(0, len(paths), 32):
        batch = torch.stack([preprocess(open_rgb(p)) for p in paths[i : i + 32]])
        _, err = run_batch(model, batch)
        err = err.numpy()
        for name, vals in image_scores(err).items():
            out[name].extend(vals.tolist())
        pixel_sample.append(err.reshape(-1)[::17])
    return {k: np.array(v) for k, v in out.items()}, (
        np.concatenate(pixel_sample) if pixel_sample else np.zeros(1)
    )


@st.cache_data(show_spinner="Calibrating on normal training images…")
def calibration():
    paths = tuple(str(p) for p in image_files(DATA_ROOT / "train" / "good"))
    if not paths:
        return None
    scores, pixels = score_paths(paths)
    return {"train": scores, "pix_ref": float(np.percentile(pixels, 99.5)), "n": len(paths)}


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC (no sklearn needed). labels: 1 = defect."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def overlay(img: Image.Image, err: np.ndarray, pix_ref: float, alpha: float) -> np.ndarray:
    base = np.asarray(img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    norm = np.clip(err / max(pix_ref, 1e-8), 0, 1)
    heat = plt.get_cmap("inferno")(norm)[..., :3]
    w = (norm * alpha)[..., None]
    return np.clip(base * (1 - w) + heat * w, 0, 1)


def analyze(img: Image.Image):
    model = load_model()
    recon, err = run_batch(model, preprocess(img).unsqueeze(0))
    err_np = err.numpy()
    return {
        "recon": recon[0].permute(1, 2, 0).numpy(),
        "err": err_np[0],
        "scores": {k: float(v[0]) for k, v in image_scores(err_np).items()},
    }


@st.cache_data(show_spinner=False)
def analyze_bytes(data: bytes):
    return analyze(open_rgb(io.BytesIO(data)))


def style_axes(fig, ax):
    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor(PANEL)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LINE)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)


# ------------------------------ PRE-FLIGHT ------------------------------
st.markdown(
    '<div class="masthead"><h1>🔎 InspectAI</h1>'
    "<p>Visual quality lab. An autoencoder trained only on normal bottles flags anything it cannot "
    "reconstruct, and shows you where.</p></div>",
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(
        f"Model weights not found at `{MODEL_PATH}`. Place `bottle_autoencoder.pth` "
        "next to `app.py` and reload."
    )
    st.stop()

try:
    load_model()
except Exception as exc:
    st.error(f"The weights could not be loaded: {exc}")
    st.info("Check that the `LightAE` class in this file matches the one used for training.")
    st.stop()

has_data = (DATA_ROOT / "train" / "good").exists()
calib = calibration() if has_data else None

# ------------------------------ SIDEBAR ------------------------------
with st.sidebar:
    st.header("Settings")
    score_mode = st.selectbox(
        "Scoring method",
        ["Peak error (top 1%)", "Mean error"],
        help="Peak error reacts to small local defects. Mean error is the classic whole-image reconstruction error.",
    )
    if calib:
        train_scores = calib["train"][score_mode]
        default_thr = float(np.percentile(train_scores, 99))
        hi = float(train_scores.max() * 4)
    else:
        default_thr, hi = 0.01, 0.1
    threshold = st.slider(
        "Defect threshold",
        min_value=0.0,
        max_value=max(hi, default_thr * 2),
        value=default_thr,
        step=max(hi / 500, 1e-6),
        format="%.5f",
        help="Images scoring above this are flagged. Default is the 99th percentile of normal training images.",
    )
    alpha = st.slider("Heatmap strength", 0.0, 1.0, 0.75, 0.05)
    st.caption(f"Device: {DEVICE.type.upper()}")
    if calib:
        st.caption(f"Calibrated on {calib['n']} normal training images.")
    else:
        st.warning("Training images not found, so the threshold is a placeholder. Set it manually.")

pix_ref = calib["pix_ref"] if calib else 0.02


# ------------------------------ RESULT VIEW ------------------------------
def render_result(image: Image.Image, res: dict, image_path: Path | None = None):
    score = res["scores"][score_mode]
    flagged = score > threshold
    st.markdown(
        f'<div class="verdict {"fail" if flagged else "ok"}"><b>'
        f'{"Defect suspected" if flagged else "Looks normal"}</b>'
        f"<p>{score_mode} {score:.5f} is {'above' if flagged else 'below'} the threshold of {threshold:.5f}.</p></div>",
        unsafe_allow_html=True,
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Anomaly score", f"{score:.5f}")
    m2.metric("Threshold", f"{threshold:.5f}")
    m3.metric("Score ÷ threshold", f"{score / threshold:.2f}×" if threshold > 0 else "n/a")

    mask_path = find_mask(image_path) if image_path else None
    cols = st.columns(4 if mask_path else 3)
    big = (IMG_SIZE * 3, IMG_SIZE * 3)
    cols[0].image(image.resize(big), caption="Input", use_container_width=True)
    cols[1].image(np.clip(res["recon"], 0, 1), caption="Autoencoder reconstruction", use_container_width=True)
    cols[2].image(
        overlay(image, res["err"], pix_ref, alpha),
        caption="Where it differs from normal",
        use_container_width=True,
    )
    if mask_path:
        cols[3].image(
            Image.open(mask_path).convert("L").resize(big),
            caption="Ground-truth defect mask",
            use_container_width=True,
        )
    st.markdown(
        '<div class="note">The model only learned what normal bottles look like. It reconstructs normal '
        "areas well and defects poorly. Bright areas in the heatmap mark where that reconstruction failed.</div>",
        unsafe_allow_html=True,
    )


tab_upload, tab_samples, tab_eval, tab_data = st.tabs(
    ["Upload images", "Test samples", "Model performance", "Dataset"]
)

# ------------------------------ TAB: UPLOAD ------------------------------
with tab_upload:
    st.subheader("Inspect your own images")
    files = st.file_uploader(
        "Drop one or more bottle images here (PNG or JPG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    if not files:
        st.info("Upload one image for a detailed view, or several for a batch report.")
    else:
        results, bad_files = [], []
        for f in files:
            try:
                results.append((f.name, f.getvalue(), analyze_bytes(f.getvalue())))
            except Exception:
                bad_files.append(f.name)
        if bad_files:
            st.error("Could not read: " + ", ".join(bad_files))

        if results:
            if len(results) > 1:
                table = pd.DataFrame(
                    [
                        {
                            "File": n,
                            "Score": round(r["scores"][score_mode], 5),
                            "Verdict": "Defect suspected" if r["scores"][score_mode] > threshold else "Normal",
                        }
                        for n, _, r in results
                    ]
                )
                flagged_n = int((table["Verdict"] != "Normal").sum())
                a, b, c = st.columns(3)
                a.metric("Images checked", len(table))
                b.metric("Flagged", flagged_n)
                c.metric("Normal", len(table) - flagged_n)
                st.dataframe(table, hide_index=True, use_container_width=True)
                st.download_button(
                    "Download report (CSV)",
                    table.to_csv(index=False).encode(),
                    "inspectai_report.csv",
                    "text/csv",
                )
                st.divider()
                pick = st.selectbox("Open details for", range(len(results)), format_func=lambda i: results[i][0])
            else:
                pick = 0
            name, data, res = results[pick]
            render_result(open_rgb(io.BytesIO(data)), res)

# ------------------------------ TAB: TEST SAMPLES ------------------------------
with tab_samples:
    st.subheader("Browse the MVTec test set")
    test_dirs = [d for d in class_folders(DATA_ROOT / "test") if image_files(d)]
    if not test_dirs:
        st.info(f"No test images found in `{DATA_ROOT / 'test'}`. Use the Upload images tab instead.")
    else:
        c1, c2 = st.columns([1, 2])
        cls = c1.selectbox("Class", [d.name for d in test_dirs])
        flist = image_files(DATA_ROOT / "test" / cls)
        idx = c2.selectbox("Image", range(len(flist)), format_func=lambda i: flist[i].name)
        img = open_rgb(flist[idx])
        render_result(img, analyze(img), flist[idx])

# ------------------------------ TAB: PERFORMANCE ------------------------------
with tab_eval:
    test_dirs = [d for d in class_folders(DATA_ROOT / "test") if image_files(d)]
    if not test_dirs:
        st.info("Add the MVTec AD test folders to evaluate the model on labeled images.")
    else:
        st.write(
            "Scores every test image with the current model and measures how well it separates "
            "defects from normal bottles."
        )
        if st.button("Run evaluation"):
            paths, labels, classes = [], [], []
            for d in test_dirs:
                for p in image_files(d):
                    paths.append(str(p))
                    labels.append(0 if d.name == "good" else 1)
                    classes.append(d.name)
            with st.spinner(f"Scoring {len(paths)} test images…"):
                scores, _ = score_paths(tuple(paths))
            st.session_state["eval"] = {
                "scores": scores,
                "labels": np.array(labels),
                "classes": np.array(classes),
            }

        ev = st.session_state.get("eval")
        if ev:
            s, y = ev["scores"][score_mode], ev["labels"]
            auc = auroc(s, y)
            pred = (s > threshold).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            tn = int(((pred == 0) & (y == 0)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            a, b, c, d = st.columns(4)
            a.metric("AUROC", f"{auc:.4f}", f"{auc - PREVIOUS_AUROC:+.4f} vs {PREVIOUS_EPOCHS}-epoch run")
            b.metric("Accuracy at threshold", f"{(tp + tn) / len(y):.1%}")
            c.metric("Defects caught", f"{tp / max(tp + fn, 1):.1%}")
            d.metric("False alarms", f"{fp / max(fp + tn, 1):.1%}")

            left, right = st.columns([3, 2])
            with left:
                fig, ax = plt.subplots(figsize=(7, 3.4))
                bins = np.linspace(s.min(), s.max(), 30)
                ax.hist(s[y == 0], bins=bins, alpha=0.85, color=GOOD, label="Normal")
                ax.hist(s[y == 1], bins=bins, alpha=0.85, color=BAD, label="Defective")
                ax.axvline(threshold, color=ACCENT, linestyle="--", linewidth=1.4, label="Threshold")
                ax.set_xlabel(score_mode)
                ax.set_ylabel("Images")
                ax.set_title("Score distribution on the test set", loc="left", fontsize=11)
                ax.legend(frameon=False, labelcolor=TEXT)
                style_axes(fig, ax)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with right:
                st.markdown("**Confusion matrix**")
                st.dataframe(
                    pd.DataFrame(
                        [[tn, fp], [fn, tp]],
                        index=["Actually normal", "Actually defective"],
                        columns=["Flagged normal", "Flagged defect"],
                    ),
                    use_container_width=True,
                )
                rows = [
                    {
                        "Defect type": cname,
                        "Images": int((ev["classes"] == cname).sum()),
                        "Caught": f"{(pred[ev['classes'] == cname] == 1).mean():.0%}",
                    }
                    for cname in sorted(set(ev["classes"]))
                    if cname != "good"
                ]
                if rows:
                    st.markdown("**Detection rate by defect type**")
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption(
                f"The {PREVIOUS_EPOCHS}-epoch run reported AUROC {PREVIOUS_AUROC:.4f}. "
                "That figure is a stored reference. The AUROC above is computed live from the loaded weights."
            )

# ------------------------------ TAB: DATASET ------------------------------
with tab_data:
    if not DATA_ROOT.exists():
        st.info(f"Dataset folder not found at `{DATA_ROOT}`.")
    else:
        train_good, test_counts, mask_counts = dataset_stats()
        a, b, c = st.columns(3)
        a.metric("Normal training images", train_good)
        b.metric("Test images", sum(test_counts.values()))
        c.metric("Defect masks", sum(mask_counts.values()))
        if test_counts:
            st.markdown("**Test images per class**")
            st.bar_chart(pd.Series(test_counts, name="Images"), color=ACCENT)
        st.markdown("**Samples**")
        for col, (cname, n) in zip(st.columns(max(len(test_counts), 1)), test_counts.items()):
            fl = image_files(DATA_ROOT / "test" / cname)
            if fl:
                col.image(open_rgb(fl[0]).resize((256, 256)), caption=f"{cname} ({n})", use_container_width=True)