"""short_report.py — Generates a concise 2‑3 page PDF summary of the HPC Backdoor Detector.
Uses ReportLab (available in the environment) and embeds existing figures from reports/figures.
"""

import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Define paths and seed manually without heavy imports
import random
RANDOM_SEED = 42
# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
# Simple seed function
def seed_everything(seed: int = RANDOM_SEED):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

seed_everything(RANDOM_SEED)

# Output locations
REPORT_PDF = REPORTS_DIR / "report.pdf"
PAGE_IMG_DIR = REPORTS_DIR / "report_pages"
PAGE_IMG_DIR.mkdir(parents=True, exist_ok=True)

# ReportLab imports
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

styles = getSampleStyleSheet()

def add_paragraph(story, text, style="Normal", space=0.2 * inch):
    story.append(Paragraph(text, styles[style]))
    story.append(Spacer(1, space))

def add_image(story, img_path, width=5 * inch):
    if Path(img_path).exists():
        img = Image(str(img_path))
        img._restrictSize(width, 7 * inch)
        story.append(img)
        story.append(Spacer(1, 0.2 * inch))

def page_one(story):
    add_paragraph(story, "<b>HPC Backdoor Detector – Method Summary</b>", "Title")
    problem = (
        "Hardware backdoors can stealthily alter micro‑architectural behaviour.<br/>"
        "Only clean HPC traces (800 rows, three counters) are available for training.<br/>"
        "The task is unsupervised anomaly detection: learn the clean data manifold and flag any deviation as a potential backdoor."
    )
    add_paragraph(story, problem)
    data_stats = (
        "Data (800 clean rows) – 3 counters:<br/>"
        "&nbsp;&nbsp;cache‑references   mean≈1.9e6, skewed, bimodal<br/>"
        "&nbsp;&nbsp;cycles             mean≈9.2e6, heavy‑tailed (kurtosis≈6.8)<br/>"
        "&nbsp;&nbsp;LLC‑loads          mean≈1.1e6, bimodal<br/>"
        "Pearson correlation(cache‑ref, LLC‑loads) = 0.92."
    )
    add_paragraph(story, data_stats)
    add_image(story, FIGURES_DIR / "hist_kde_features.png")

def page_two(story):
    add_paragraph(story, "<b>Pre‑processing & Model Choice</b>", "Title")
    pp_text = (
        "* Raw counters are kept (no ratios – they hurt LOF).<br/>"
        "* RobustScaler (median & IQR) is used to mitigate heavy tails and outliers.<br/>"
        "* Feature pipeline is deterministic (seed=42) and exported via joblib."
    )
    add_paragraph(story, pp_text)
    # Benchmark table
    table_data = [
        ["Model", "AUROC (mean±std)", "Worst‑case AUROC"],
        ["LOF (k=20)", "0.9241 ± 0.0031", "0.7577"],
        ["GMM (BIC)", "0.9237 ± 0.0024", "0.7414"],
        ["Ensemble (rank‑fusion)", "0.9215 ± 0.0034", "0.7474"],
    ]
    tbl = Table(table_data, hAlign='LEFT')
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.2 * inch))
    final_desc = (
        "* Selected model: <b>Ensemble</b> of LOF, GMM (BIC), IsolationForest.<br/>"
        "* Scores are rank‑normalized and averaged – robust to per‑model drift.<br/>"
        "* Threshold chosen as the 95th percentile of out‑of‑fold clean scores (P95), giving an empirical FPR ≈ 5 % on training data."
    )
    add_paragraph(story, final_desc)
    add_image(story, FIGURES_DIR / "threshold_tradeoff.png")

def page_three(story):
    add_paragraph(story, "<b>Validation & Outlook</b>", "Title")
    val_text = (
        "* 5‑fold cross‑validation on the 800 clean rows (shuffle, seed = 42).<br/>"
        "* Synthetic anomalies (mean‑shift, variance‑inflation, correlation‑break, ratio‑perturb, uniform box) act as a proxy for unseen backdoor behaviour.<br/>"
        "* Evaluation metrics: Accuracy, TPR, FPR, F1, AUROC.<br/>"
        "* The ensemble achieves AUROC ≈ 0.92 on synthetic anomalies while keeping clean‑data FPR ≤ 5 % (target)."
    )
    add_paragraph(story, val_text)
    add_image(story, FIGURES_DIR / "benchmark_heatmap.png")
    lim_text = (
        "<b>Limitations</b><br/>"
        "* Synthetic anomalies are only a proxy – real backdoors may be subtler.<br/>"
        "* Training set size (N = 800) limits stability; bootstrap CV ≈ 5 % after ensemble.<br/>"
        "* Current model assumes the same hardware platform – systematic scaling offsets (±2 %) still cause high FPR."
    )
    add_paragraph(story, lim_text)
    fut_text = (
        "<b>Future Work</b><br/>"
        "* Collect clean traces from multiple CPUs / NN architectures.<br/>"
        "* Incorporate a lightweight OCSVM component to improve sensitivity to ratio shifts.<br/>"
        "* Explore Conformal Anomaly Detection for calibrated p‑values per prediction.<br/>"
        "* Periodically re‑calibrate the RobustScaler with a few fresh clean traces to handle systematic drift."
    )
    add_paragraph(story, fut_text)

def build_pdf():
    doc = SimpleDocTemplate(str(REPORT_PDF), pagesize=letter)
    story = []
    page_one(story)
    story.append(Spacer(1, 0.5 * inch))
    page_two(story)
    story.append(Spacer(1, 0.5 * inch))
    page_three(story)
    doc.build(story)
    print(f"[DONE] Concise PDF generated -> {REPORT_PDF}")

if __name__ == "__main__":
    build_pdf()
