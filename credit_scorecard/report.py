import pandas as pd
from io import BytesIO
from pathlib import Path
import matplotlib.pyplot as plt
from aiweb_common.report_builder.report_builder import ReportBuilder

from credit_scorecard.config.config import logger


def build_report(pred_path: str, fig_path: str = "eval_plot.png") -> BytesIO:
    """
    Build a report zip containing predictions and (optionally) an evaluation plot.
    Returns BytesIO zip object.
    """
    # Load predictions DataFrame
    df = pd.read_csv(pred_path)

    fig = None
    fig_file = Path(fig_path)

    # Only try to load the plot if it exists
    if fig_path and fig_file.exists():
        fig = plt.figure()
        img = plt.imread(str(fig_file))
        plt.imshow(img)
        plt.axis("off")
        plt.close(fig)
    else:
        logger.info(f"Report: plot not found, skipping figure: {fig_path}")

    with ReportBuilder() as rb:
        rb.add_dataframe(df, "predictions.csv")
        if fig is not None:
            rb.add_figure(fig, "eval_plot.png")  # keep name consistent with your pipeline
        zip_bytes = rb.build_zip()

    return zip_bytes