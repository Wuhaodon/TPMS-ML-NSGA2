# -*- coding: utf-8 -*-
"""
层次 2：CFD 代理模型 + 可视化示例
---------------------------------------------
功能：
  1. 从 data/dataset_surrogate.csv 读取样本数据
  2. 用随机森林分别拟合：
       - dp_per_L  （单位压降）
       - perm      （渗透率）
  3. 输出：
       - out/dp_per_L_metrics.json
       - out/perm_metrics.json
       - out/joint_dp_per_L.png
       - out/joint_perm.png

说明：
  - 设计变量：u_in, t_wall, cell_size, tpms_type
  - tpms_type 作为类别特征，用 One-Hot 编码
  - 可视化使用 seaborn.JointGrid，展示 Train/Test 真值-预测值
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =====================================================
# 全局字体和绘图风格设置（Times New Roman + 白底网格）
# =====================================================
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman"]  # 若系统无该字体，会自动回退
mpl.rcParams["axes.unicode_minus"] = False        # 负号正常显示
mpl.rcParams["font.size"] = 10

sns.set_theme(
    style="whitegrid",
    rc={
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.unicode_minus": False,
        "grid.linestyle": "--",
        "grid.alpha": 0.3,
    },
)


# -----------------------------
# 路径配置（相对路径）
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "dataset_surrogate.csv"
OUT_DIR = ROOT / "out"


# -----------------------------
# 工具函数
# -----------------------------
def load_data(path: Path) -> pd.DataFrame:
    """读取 CSV 数据，并尽量把列转成数值"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到数据文件：{path}")
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype.kind not in "biufc" and c not in ["tpms_type"]:
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


def make_preprocessor(num_cols, cat_cols):
    """数值特征直通，类别特征做独热编码"""
    transformers = [("num", "passthrough", num_cols)]
    if cat_cols:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        transformers.append(("cat", ohe, cat_cols))
    return ColumnTransformer(transformers, remainder="drop")


def train_one_target(
    df: pd.DataFrame,
    target: str,
    num_cols,
    cat_cols,
    random_state: int = 42,
):
    """
    训练单个目标的随机森林回归模型。
    返回：
      model, metrics_dict, y_train, y_pred_train, y_test, y_pred_test
    """
    if target not in df.columns:
        raise ValueError(f"数据中缺少目标列：{target}")

    X = df[num_cols + cat_cols].copy()
    y = pd.to_numeric(df[target], errors="coerce").values

    # 去掉含 NaN 的样本
    mask = ~X[num_cols].isna().any(axis=1) & np.isfinite(y)
    X_use = X.loc[mask]
    y_use = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X_use, y_use, test_size=0.2, random_state=random_state
    )

    preproc = make_preprocessor(num_cols, cat_cols)
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=22,
        random_state=random_state,
        n_jobs=-1,
    )
    model = Pipeline(
        [
            ("prep", preproc),
            ("rf", rf),
        ]
    )

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "r2_train": float(r2_score(y_train, y_pred_train)),
        "r2_test": float(r2_score(y_test, y_pred_test)),
        "mae_train": float(mean_absolute_error(y_train, y_pred_train)),
        "mae_test": float(mean_absolute_error(y_test, y_pred_test)),
        "rmse_train": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
    return model, metrics, y_train, y_pred_train, y_test, y_pred_test


def plot_true_pred_joint(
    y_train,
    y_pred_train,
    y_test,
    y_pred_test,
    r2_train,
    r2_test,
    model_name: str,
    title: str,
    save_path: Path,
    use_sci: bool = False,
):
    """
    基于 JointGrid 的 True vs Predicted 图（美化版）。
    use_sci=True 时，坐标轴使用科学计数法（显示 1e-7 等）。
    """
    # 拼 DataFrame
    data_train = pd.DataFrame(
        {"True": y_train, "Predicted": y_pred_train, "Set": "Train"}
    )
    data_test = pd.DataFrame(
        {"True": y_test, "Predicted": y_pred_test, "Set": "Test"}
    )
    data = pd.concat([data_train, data_test], ignore_index=True)

    # 调色板
    palette = {"Train": "#457b9d", "Test": "#e63946"}

    # JointGrid
    g = sns.JointGrid(
        data=data,
        x="True",
        y="Predicted",
        hue="Set",
        height=7,
        ratio=3,
        space=0,
    )

    # 中心散点图
    g.plot_joint(
        sns.scatterplot,
        alpha=0.7,
        s=30,
        palette=palette,
        edgecolor="none",
    )

    # 训练集回归线
    sns.regplot(
        data=data_train,
        x="True",
        y="Predicted",
        scatter=False,
        ax=g.ax_joint,
        color=palette["Train"],
        line_kws={"linewidth": 2, "linestyle": "-"},
        label="Train fit",
    )

    # 测试集回归线
    sns.regplot(
        data=data_test,
        x="True",
        y="Predicted",
        scatter=False,
        ax=g.ax_joint,
        color=palette["Test"],
        line_kws={"linewidth": 2, "linestyle": "-"},
        label="Test fit",
    )

    # x = y 参考线
    xy_min = data["True"].min()
    xy_max = data["True"].max()
    g.ax_joint.plot(
        [xy_min, xy_max],
        [xy_min, xy_max],
        linestyle="--",
        linewidth=1.5,
        color="black",
        alpha=0.8,
        label="x = y",
    )

    # 上、右边际直方图
    g.plot_marginals(
        sns.histplot,
        bins=25,
        alpha=0.6,
        element="bars",
        multiple="stack",
        palette=palette,
    )

    # 坐标轴与标题
    ax = g.ax_joint
    ax.set_xlabel("True Value", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predicted Value", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # ===== 科学计数法（显示 1e-7 等）=====
    if use_sci:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))  # 始终使用科学计数法
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(
            style="sci", axis="both", scilimits=(0, 0), useOffset=False
        )
    # ====================================

    # 模型名文本框
    ax.text(
        0.98,
        0.98,
        f"Model = {model_name}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"),
    )

    # R^2 文本框
    ax.text(
        0.7,
        0.02,
        rf"Train $R^2$ = {r2_train:.3f}"
        + "\n"
        + rf"Test  $R^2$ = {r2_test:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"),
    )

    # 图例
    ax.legend(
        loc="upper left",
        fontsize=11,
        frameon=True,
        facecolor="white",
        edgecolor="gray",
        framealpha=0.8,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# 主程序
# -----------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_CSV)

    # 设计变量列
    num_cols = ["u_in", "t_wall", "cell_size"]
    cat_cols = ["tpms_type"]

    metrics_all = {}

    # ---- 1) 渗透率 perm ----
    model_perm, m_perm, y_tr_p, y_pr_tr_p, y_te_p, y_pr_te_p = train_one_target(
        df, "perm", num_cols, cat_cols, random_state=42
    )
    metrics_all["perm"] = m_perm

    plot_true_pred_joint(
        y_tr_p,
        y_pr_tr_p,
        y_te_p,
        y_pr_te_p,
        r2_train=m_perm["r2_train"],
        r2_test=m_perm["r2_test"],
        model_name="RandomForest",
        title="perm: True vs Predicted",
        save_path=OUT_DIR / "joint_perm.png",
        use_sci=True,  # perm 量级小，用科学计数法
    )

    # ---- 2) 压降 dp_per_L ----
    model_dp, m_dp, y_tr_d, y_pr_tr_d, y_te_d, y_pr_te_d = train_one_target(
        df, "dp_per_L", num_cols, cat_cols, random_state=52
    )
    metrics_all["dp_per_L"] = m_dp

    plot_true_pred_joint(
        y_tr_d,
        y_pr_tr_d,
        y_te_d,
        y_pr_te_d,
        r2_train=m_dp["r2_train"],
        r2_test=m_dp["r2_test"],
        model_name="RandomForest",
        title="dp_per_L: True vs Predicted",
        save_path=OUT_DIR / "joint_dp_per_L.png",
        use_sci=False,  # 压降量级正常，保持普通刻度
    )

    # 保存指标：每个目标单独一个 JSON，便于配合 README 说明
    (OUT_DIR / "perm_metrics.json").write_text(
        json.dumps(m_perm, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUT_DIR / "dp_per_L_metrics.json").write_text(
        json.dumps(m_dp, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("训练完成，结果已保存到 out/ 目录。")


if __name__ == "__main__":
    main()
