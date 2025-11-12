# -*- coding: utf-8 -*-
"""
Level 3：NSGA-II 多目标优化（按 tpms_type 分组）
-------------------------------------------------------
设计思路：
  - 对每一种 TPMS 结构 (G / KEL / VOR 等) 分别构建代理模型、
          分别在该结构族的数据范围内做 NSGA-II 优化，
          最后把所有结构的 Pareto 解合并到一个文件和一张图中。

🎯 目标：
  - 最小化：dp_per_L_pred   （单位压降）
  - 最大化：perm_pred       （渗透率，在算法中用 -perm 表示）

🔧 设计变量（连续变量）：
  - u_in
  - t_wall
  - cell_size


📂 输入数据：
  - data/dataset_surrogate.csv

📤 输出结果：
  - out/pareto_level3_all_types.csv   : 合并后的帕累托解（含 tpms_type）
  - out/pareto_level3_all_types.png   : 不同结构类别的 Pareto 前沿对比图
  - out/nsga2_level3_config.json      : 优化配置快照
  - out/rf_metrics_level3.json        : 各结构代理模型精度指标
"""

import os
from pathlib import Path
import json
import random
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

import sklearn
import matplotlib.pyplot as plt


# =====================================================
# 0. 全局随机种子设置（保证可复现性）
# =====================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)


def set_global_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)


# =====================================================
# 1. 路径与基础配置
# =====================================================
ROOT = Path(__file__).resolve().parents[1]
DATASET_CSV = ROOT / "data" / "dataset_surrogate.csv"   # 或 绝对路径
OUT_DIR = ROOT / "out"

# 设计变量候选
X_VARS_BASE = ["u_in", "t_wall", "eps", "cell_size"]
INCLUDE_DH_IN_X = True
INCLUDE_UVOL_IN_X = True

# 目标列
OBJ_MIN = "dp_per_L"
OBJ_MAX = "perm"

# RF & NSGA-II 参数
TEST_SIZE = 0.2
RF_PARAMS = dict(
    n_estimators=500,
    max_depth=22,
    random_state=SEED,
    n_jobs=1,    # 单线程保证可复现
)
POP = 80
GEN = 40
BOUNDS_MODE = ("quantile", 0.01, 0.99)  # 使用 1% ~ 99% 分位数作为边界


# =====================================================
# 2. 工具函数：数据与特征
# =====================================================
def load_df(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    for c in df.columns:
        if df[c].dtype.kind not in "biufc" and c not in ["run_id", "tpms_type"]:
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


def choose_design_cols(df: pd.DataFrame) -> List[str]:
    """
    给定一个子数据集（单一 tpms_type），自动决定连续设计变量列。
    注意：这里故意不把 tpms_type 放到 X 里，因为已经按类型分组。
    """
    x_vars = list(X_VARS_BASE)
    if INCLUDE_DH_IN_X and "Dh" in df.columns:
        x_vars.append("Dh")
    if INCLUDE_UVOL_IN_X and "u_vol" in df.columns:
        x_vars.append("u_vol")

    num_cols = [c for c in x_vars if c in df.columns]
    return num_cols


def build_preproc(num_cols: List[str]) -> ColumnTransformer:
    """
    对于单一结构类型的数据，tpms_type 没有区分度，因此不作为输入特征；
    只对连续特征做 passthrough。
    """
    return ColumnTransformer([("num", "passthrough", num_cols)], remainder="drop")


# =====================================================
# 3. 代理模型训练（对每个 tpms_type 独立训练）
# =====================================================
def train_model_for_type(
    df_type: pd.DataFrame,
    num_cols: List[str],
    ycol: str,
    seed: int,
) -> Tuple[Pipeline, Dict]:
    """
    对某一个 tpms_type 的数据子集，训练单目标 RF 回归模型。
    """
    if ycol not in df_type.columns:
        raise ValueError(f"数据集中缺少目标列：{ycol}")

    X = df_type[num_cols].copy()
    y = pd.to_numeric(df_type[ycol], errors="coerce").values

    mask = ~X.isna().any(axis=1) & np.isfinite(y)
    Xuse, yuse = X.loc[mask], y[mask]
    if len(yuse) < 30:
        raise ValueError(f"[{ycol}] 类型 {df_type['tpms_type'].iloc[0]} 可用样本过少：{len(yuse)}")

    Xtr, Xte, ytr, yte = train_test_split(
        Xuse,
        yuse,
        test_size=TEST_SIZE,
        random_state=seed,
    )

    prep = build_preproc(num_cols)
    rf = RandomForestRegressor(**{**RF_PARAMS, "random_state": seed})
    pipe = Pipeline([("prep", prep), ("rf", rf)])
    pipe.fit(Xtr, ytr)

    pred = pipe.predict(Xte)
    metrics = dict(
        r2=float(r2_score(yte, pred)),
        mae=float(mean_absolute_error(yte, pred)),
        rmse=float(np.sqrt(mean_squared_error(yte, pred))),
        n_test=int(len(yte)),
    )
    return pipe, metrics


def bounds_from_data(df_type: pd.DataFrame, num_cols: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    在单一 tpms_type 子集上，用分位数给出连续设计变量的边界，
    避免把其它结构的极端 eps / cell_size 引入这个结构族的搜索空间。
    """
    bounds = {}
    for c in num_cols:
        s = pd.to_numeric(df_type[c], errors="coerce").dropna()
        if not len(s):
            bounds[c] = (0.0, 1.0)
            continue

        if BOUNDS_MODE[0] == "quantile":
            lo, hi = s.quantile(BOUNDS_MODE[1]), s.quantile(BOUNDS_MODE[2])
        else:
            lo, hi = s.min(), s.max()

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = s.min(), s.max()
        bounds[c] = (float(lo), float(hi))
    return bounds


# =====================================================
# 4. NSGA-II 问题（ElementwiseProblem）
# =====================================================
class PerTypeProblem(ElementwiseProblem):
    """
    对“某一种 tpms_type”的结构，在其数据对应的连续设计空间上做双目标优化：

      F[0] = dp_per_L_pred
      F[1] = -perm_pred
    """

    def __init__(
        self,
        num_cols: List[str],
        var_bounds: Dict[str, Tuple[float, float]],
        model_dp: Pipeline,
        model_perm: Pipeline,
    ):
        self.num_cols = num_cols
        self.var_bounds = var_bounds   # ⚠️ 注意这里避免用 self.bounds 覆盖基类方法
        self.model_dp = model_dp
        self.model_perm = model_perm

        xl = np.array([self.var_bounds[v][0] for v in num_cols], dtype=float)
        xu = np.array([self.var_bounds[v][1] for v in num_cols], dtype=float)

        super().__init__(n_var=len(num_cols), n_obj=2, n_constr=0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        row = {v: float(x[i]) for i, v in enumerate(self.num_cols)}
        Xdf = pd.DataFrame([row])

        dp = float(self.model_dp.predict(Xdf[self.num_cols])[0])
        pr = float(self.model_perm.predict(Xdf[self.num_cols])[0])

        out["F"] = np.array([dp, -pr], dtype=float)


# =====================================================
# 5. 对单一 tpms_type 运行 NSGA-II 并返回 Pareto 解
# =====================================================
def optimize_one_type(
    df_type: pd.DataFrame,
    tpms_type_value: str,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    对单一 tpms_type（例如 'G' 或 'KEL'）：
      1. 选择设计变量列
      2. 训练 dp_per_L / perm 的 RF 代理
      3. 计算连续变量边界
      4. 跑 NSGA-II，返回帕累托解 DataFrame 和模型指标
    """
    set_global_seed(seed)

    num_cols = choose_design_cols(df_type)

    # 训练两个代理模型
    model_dp, m_dp = train_model_for_type(df_type, num_cols, OBJ_MIN, seed + 1)
    model_perm, m_perm = train_model_for_type(df_type, num_cols, OBJ_MAX, seed + 2)

    metrics = {OBJ_MIN: m_dp, OBJ_MAX: m_perm}

    # 边界（仅基于当前结构的样本）
    bounds = bounds_from_data(df_type, num_cols)

    # 配置 NSGA-II 问题与算法
    problem = PerTypeProblem(
        num_cols=num_cols,
        var_bounds=bounds,      # ✅ 这里传的是 var_bounds
        model_dp=model_dp,
        model_perm=model_perm,
    )

    mut_prob = 1.0 / max(1, len(num_cols))
    algo = NSGA2(
        pop_size=POP,
        sampling=LHS(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=mut_prob, eta=20),
    )
    termination = get_termination("n_gen", GEN)

    set_global_seed(seed)
    res = minimize(
        problem,
        algo,
        termination,
        seed=seed,
        verbose=True,
    )

    X = getattr(res, "X", None)
    if X is None or len(X) == 0:
        return pd.DataFrame(), metrics

    pareto = pd.DataFrame(X, columns=num_cols)
    pareto[OBJ_MIN + "_pred"] = model_dp.predict(pareto[num_cols])
    pareto[OBJ_MAX + "_pred"] = model_perm.predict(pareto[num_cols])

    pareto = pareto.sort_values(
        by=[OBJ_MIN + "_pred", OBJ_MAX + "_pred"],
        ascending=[True, False],
    ).reset_index(drop=True)

    # 补上结构类型
    pareto["tpms_type"] = tpms_type_value

    return pareto, metrics


# =====================================================
# 6. 主流程：对所有 tpms_type 依次优化并汇总
# =====================================================
def main():
    set_global_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_df(DATASET_CSV)

    if "tpms_type" not in df.columns:
        raise ValueError("数据集中缺少 tpms_type 列，请检查！")

    all_types = sorted(df["tpms_type"].dropna().unique().tolist())
    if len(all_types) == 0:
        raise ValueError("tpms_type 列为空，请检查！")

    print(f"检测到的 TPMS 类型：{all_types}")

    all_pareto = []
    all_metrics = {}

    for i, tp in enumerate(all_types):
        print("=" * 60)
        print(f"[Level 3] 正在优化结构类型 tpms_type = {tp}")
        df_type = df[df["tpms_type"] == tp].copy()

        pareto_tp, metrics_tp = optimize_one_type(
            df_type=df_type,
            tpms_type_value=tp,
            seed=SEED + i * 100,
        )

        if not pareto_tp.empty:
            all_pareto.append(pareto_tp)
            all_metrics[tp] = metrics_tp
            print(f"  -> {tp} 类型得到 {len(pareto_tp)} 个帕累托解。")
        else:
            print(f"  -> {tp} 类型没有得到有效解，跳过。")

    if not all_pareto:
        print("⚠ 所有类型都未获得帕累托解，终止。")
        return

    pareto_all = pd.concat(all_pareto, ignore_index=True)

    out_csv = OUT_DIR / "pareto_level3_all_types.csv"
    pareto_all.to_csv(
        out_csv,
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )
    print(f"✅ 合并后的帕累托解已保存：{out_csv} (rows = {len(pareto_all)})")

    # 保存 RF 精度指标（按类型）
    (OUT_DIR / "rf_metrics_level3.json").write_text(
        json.dumps(all_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("✅ 各结构类型的 RF 模型指标已保存：out/rf_metrics_level3.json")

    # 画 Pareto 前沿：不同 tpms_type 用不同颜色
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(6, 4))

    for tp in all_types:
        sub = pareto_all[pareto_all["tpms_type"] == tp]
        if sub.empty:
            continue
        plt.scatter(
            sub[OBJ_MIN + "_pred"],
            sub[OBJ_MAX + "_pred"],
            s=20,
            alpha=0.7,
            label=f"{tp}",
        )

    plt.xlabel(f"{OBJ_MIN}_pred", fontsize=11)
    plt.ylabel(f"{OBJ_MAX}_pred", fontsize=11)
    plt.title("Pareto Fronts by TPMS Type (Level 3)", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(title="tpms_type", fontsize=9)
    plt.tight_layout()

    out_fig = OUT_DIR / "pareto_level3_all_types.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"✅ 多结构类型的 Pareto 对比图已保存：{out_fig}")

    # 保存配置快照
    snapshot = {
        "DATASET_CSV": str(DATASET_CSV),
        "tpms_types": all_types,
        "POP": POP,
        "GEN": GEN,
        "SEED": SEED,
        "BOUNDS_MODE": BOUNDS_MODE,
        "X_VARS_BASE": X_VARS_BASE,
        "INCLUDE_DH_IN_X": INCLUDE_DH_IN_X,
        "INCLUDE_UVOL_IN_X": INCLUDE_UVOL_IN_X,
        "NOTE": "Level 3 per-type NSGA-II; each tpms_type optimized in its own design space.",
    }
    (OUT_DIR / "nsga2_level3_config.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("✅ 配置快照已保存：out/nsga2_level3_config.json")


if __name__ == "__main__":
    main()
