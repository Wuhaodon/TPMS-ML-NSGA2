# TPMS-ML: Surrogate Modeling and Optimization for Porous Structures

**Author:** 武昊东（WUST）  
**Project Level:** 本科课程设计 / 研究生科研拓展  
**Keywords:** CFD, Machine Learning, Random Forest, NSGA-II, TPMS Structures

---

## 📘 Project Overview

本项目展示了如何使用机器学习（ML）替代 CFD 模拟，并基于代理模型实现多目标优化。  
适用于材料科学方向学生进行 **“数字孪生 + 多目标设计”** 的入门实践。

项目共分为三层结构：

| 层次          | 名称         | 主要内容                      | 适用人群          |
|-------------|------------|---------------------------|---------------|
| **Level 1** | 数据准备层      | 整理 TPMS 几何与流动特征数据         | 作者提供          |
| **Level 2** | CFD → ML 层 | 使用机器学习拟合 CFD 结果并进行可视化     | 本科生主要任务       |
| **Level 3** | 优化层        | 基于 ML 代理模型的多目标 NSGA-II 优化 | 研究生 / 本科生拓展任务 |

---

## 🧠 Level 2: CFD → ML Surrogate Modeling

**脚本文件：** `src/surrogate_cfd_ml.py`

**主要功能：**
- **输入变量：** `u_in`, `t_wall`, `cell_size`, `tpms_type`
- **输出目标：** `dp_per_L`, `perm`
- **算法模型：** 随机森林回归（Random Forest Regressor）
- **可视化输出：**
  - `JointGrid` 预测对比图（含 $R^2$ 值与回归线）
  - 自动保存模型性能指标（JSON 格式）

**运行命令：**
```bash
python src/surrogate_cfd_ml.py
````

> 若报依赖缺失，可执行镜像网站安装：
>
> ```bash
> pip install datetime -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

**输出结果目录：**

```
out/
 ├─ joint_dp_per_L.png      # 压降预测对比图
 ├─ joint_perm.png          # 渗透率预测对比图
 ├─ dp_per_L_metrics.json   # 压降预测性能指标
 └─ perm_metrics.json       # 渗透率预测性能指标
```

---

## ⚙️ Level 3: Multi-Objective Optimization (NSGA-II)

**脚本文件：** `src/nsga2_simple_student.py`

**设计变量：**

* `u_in`, `t_wall`, `cell_size`
* `tpms_type`通过 “按类型分组，在各自的设计空间内单独优化” 的方式来对比不同结构族的性能。

**优化目标：**

* **最小化：** 压降 (`dp_per_L_pred`)
* **最大化：** 渗透率 (`perm_pred`)

**说明：**
利用Level 3 在 Level 2 的基础上，进一步完成： 基于代理模型，用 NSGA-II 多目标遗传算法 搜索最优设计； 并对 不同 TPMS 结构类型（如 G, KEL, VOR）分别进行优化，并在结果中进行比较。

**运行命令：**

```bash
python src/nsga2_simple_student.py
```

**输出结果目录：**

```
out/
 ├─ pareto_level3_all_types.csv      # 合并后的帕累托解集（包含 tpms_type）
 ├─ pareto_level3_all_types.png      # 不同 TPMS 结构类型的 Pareto 前沿对比图
 ├─ rf_metrics_level3.json           # 各结构类型的 RF 模型性能指标
 └─ nsga2_level3_config.json         # 本次优化运行配置快照（便于复现）
```
**结果说明：**

在 `pareto_level3_all_types.csv `中，每行代表一组“最优平衡”设计方案




---

## 📦 Recommended Environment

* Python ≥ 3.9
* scikit-learn
* numpy / pandas
* matplotlib / seaborn
* pymoo (for NSGA-II)

> 可使用以下命令快速安装依赖：
>
> ```bash
> pip install -r requirements.txt
> ```

---

## 📈 Educational Value

本项目适合用于：

* 材料科学专业的 **数字孪生与智能设计入门教学**
* 本科课程设计、研究生科研训练
* TPMS 结构的 **多目标优化与代理建模演示**

---

## 🧩 Future Extensions

* 使用更多 ML 模型（如 XGBoost、GBoost、TabPFN 等）对比性能；
* 引入 QoI 约束（例如温度均匀性、剪切率等）构成“有约束的多目标优化”；
* 结合主动学习（Active Learning）策略，自动挑选新的 CFD 计算点扩充样本；
* 与 COMSOL / Fluent 等商业软件建立自动化接口，实现 “仿真–代理–优化–再仿真” 的闭环。

---

## 🧠 Tips

* 机器学习初学项目 https://github.com/fengdu78/machine_learning_beginner

* 进阶机器学习深度学习项目 https://github.com/lucidrains

* kaggle数据库网站 https://www.kaggle.com/

* csdn网站  https://www.csdn.net/

---
© 2025 武昊东, Wuhan University of Science and Technology. All Rights Reserved.


