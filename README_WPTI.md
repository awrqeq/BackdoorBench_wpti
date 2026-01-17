# WPTI（Ours）使用说明

本文档专门介绍本仓库中 `attack/wpti.py`（WPTI）攻击的运行方式与常用配置项，便于你在不同数据集/模型上快速复现实验、做消融、以及对接检测/防御脚本。

---

## 1. 方法概览

WPTI 的核心流程可以理解为：

1. **频域分解**：对图像做小波包分解（WPD），从若干子带系数中选一个“承载子带”（carrier）。
2. **PCA 统计触发方向**：对承载子带的系数向量做 PCA，取特定方向（默认最小方差方向）作为全局触发器方向 `w`（并归一化）。
3. **注入**：对被选中的样本，把 `beta * w`（或按模式缩放后的 `beta`）加回承载子带，再逆变换回图像空间。

输出会保存到 `record/...`，默认会写 `attack_result.pt`（用于后续 NC/STRIP/FP 等）以及可选保存落盘的中毒数据集。

---

## 2. 最常用的运行方式（攻击训练 + 测试）

### 2.1 CIFAR-10 + PreActResNet18（示例）

```bash
python attack/wpti.py \
  --yaml_path config/attack/prototype/cifar10.yaml \
  --bd_yaml_path config/attack/wpti/default.yaml \
  --dataset cifar10 \
  --model preactresnet18 \
  --device cuda:0 \
  --random_seed 42 \
  --pratio 0.1 \
  --beta 0.125
```

### 2.2 指定保存目录（推荐做主表/消融时固定目录结构）

```bash
python attack/wpti.py \
  --yaml_path config/attack/prototype/cifar10.yaml \
  --bd_yaml_path config/attack/wpti/default.yaml \
  --dataset cifar10 \
  --model preactresnet18 \
  --device cuda:0 \
  --random_seed 42 \
  --pratio 0.1 \
  --beta 0.125 \
  --save_folder_name _main_table/wpti/cifar10/preactresnet18/seed42_beta0125
```

会生成：
- `record/_main_table/wpti/cifar10/preactresnet18/seed42_beta0125/attack_result.pt`
- `record/_main_table/wpti/cifar10/preactresnet18/seed42_beta0125/attack_df.csv`（每轮指标）
- `record/_main_table/wpti/cifar10/preactresnet18/seed42_beta0125/attack_df_summary.csv`（汇总）

---

## 3. 关键开关与推荐设置

WPTI 的攻击参数来自两部分：
- **通用训练配置**：`--yaml_path config/attack/prototype/*.yaml`
- **WPTI 专属配置**：`--bd_yaml_path config/attack/wpti/*.yaml`

你可以在 YAML 里改，也可以用命令行覆盖其中部分字段（例如 `--beta`）。

### 3.1 触发强度 `beta`

- `--beta <float>`：触发强度（推荐配合你的 PSNR/ASR sweep 决定）

### 3.2 强度模式 `wpti_beta_mode`

在 `attack/wpti.py` 中支持：
- `fixed`（默认）：所有样本使用同一个 `beta`
- `per_sample_minvar_std`：每个样本用 **最低方差方向的标准差** 来缩放，使每个样本注入强度不同

用法：
```bash
... --wpti_beta_mode fixed
# 或
... --wpti_beta_mode per_sample_minvar_std
```

### 3.3 PCA 方向选择（minvar / midvar / maxvar / random）

WPTI 的 PCA 方向选择在 `bd_yaml_path` 的 `pca.direction_mode` 中配置（也可用命令行覆盖）：

- `minvar`：最小方差方向（默认）
- `midvar`：中间方差方向（消融用）
- `maxvar`：最大方差方向（消融用）
- `random` / `random_full`：全维随机单位向量方向（消融用）


### 3.4 “最小方差 top-k”训练随机、测试用均值（消融/增强）

`--wpti_minvar_topk_train_random K`：
- 训练时：在 **最小方差的前 K 个方向** 中随机选 1 个注入（每样本随机）
- 测试时：用这 K 个方向的均值作为触发方向

```bash
... --wpti_minvar_topk_train_random 3
```

---

## 4. 数据集与配置文件对照

常用：
- CIFAR-10：`--yaml_path config/attack/prototype/cifar10.yaml`
- CIFAR-100：`--yaml_path config/attack/prototype/cifar100.yaml`
- GTSRB：`--yaml_path config/attack/prototype/gtsrb.yaml`
- Tiny：`--yaml_path config/attack/prototype/tiny.yaml`
- Imagenette：`--yaml_path config/attack/prototype/imagenette.yaml`

WPTI 默认参数：
- `config/attack/wpti/default.yaml`
- Imagenette 特化：`config/attack/wpti/imagenette*.yaml`

---

## 5. 输出文件与后处理（用于检测/防御）

WPTI 默认会保存：
- `attack_result.pt`：后续 STRIP/NC/FP/NAD 等都依赖它加载模型与数据 wrapper
- `attack_df.csv` / `attack_df_summary.csv`
- `train_poison_index_list.pickle`

并且你可以通过 YAML 或命令行决定是否落盘中毒数据集：
- `--save_bd_dataset true`（或 `config/attack/wpti/default.yaml` 里设 `save_bd_dataset: true`）

落盘目录示例：
- `record/<run>/bd_train_dataset/`
- `record/<run>/bd_test_dataset/`

---