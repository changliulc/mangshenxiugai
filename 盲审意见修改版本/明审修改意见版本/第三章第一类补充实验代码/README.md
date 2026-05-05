# 第三章第一类补充实验代码

本文件夹只用于准备补充实验代码。本轮不运行实验，也不修改论文正文。

## 设计原则

除模型核心结构不同外，所有实验共用同一套口径：

- 数据入口均为车辆检测后得到的 `ProcessedData + targetLength`。
- 每条样本按 `targetLength` 截取有效事件片段。
- 统一线性重采样到 `L=176`。
- 统一使用 `x/y/z/模值` 四通道输入，张量形状为 `(N, 4, 176)`。
- 归一化参数只由训练集估计，再应用到验证集和测试集。
- 训练阶段默认沿用原论文 pipeline 的普通 shuffle，不启用少数类过采样；如需分析类别均衡采样影响，可显式加入 `--oversample`。
- 模型选择只依据验证集 Macro-F1，测试集只用于最终报告。
- 输出 Accuracy、Macro-F1、三类 Precision/Recall/F1、混淆矩阵和参数量。

## 训练预设

代码提供两套预设：

```python
TRAIN_PRESETS = {
    "legacy_fast": dict(lr=1e-3, weight_decay=0.0, max_epochs=30, patience=6),
    "paper_fair": dict(lr=5e-4, weight_decay=1e-4, max_epochs=200, patience=30),
}
```

`paper_fair` 是默认值，并与当前论文第三章正文中的卷积网络训练协议一致。

`legacy_fast` 仅用于复核旧 pipeline 默认参数。若该预设下的 CNN 数字与论文补充表不同，不应直接混用；论文补充表建议采用 `paper_fair` 下 CNN 和 LSTM 的公平对比结果。

## 模型

- `fixed_cnn_1d`：线性定长 1D-CNN，卷积核长度为 `7/5/3`，通道为 `4 -> 32 -> 64 -> 128`。
- `lstm_h64`：两层单向 LSTM，隐藏维度为 64。
- `lstm_h32`：两层单向 LSTM，隐藏维度为 32，用作容量敏感性对照。

LSTM 使用固定长度输入，是为了与 1D-CNN 使用同一事件表示和同一预处理流程，使对比主要反映模型结构差异，而不是变长序列处理策略差异。LSTM 因四门控结构天然参数密度较高，`hidden_size=64` 已属于较小配置；结果表中会记录参数量，便于解释模型容量差异。

## 推荐运行顺序

在本文件夹下运行。

### 1. 主公平对比

```powershell
python run_baselines_lstm.py --preset paper_fair --out outputs/baselines_paper_fair
```

默认数据路径：

```text
D:\xidian_Master\研究生论文\毕业论文\xduts-main\非核心文件\绘图\第三章\20260304\processedVehicleData_3class_REAL (2).mat
```

### 2. 可选：旧 pipeline 默认参数复核

```powershell
python run_baselines_lstm.py --preset legacy_fast --out outputs/baselines_legacy_fast
```

该结果仅用于解释旧脚本默认参数，不建议与 `paper_fair` 的补充表直接混合。

### 3. 同源分组验证

```powershell
python run_source_group_validation.py --preset paper_fair --out outputs/source_group_paper_fair
```

默认数据路径：

```text
D:\xidian_Master\研究生论文\毕业论文\xduts-main\非核心文件\绘图\第三章\20260304\processedVehicleData_3class_REAL (2).mat
```

该脚本按 `(类别, sourceIndex)` 分组后再进行 `70/15/15` 划分，保证同源组不跨集合。这个实验只能表述为“同源分组稳定性验证”或“泛化能力初步补充”，不能写成严格跨路段泛化实验。严格跨路段实验仍需要样本到真实路段或点位的标签映射。

### 4. 多次运行

```powershell
python run_baselines_lstm.py --preset paper_fair --n_runs 5 --out outputs/baselines_paper_fair_5runs
```

`--n_runs 5` 会使用 `seed, seed+1, ...` 进行重复实验，并输出均值和标准差。

### 5. 汇总结果

```powershell
python summarize_required_experiments.py --in_dir outputs --out tables_required
```

汇总结果包括：

- `tables_required/per_run_metrics_all.csv`
- `tables_required/required_experiment_summary.csv`
- `tables_required/result_provenance.csv`

## 常用参数

- `--models fixed_cnn_1d,lstm_h64,lstm_h32`
- `--preset paper_fair`
- `--seed 42`
- `--n_runs 1`
- `--L 176`
- `--oversample`
- `--device auto`

默认不启用训练集少数类随机过采样，以保持与原论文 pipeline 的采样方式一致。如需额外观察类别均衡采样影响，可加入 `--oversample`；该选项只改变训练集采样概率，每个 epoch 的采样总数保持为训练集样本数。

## 输出说明

每个实验输出目录包含：

- `results.json`：完整配置、split 信息、指标和预测结果。
- `metrics.csv`：每次运行、每个模型的总体指标。
- `summary.csv`：多次运行时的均值和标准差。
- `per_class_metrics.csv`：三类 Precision/Recall/F1。
- `confusion_matrices.csv`：测试集混淆矩阵。
- `split_indices_seed*.npz` 或 `source_group_split_seed*.npz`：对应划分索引。
