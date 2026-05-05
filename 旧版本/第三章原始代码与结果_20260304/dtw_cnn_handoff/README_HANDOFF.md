# DTW 对齐 + CNN（复现闭环交接说明）

本目录用于把当前阶段的实验复现、关键结论、以及完整代码交接给后续 AI/同学继续扩展。

---

## 1. 输入数据与约定

- **数据文件**：`ac949708-72b7-4e9b-be27-bb6c78f1f5b6.mat`
- **MD5**：`a63028f89be3cbf86c49392c9792cedc`

该 `.mat` 文件至少包含两个变量：
- `ProcessedData`：长度为 3 的 cell（3 类），每个元素是该类的样本序列列表（每条样本为 `(T,3)` 的 xyz 加速度）
- `targetLength`：长度为 3 的 cell，与 `ProcessedData` 对应，给出每条样本的有效长度 `T`

预处理（与代码一致）：
- 统一线性插值到固定长度 **L=176**
- 额外计算 `mag = sqrt(x^2+y^2+z^2)`，最终输入为 **4 通道**（x,y,z,mag）

---

## 2. 切分与无泄漏原则

- 切分方式：**分层随机** `70%/15%/15%`
- 固定随机种子：`seed=42`
- 切分结果：`train=956 / val=205 / test=205`

严格无泄漏：
- 归一化 `mean/std` **只用 train 计算**
- DTW 模板 **只从 train 选取/构造**

---

## 3. 方法定义（与代码实现一致）

### 3.1 Baseline CNN（不对齐）
- 输入：`(4,176)`
- 训练：在 train 上训练；val 监控 macro-F1；test 仅做最终评估

### 3.2 DTW-ClsMin + CNN（单模板 + 硬路由）
- 每类 1 个模板（原始脚本用 **类内算术均值**）
- 对每条样本：
  1) 仅用 `mag` 通道计算到 3 个类模板的 DTW 距离，选最小类 `c*`
  2) 将 4 通道序列 DTW 对齐到 `c*` 类模板
  3) 对齐后仍是 `(4,176)` 输入 CNN

### 3.3 DTW-Multi + CNN（多模板拼接）
- 每类随机抽 `templates_per_class = K` 条训练样本做模板，总模板 `T=3K`
- 对每条样本：分别对齐到每个模板，得到 `T` 份对齐序列；沿通道维拼接
- 输入维度：`(4*T,176)`，如 `K=4` 则输入为 `(48,176)`

---

## 4. 完整 sweep 网格（由脚本内 configs 固定）

### 4.1 ClsMin sweep（`run_dtw_clsmin_sweep.py`）
配置（wR=窗口比，step=步进惩罚）：
- (0.10,0.00), (0.15,0.00), (0.20,0.00)
- (0.10,0.05), (0.15,0.05), (0.20,0.05)
- (0.15,0.10)

选参：以 **val_macroF1 最大** 为 best。

### 4.2 Multi sweep（`run_dtw_multi_sweep.py`）
配置（tpc=templates_per_class）：
- (tpc=3, wR=0.10, step=0.00)
- (tpc=3, wR=0.15, step=0.00)
- (tpc=3, wR=0.20, step=0.00)
- (tpc=3, wR=0.15, step=0.05)
- (tpc=4, wR=0.15, step=0.00)
- (tpc=4, wR=0.15, step=0.05)
- (tpc=5, wR=0.15, step=0.00)

选参：以 **val_macroF1 最大** 为 best。

---

## 5. 关键复现结果（seed=42, L=176）

结果来源：`results/*.txt`（原始 stdout），以及整理后的 CSV。

### 5.1 Baseline CNN
- `val_acc=0.9220, val_macroF1=0.9180`
- `test_acc=0.8683, test_macroF1=0.8601`

### 5.2 DTW-ClsMin + CNN（均值模板）
- **BEST_by_val_macroF1**：`wR=0.10, step=0.05`
  - `val_macroF1=0.9280`
  - `test_acc=0.8732, test_macroF1=0.8733`
- 观察：按 test 来看更高的一组是 `wR=0.15, step=0.05`，`test_acc=0.8878, test_macroF1=0.8916`，但它不是 val 最优。

### 5.3 DTW-Multi + CNN（拼接）
- **BEST_by_val_macroF1**：`templates_per_class=4, wR=0.15, step=0.05`
  - `val_macroF1=0.9676`
  - `test_acc=0.9659, test_macroF1=0.9649`

> 解释要点：Multi-template 输入通道从 4 扩展到 48，第一层卷积参数量约线性 ×12；性能提升来源包括“多模板覆盖类内多模态 + 更高维输入/模型容量”。论文建议补充参数量公平性消融。

---

## 6. 建议的下一步（给后续 AI 的任务列表）

1) **公平性消融（强烈建议）**
   - 让 baseline CNN 的参数量接近 Multi（增大 base/通道数），比较是否仍显著落后。
   - 或在 Multi 输入后增加 `1×1 Conv` 压缩通道（例如 48→8/16），分离“模板库表示”与“容量提升”。

2) **ClsMin 改进但不改变主体**
   - 单模板：用 DTW 语义更合理的模板（medoid 或 DBA）替代算术均值。
   - 路由：从 hard argmin 改为 top-k/softmin（仍输出 4×L，尽量不改 CNN）。

3) **稳定性**
   - 多 seed（≥10）重复，报告 mean±std。

4) **若存在 source/session 信息**
   - 增加 group split（按 source）验证泛化；避免同源泄漏导致的虚高。

---

## 7. 代码与文件结构

- `code/`：原始脚本（6 个）+ 额外脚本
  - `run_baseline_min.py`
  - `run_dtw_clsmin_sweep.py`
  - `run_dtw_multi_sweep.py`
  - `run_cnn_dtw_experiment.py`
  - `run_dtw_numba2.py`
  - `run_dtw_clsmin_numba2.py`
  - **`run_dtw_clsmin_dba.py`**：附加（DBA 模板版 ClsMin），用于探索“最小改动提升 ClsMin”的可能性。

- `results/`：本次复现输出与汇总
  - `clsmin_sweep_out.txt`
  - `multi_full_out_ac949.txt`
  - `multi_full_summary_ac949_seed42_L176.csv`
  - `closed_loop_summary_ac949_seed42_L176.csv/json`

---

## 8. 运行方式

原脚本默认把数据写死在：
`/mnt/data/3f82e08c-9373-4134-b811-015b645b3a9e.mat`

因此推荐用软链接/拷贝保证路径一致。

一键复现：
```bash
bash run_all.sh /absolute/path/to/ac949708-72b7-4e9b-be27-bb6c78f1f5b6.mat
```

依赖见 `requirements.txt`。
