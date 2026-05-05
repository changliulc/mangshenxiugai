本压缩包用于第3章（车型三分类）论文编译闭环：表格行文件、图像文件与 MATLAB 绘图数据。

目录结构：
- figures/：论文 \includegraphics 使用的图片文件（png）
- latex_out/tables/：论文 \input 使用的表格行文件（tex）
- matlab_data/：可直接在 MATLAB 中 load 的绘图数据（mat）

说明：
1) 本包不包含 ch3_baseline_flow.pdf 与 ch3_scene_setup.png，这两张通常需要手工绘制/现场照片，请按论文 figures/ 路径自行补齐。
2) ablation_K.png 为重新绘制版本：同时给出验证集与测试集 Macro-F1 曲线，并以验证集选择 K_MT*。
