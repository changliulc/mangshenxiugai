使用说明（plot_ch3_figures_final.m）

1) 将 plot_ch3_figures_final.m 放在如下文件同级目录：
   - processedVehicleData_3class_REAL (2).mat
   - ch3_assets_plus_matlab/   （其中包含 matlab_data/）

2) 在 MATLAB 中将当前工作目录切换到该目录，然后运行：
   plot_ch3_figures_final

3) 输出位于 figures/ 目录：
   - ch3_waveform_by_class.png            （可选：三类波形示例）
   - fig_motivation_speed.png
   - fig_motivation_dtw_align_z.png
   - cm_cnn_baseline.png
   - cm_dtw_multi_cnn.png
   - ablation_K.png

4) 三类波形图的示例样本会同时导出为 CSV：
   - waveform_class1_idx*.csv
   - waveform_class2_idx*.csv
   - waveform_class3_idx*.csv

5) 若你希望波形图更“紧凑”（更接近师姐论文的 0~1s 量级），可适当调大：
   cropCfg.frac（例如 0.15 或 0.20），它会提高有效段阈值，从而缩短绘图片段。
