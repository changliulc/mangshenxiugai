%% 第三章论文图表 - MATLAB 绘图脚本
% 基于 ch3_assets_final/matlab_data/ 中的数据文件

%% 1. K值消融曲线 (K_MT Scanning)
% 文件: ablation_K_data.mat
% 变量: K_list, val_f1, test_f1, val_acc, test_acc, K_best

clc; clear; close all;

cd('d:\xidian_Master\研究生论文\毕业论文\ai写文档需要思考放这里\第三章\20260304\ch3_assets_final');

D = load('matlab_data/ablation_K_data.mat');
K = double(D.K_list);
val = 100*double(D.val_f1);
test = 100*double(D.test_f1);

figure;
plot(K, val, '-o', 'LineWidth', 2, 'MarkerSize', 8); hold on;
plot(K, test, '-s', 'LineWidth', 2, 'MarkerSize', 8);
xline(double(D.K_best), '--', 'LineWidth', 1.5);
grid on;
xlabel('每类模板数 $K_{MT}$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Macro-F_1 (%)', 'FontSize', 14);
legend('验证集 Macro-F_1', '测试集 Macro-F_1', 'Location', 'best', 'FontSize', 12);
title('$K_{MT}$ Scanning: Macro-F_1 Curve', 'Interpreter', 'latex', 'FontSize', 14);
set(gca, 'FontSize', 12);

saveas(gcf, 'figures/ablation_K.png', 'png');
saveas(gcf, 'figures/ablation_K.fig', 'fig');
close;


%% 2. 时间伸缩动机图 (Speed Stretch Motivation)
% 文件: fig_motivation_speed_data.mat
% 变量: t_short, mag_short, t_long, mag_long

D = load('matlab_data/fig_motivation_speed_data.mat');

figure('Units', 'centimeters', 'Position', [10, 10, 14, 5]);
subplot(1, 2, 1);
plot(D.t_short, D.mag_short, 'b-', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title('Short Signal (Length = 176)', 'FontSize', 13);
grid on; box on;

subplot(1, 2, 2);
plot(D.t_long, D.mag_long, 'r-', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title('Long Signal (Length = 352)', 'FontSize', 13);
grid on; box on;

saveas(gcf, 'figures/fig_motivation_speed.png', 'png');
saveas(gcf, 'figures/fig_motivation_speed.fig', 'fig');
close;


%% 3. CNN Baseline 混淆矩阵
% 文件: cm_cnn_baseline_data.mat
% 变量: cm (3×3), labels

D = load('matlab_data/cm_cnn_baseline_data.mat');
cm = double(D.cm);
labels = string(D.labels);

figure;
imagesc(cm); axis image; colorbar;
xticks(1:3); xticklabels(labels);
yticks(1:3); yticklabels(labels);
xlabel('预测标签', 'FontSize', 13);
ylabel('真实标签', 'FontSize', 13);
title('CNN Baseline 混淆矩阵', 'FontSize', 14);

for i = 1:3
    for j = 1:3
        text(j, i, num2str(cm(i,j)), 'HorizontalAlignment', 'center', 'FontSize', 12);
    end
end

saveas(gcf, 'figures/cm_cnn_baseline.png', 'png');
saveas(gcf, 'figures/cm_cnn_baseline.fig', 'fig');
close;


%% 4. DTW-MultiTemplate 混淆矩阵
% 文件: cm_dtw_multi_cnn_data.mat
% 变量: cm (3×3), labels

D = load('matlab_data/cm_dtw_multi_cnn_data.mat');
cm = double(D.cm);
labels = string(D.labels);

figure;
imagesc(cm); axis image; colorbar;
xticks(1:3); xticklabels(labels);
yticks(1:3); yticklabels(labels);
xlabel('预测标签', 'FontSize', 13);
ylabel('真实标签', 'FontSize', 13);
title('DTW-MultiTemplate 混淆矩阵', 'FontSize', 14);

for i = 1:3
    for j = 1:3
        text(j, i, num2str(cm(i,j)), 'HorizontalAlignment', 'center', 'FontSize', 12);
    end
end

saveas(gcf, 'figures/cm_dtw_multi_cnn.png', 'png');
saveas(gcf, 'figures/cm_dtw_multi_cnn.fig', 'fig');
close;


%% 5. DTW对齐示例 (线性定长 vs DTW)
% 文件: fig_motivation_dtw_align_z_data.mat
% 实际变量: t, mag_lin, mag_warp, mag_tpl, x_lin_4L, x_warp_L4, tpl_4L

D = load('matlab_data/fig_motivation_dtw_align_z_data.mat');

figure('Units', 'centimeters', 'Position', [10, 10, 14, 5]);

subplot(1, 2, 1);
plot(D.t, D.mag_lin, 'b-', 'LineWidth', 1.5); hold on;
plot(D.t, D.mag_tpl, 'r--', 'LineWidth', 1.5);
title('线性定长对齐 (Linear Fixed-Length)', 'FontSize', 13);
xlabel('Sample Index', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
legend({'Query (Linear)', 'Template'}, 'Location', 'best');
grid on; box on;

subplot(1, 2, 2);
plot(D.t, D.mag_warp, 'b-', 'LineWidth', 1.5); hold on;
plot(D.t, D.mag_tpl, 'r--', 'LineWidth', 1.5);
title('DTW对齐 (DTW Alignment)', 'FontSize', 13);
xlabel('Sample Index', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
legend({'Query (DTW)', 'Template'}, 'Location', 'best');
grid on; box on;

saveas(gcf, 'figures/fig_motivation_dtw_align_z.png', 'png');
saveas(gcf, 'figures/fig_motivation_dtw_align_z.fig', 'fig');
close;


%% 6. 三类波形示例 (需要从原始数据生成)
% 此图需要根据实际传感器数据生成

fprintf('Done! All figures saved to ./figures/\n');
