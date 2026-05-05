%% 第三章论文图片 - 统一绘图脚本
% 生成所有可自动生成的图片 (PNG格式)
% 需排除: ch3_baseline_flow.pdf, ch3_scene_setup.png (需手工)

clc; clear; close all;

%% ========================== 配置 ==========================
% 脚本所在目录
scriptDir = 'd:\xidian_Master\研究生论文\毕业论文\ai写文档需要思考放这里\第三章\20260304';
cd(scriptDir);

dataDir = fullfile(scriptDir, 'ch3_assets_plus_matlab');          % 数据目录
rawDataFile = fullfile(scriptDir, 'processedVehicleData_3class_REAL (2).mat');  % 原始数据
outDir = fullfile(dataDir, 'figures');
if ~exist(outDir, 'dir'); mkdir(outDir); end

fs = 50; N0 = 10; wR = 0.15; lambda = 0.05;
classNames = {'小型车','中型车','大型车'};

fprintf('=== 第三章图片生成开始 ===\n');
fprintf('输出目录: %s\n\n', outDir);

%% ========================== 1. 三类波形图 ==========================
% 输出: ch3_waveform_by_class.png
fprintf('[1/6] 生成三类波形图...\n');

S = load(rawDataFile);
ProcessedData = S.ProcessedData;
targetLength = S.targetLength;

tolLen = 3; repIdx = zeros(1,3); repLen = zeros(1,3);
for c = 1:3
    lenVec = double(targetLength{c}(:));
    medLen = round(median(lenVec));
    cand = find(abs(lenVec - medLen) <= tolLen);
    if isempty(cand); cand = 1:numel(lenVec); end
    bestE = -inf; bestI = cand(1);
    for ii = cand(:)'
        [~, ~, b, ~] = extract_event(ProcessedData{c}{ii}, lenVec(ii), fs, N0);
        E = sum(b.^2);
        if E > bestE; bestE = E; bestI = ii; end
    end
    repIdx(c) = bestI; repLen(c) = lenVec(bestI);
end

fig1 = figure('Color','w','Position',[100 100 900 650]);
for c = 1:3
    [~, ~, b, t] = extract_event(ProcessedData{c}{repIdx(c)}, repLen(c), fs, N0);
    subplot(3,1,c);
    plot(t, b, 'LineWidth', 1.2);
    grid on;
    xlabel('时间 / s');
    ylabel('b[n]');
    title(sprintf('%s（示例：index=%d，N=%d）', classNames{c}, repIdx(c), repLen(c)));
end
print(fig1, fullfile(outDir, 'ch3_waveform_by_class.png'), '-dpng', '-r300');
close(fig1);

%% ========================== 2. 时间伸缩动机图 ==========================
% 输出: fig_motivation_speed.png
fprintf('[2/6] 生成时间伸缩动机图...\n');

cStretch = 2;
lenVec = double(targetLength{cStretch}(:));
q = 0.15;
idxSort = sortrows([(1:numel(lenVec))', lenVec], 2);
nAll = size(idxSort,1);
shortPool = idxSort(1: max(1, round(q*nAll)), 1);
longPool = idxSort(max(1, round((1-q)*nAll)) : nAll, 1);

idxShort = pick_max_energy(ProcessedData{cStretch}, lenVec, shortPool, fs, N0);
idxLong = pick_max_energy(ProcessedData{cStretch}, lenVec, longPool, fs, N0);

[~, ~, b_s, t_s] = extract_event(ProcessedData{cStretch}{idxShort}, lenVec(idxShort), fs, N0);
[~, ~, b_l, t_l] = extract_event(ProcessedData{cStretch}{idxLong}, lenVec(idxLong), fs, N0);

fig2 = figure('Color','w','Position',[100 100 900 420]);
plot(t_s, b_s, 'LineWidth', 1.2); hold on;
plot(t_l, b_l, 'LineWidth', 1.2);
grid on;
xlabel('时间 / s');
ylabel('b[n]');
legend({sprintf('短时长样本（N=%d）', lenVec(idxShort)), ...
        sprintf('长时长样本（N=%d）', lenVec(idxLong))}, 'Location','best');
title(sprintf('车速变化引起的时间伸缩示意（%s）', classNames{cStretch}));
print(fig2, fullfile(outDir, 'fig_motivation_speed.png'), '-dpng', '-r300');
close(fig2);

%% ========================== 3. DTW对齐示例 ==========================
% 输出: fig_motivation_dtw_align_z.png
fprintf('[3/6] 生成DTW对齐示例图...\n');

[~, dB_s, ~, ~] = extract_event(ProcessedData{cStretch}{idxShort}, lenVec(idxShort), fs, N0);
[~, dB_l, ~, ~] = extract_event(ProcessedData{cStretch}{idxLong}, lenVec(idxLong), fs, N0);
X4 = [dB_s, b_s]; Y4 = [dB_l, b_l];
zX = dB_s(:,3); zY = dB_l(:,3);

N = size(X4,1); M = size(Y4,1);
w = max(floor(wR * max(N,M)), abs(N-M));
[path, ~] = dtw_path_4ch(X4, Y4, w, lambda);
X4_aligned = warp_to_ref_axis(X4, path, M);
zX_aligned = X4_aligned(:,3);
tRef = (0:M-1)'/fs;

fig3 = figure('Color','w','Position',[100 100 900 600]);
subplot(2,1,1);
plot((0:N-1)'/fs, zX, 'LineWidth', 1.2); hold on;
plot((0:M-1)'/fs, zY, 'LineWidth', 1.2);
grid on; xlabel('时间 / s'); ylabel('\Delta B_z[n]');
title('对齐前（同类样本存在时间伸缩与局部错位）');
legend({sprintf('样本A（N=%d）', N), sprintf('样本B（N=%d）', M)}, 'Location','best');

subplot(2,1,2);
plot(tRef, zY, 'LineWidth', 1.2); hold on;
plot(tRef, zX_aligned, 'LineWidth', 1.2);
grid on; xlabel('参考时间轴 / s'); ylabel('\Delta B_z[n]');
title('DTW 对齐后（映射到参考时间轴）');
legend({'参考样本B','对齐后的样本A'}, 'Location','best');

print(fig3, fullfile(outDir, 'fig_motivation_dtw_align_z.png'), '-dpng', '-r300');
close(fig3);

%% ========================== 4. CNN Baseline混淆矩阵 ==========================
% 输出: cm_cnn_baseline.png
fprintf('[4/6] 生成CNN Baseline混淆矩阵...\n');

D = load(fullfile(dataDir, 'matlab_data', 'cm_cnn_baseline_data.mat'));
cm = double(D.cm); labels = string(D.labels);

fig4 = figure;
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
print(fig4, fullfile(outDir, 'cm_cnn_baseline.png'), '-dpng', '-r300');
close(fig4);

%% ========================== 5. DTW-MultiTemplate混淆矩阵 ==========================
% 输出: cm_dtw_multi_cnn.png
fprintf('[5/6] 生成DTW-MultiTemplate混淆矩阵...\n');

D = load(fullfile(dataDir, 'matlab_data', 'cm_dtw_multi_cnn_data.mat'));
cm = double(D.cm); labels = string(D.labels);

fig5 = figure;
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
print(fig5, fullfile(outDir, 'cm_dtw_multi_cnn.png'), '-dpng', '-r300');
close(fig5);

%% ========================== 6. K_MT扫描曲线 ==========================
% 输出: ablation_K.png
fprintf('[6/6] 生成K_MT扫描曲线...\n');

D = load(fullfile(dataDir, 'matlab_data', 'ablation_K_data.mat'));
K = double(D.K_list); val = 100*double(D.val_f1); test = 100*double(D.test_f1);

fig6 = figure;
plot(K, val, '-o', 'LineWidth', 2, 'MarkerSize', 8); hold on;
plot(K, test, '-s', 'LineWidth', 2, 'MarkerSize', 8);
xline(double(D.K_best), '--', 'LineWidth', 1.5);
grid on;
xlabel('每类模板数 $K_{MT}$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Macro-F_1 (%)', 'FontSize', 14);
legend('验证集 Macro-F_1', '测试集 Macro-F_1', 'Location', 'best', 'FontSize', 12);
title('$K_{MT}$ Scanning: Macro-F_1 Curve', 'Interpreter', 'latex', 'FontSize', 14);
set(gca, 'FontSize', 12);
print(fig6, fullfile(outDir, 'ablation_K.png'), '-dpng', '-r300');
close(fig6);

fprintf('\n=== 全部完成! ===\n');
fprintf('输出目录: %s\n', outDir);
fprintf('生成图片:\n');
fprintf('  1. ch3_waveform_by_class.png (三类波形)\n');
fprintf('  2. fig_motivation_speed.png (时间伸缩)\n');
fprintf('  3. fig_motivation_dtw_align_z.png (DTW对齐)\n');
fprintf('  4. cm_cnn_baseline.png (CNN混淆矩阵)\n');
fprintf('  5. cm_dtw_multi_cnn.png (DTW混淆矩阵)\n');
fprintf('  6. ablation_K.png (K_MT扫描)\n');

%% ========================== 辅助函数 ==========================
function [B, dB, b, t] = extract_event(Bpad, N, fs, N0)
    N = double(N);
    B = double(Bpad(1:N, :));
    n0 = min(N0, N);
    B0 = mean([B(1:n0,:); B(N-n0+1:N,:)], 1);
    dB = B - B0;
    b = sqrt(sum(dB.^2, 2));
    t = (0:N-1)'/fs;
end

function idx = pick_max_energy(classCell, lenVec, idxPool, fs, N0)
    bestE = -inf; idx = idxPool(1);
    for ii = idxPool(:)'
        [~, ~, b, ~] = extract_event(classCell{ii}, lenVec(ii), fs, N0);
        E = sum(b.^2);
        if E > bestE; bestE = E; idx = ii; end
    end
end

function [path, D] = dtw_path_4ch(X, Y, w, lambda)
    N = size(X,1); M = size(Y,1);
    D = inf(N, M);
    D(1,1) = sum((X(1,:)-Y(1,:)).^2);
    for i = 2:N
        if abs(i-1) <= w
            D(i,1) = sum((X(i,:)-Y(1,:)).^2) + D(i-1,1) + lambda;
        end
    end
    for j = 2:M
        if abs(1-j) <= w
            D(1,j) = sum((X(1,:)-Y(j,:)).^2) + D(1,j-1) + lambda;
        end
    end
    for i = 2:N
        jStart = max(2, i-w); jEnd = min(M, i+w);
        for j = jStart:jEnd
            d = sum((X(i,:)-Y(j,:)).^2);
            D(i,j) = d + min([D(i-1,j-1), D(i-1,j)+lambda, D(i,j-1)+lambda]);
        end
    end
    i = N; j = M; path = [i, j];
    while ~(i==1 && j==1)
        candidates = [inf, inf, inf];
        if i>1 && j>1; candidates(1) = D(i-1,j-1); end
        if i>1; candidates(2) = D(i-1,j)+lambda; end
        if j>1; candidates(3) = D(i,j-1)+lambda; end
        [~, k] = min(candidates);
        if k==1; i=i-1; j=j-1; elseif k==2; i=i-1; else; j=j-1; end
        path = [[i,j]; path];
    end
end

function X_aligned = warp_to_ref_axis(X, path, M)
    C = size(X,2); X_aligned = zeros(M, C);
    for j = 1:M
        iList = path(path(:,2)==j, 1);
        if isempty(iList)
            [~, k] = min(abs(path(:,2)-j));
            iList = path(k,1);
        end
        X_aligned(j,:) = mean(X(iList,:), 1);
    end
end
