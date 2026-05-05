%% ========================================================================
%  第三章：车型三分类（单地磁事件切片）论文图片统一绘图脚本（MATLAB�?%
%  功能�?%    1) 生成三类车辆代表性三轴波形示例（用于论文“直观展示”类图片，可选）
%    2) 生成“车速伸缩动机”示意图
%    3) 生成“DTW 对齐示意图�?%    4) 生成离线 CNN Baseline �?DTW-MultiTemplate 的混淆矩阵（MATLAB confusionchart 风格�?%    5) 生成 K_MT 扫描曲线（验证集与测试集 Macro-F1�?%
%  依赖文件�?%    - processedVehicleData_3class_REAL (2).mat
%    - ch3_assets_plus_matlab/matlab_data/*.mat   (混淆矩阵�?K_MT 扫描数据)
%
%  输出文件（PNG�?00 DPI）：
%    figures/ch3_waveform_by_class.png           （可选：三类代表性三轴波形）
%    figures/fig_motivation_speed.png
%    figures/fig_motivation_dtw_align_z.png
%    figures/cm_cnn_baseline.png
%    figures/cm_dtw_multi_cnn.png
%    figures/ablation_K.png
%
%  说明：脚本中的符号命名与论文保持一致：
%    ΔB_x[n], ΔB_y[n], ΔB_z[n] 由原�?B_x,B_y,B_z 基线扣除得到�?%    b[n] = ||ΔB[n]||_2�?%    K 表示端侧特征数量；K_MT 表示每类模板数�?% ========================================================================

clc; clear; close all;

%% -------------------------- 路径与输出目�?------------------------------
thisFile = mfilename('fullpath');
if isempty(thisFile)
    rootDir = pwd;   % 直接在命令行运行脚本�?else
    rootDir = fileparts(thisFile);
end

rawDataFile   = fullfile(rootDir, 'processedVehicleData_3class_REAL (2).mat');
assetsDir     = fullfile(rootDir, 'ch3_assets_plus_matlab');
matlabDataDir = fullfile(assetsDir, 'matlab_data');
outDir        = fullfile(rootDir, 'figures');
if ~exist(outDir, 'dir'); mkdir(outDir); end

assert(exist(rawDataFile, 'file')==2, '未找到原始数据文件：%s', rawDataFile);

%% -------------------------- 全局绘图风格 --------------------------------
fontCN = pick_chinese_font();
fprintf('选用字体 fontCN = %s\n', fontCN);
set_plot_defaults(fontCN);

%% -------------------------- 关键参数（与论文一致） -----------------------
fs      = 50;     % 采样�?(Hz)
N0      = 10;     % 基线估计窗口长度（采样点数）
wR      = 0.15;   % Sakoe--Chiba 窗口比例
lambda  = 0.05;   % DTW 步进惩罚项（论文中的 \lambda�?L       = 176;    % 离线统一定长长度（与论文一致，用于“代表性样本选择”的对齐尺度�?
classNames = {'小型�?,'中型�?,'大型�?};

fprintf('=== 第三章图片生成开�?===\n');
fprintf('rootDir = %s\n', rootDir);
fprintf('outDir  = %s\n\n', outDir);

%% -------------------------- 载入原始数据 --------------------------------
S = load(rawDataFile);
ProcessedData = S.ProcessedData;   % 1x3 cell，每�?1xN cell，每样本�?[512x3]
targetLength  = S.targetLength;    % 1x3 cell，每�?1xN 长度

%% ========================================================================
%  1) 三类车辆代表性三轴波形（可选）
%     目的：给出三类车辆在三轴地磁扰动上的典型形态差异，用于直观展示�?%     选择策略（推荐）：在每一类内部，先以 b[n] 的形态“接近类均值模板”为准，
%     并辅以能量与长度分位约束，避免选择能量过低或明显异常的样本�?% ========================================================================

fprintf('[1/6] 生成三类代表性三轴波形图...\n');

[repIdx, repLen, repScore] = select_representative_samples( ...
    ProcessedData, targetLength, fs, N0, L);

fprintf('  代表性样本索引（1-based）：小型=%d，中�?%d，大�?%d\n', repIdx(1), repIdx(2), repIdx(3));
fprintf('  对应长度 N�?             小型=%d，中�?%d，大�?%d\n', repLen(1), repLen(2), repLen(3));

cropCfg.frac   = 0.10;   % 以峰�?10%% 作为有效段阈�?cropCfg.margin = 5;      % 前后各保�?margin 个采样点

fig = figure('Color','w','Position',[100 100 1100 320]);
tl = tiledlayout(fig, 1, 3, 'TileSpacing','compact', 'Padding','compact');

waveformExamples = struct();

axHandles = zeros(1, 3);
tMaxAll = 0;
yMaxAll = 0;

for c = 1:3
    idx = repIdx(c);
    N   = repLen(c);

    [~, dB, b, t] = extract_event(ProcessedData{c}{idx}, N, fs, N0);
    useIdx = crop_by_mag(b, cropCfg.frac, cropCfg.margin);

    t_use  = t(useIdx) - t(useIdx(1));
    dB_use = dB(useIdx, :);
    b_use  = b(useIdx);

    ax = nexttile(tl, c);
    axHandles(c) = ax;
    plot(ax, t_use, dB_use(:,1)); hold(ax,'on');
    plot(ax, t_use, dB_use(:,2));
    plot(ax, t_use, dB_use(:,3));

    grid(ax,'on'); box(ax,'on');
    xlabel(ax, '时间 / s');
    if c==1
        ylabel(ax, '磁场扰动 / nT');
    end
    legend(ax, {'X�?,'Y�?,'Z�?}, 'Location','best');

    tMaxAll = max(tMaxAll, max(t_use));
    yMaxAll = max(yMaxAll, max(abs(dB_use(:))));

    title(ax, classNames{c});

    waveformExamples(c).className = classNames{c};
    waveformExamples(c).index     = idx;
    waveformExamples(c).N         = N;
    waveformExamples(c).score     = repScore(c);
    waveformExamples(c).t         = t_use;
    waveformExamples(c).dB        = dB_use;
    waveformExamples(c).b         = b_use;

    % 同时导出 CSV，便于你�?MATLAB 中二次排版或微调
    T = table(t_use, dB_use(:,1), dB_use(:,2), dB_use(:,3), b_use, ...
        'VariableNames', {'t_s','dBx_nT','dBy_nT','dBz_nT','b_nT'});
    csvName = sprintf('waveform_class%d_idx%d.csv', c, idx);
    writetable(T, fullfile(outDir, csvName));
end

% 统一设置所有子图的 x 轴和 y 轴范�?if yMaxAll > 0
    for c = 1:3
        xlim(axHandles(c), [0, tMaxAll]);
        ylim(axHandles(c), 1.10 * yMaxAll * [-1, 1]);
    end
end

save_png(fig, fullfile(outDir, 'ch3_waveform_by_class.png'), 300);
close(fig);

save(fullfile(outDir, 'ch3_waveform_examples.mat'), 'waveformExamples', 'cropCfg', 'fs', 'N0');

%% ========================================================================
%  2) 车速变化引起的时间伸缩动机图（fig_motivation_speed.png�?% ========================================================================

fprintf('[2/6] 生成时间伸缩动机�?..\n');

% 为保证“同类不同速度”差异明显，默认使用中型车类（可按需改为 1 �?3�?cStretch = 2;
lenVec = double(targetLength{cStretch}(:));

% 在长度分布的两端分别挑选一个样本：短时长与长时�?q = 0.15;
idxSort = sortrows([(1:numel(lenVec))', lenVec], 2);
nAll = size(idxSort,1);
shortPool = idxSort(1: max(1, round(q*nAll)), 1);
longPool  = idxSort(max(1, round((1-q)*nAll)) : nAll, 1);

idxShort = pick_max_energy(ProcessedData{cStretch}, lenVec, shortPool, fs, N0);
idxLong  = pick_max_energy(ProcessedData{cStretch}, lenVec, longPool,  fs, N0);

[~, ~, b_s, t_s] = extract_event(ProcessedData{cStretch}{idxShort}, lenVec(idxShort), fs, N0);
[~, ~, b_l, t_l] = extract_event(ProcessedData{cStretch}{idxLong},  lenVec(idxLong),  fs, N0);

fig = figure('Color','w','Position',[100 100 900 420]);
plot(t_s, b_s, 'LineWidth', 1.6); hold on;
plot(t_l, b_l, 'LineWidth', 1.6);
grid on; box on;

xlabel('时间 / s');
ylabel('模值序�?b[n] / nT');
legend({sprintf('短时长样本（N=%d�?, lenVec(idxShort)), ...
        sprintf('长时长样本（N=%d�?, lenVec(idxLong))}, ...
        'Location','best');

title(sprintf('车速变化引起的时间伸缩示意�?s�?, classNames{cStretch}));

save_png(fig, fullfile(outDir, 'fig_motivation_speed.png'), 300);
close(fig);

%% ========================================================================
%  3) DTW 对齐示意图（fig_motivation_dtw_align_z.png�?%     说明：DTW 路径计算使用四通道联合距离，但示意图仅展示 ΔB_z[n]�?% ========================================================================

fprintf('[3/6] 生成 DTW 对齐示意�?..\n');

% 重新提取 dB，用�?4 通道 DTW
[~, dB_s, ~, ~] = extract_event(ProcessedData{cStretch}{idxShort}, lenVec(idxShort), fs, N0);
[~, dB_l, ~, ~] = extract_event(ProcessedData{cStretch}{idxLong},  lenVec(idxLong),  fs, N0);

b_s = sqrt(sum(dB_s.^2, 2));
b_l = sqrt(sum(dB_l.^2, 2));

X4 = [dB_s, b_s];
Y4 = [dB_l, b_l];

zX = dB_s(:,3);
zY = dB_l(:,3);

N = size(X4,1);
M = size(Y4,1);

% 与论文口径一致：采用 Sakoe--Chiba 约束；这里用 max(N,M) 给出可行的窗口宽�?w = max(floor(wR * max(N,M)), abs(N-M));

[path, ~] = dtw_path_4ch(X4, Y4, w, lambda);
X4_aligned = warp_to_ref_axis(X4, path, M);
zX_aligned = X4_aligned(:,3);

tRef = (0:M-1)'/fs;

fig = figure('Color','w','Position',[100 100 900 600]);
tl = tiledlayout(fig, 2, 1, 'TileSpacing','compact', 'Padding','compact');

ax1 = nexttile(tl,1);
plot(ax1, (0:N-1)'/fs, zX, 'LineWidth', 1.6); hold(ax1,'on');
plot(ax1, (0:M-1)'/fs, zY, 'LineWidth', 1.6);
grid(ax1,'on'); box(ax1,'on');
xlabel(ax1,'时间 / s');
ylabel(ax1,'\Delta B_z[n] / nT');
legend(ax1, {sprintf('样本A（N=%d�?, N), sprintf('样本B（N=%d�?, M)}, 'Location','best');
title(ax1, '对齐前（同类样本存在时间伸缩与局部错位）');

ax2 = nexttile(tl,2);
plot(ax2, tRef, zY, 'LineWidth', 1.6); hold(ax2,'on');
plot(ax2, tRef, zX_aligned, 'LineWidth', 1.6);
grid(ax2,'on'); box(ax2,'on');
xlabel(ax2,'参考时间轴 / s');
ylabel(ax2,'\Delta B_z[n] / nT');
legend(ax2, {'参考样本B','对齐后的样本A'}, 'Location','best');
title(ax2, 'DTW 对齐后（映射到参考时间轴�?);

save_png(fig, fullfile(outDir, 'fig_motivation_dtw_align_z.png'), 300);
close(fig);

%% ========================================================================
%  4) 混淆矩阵（MATLAB confusionchart 风格�?% ========================================================================

fprintf('[4/6] 生成混淆矩阵（CNN Baseline�?..\n');

make_confusion_chart( ...
    fullfile(matlabDataDir,'cm_cnn_baseline_data.mat'), ...
    fullfile(outDir,'cm_cnn_baseline.png'), ...
    '离线混淆矩阵（CNN Baseline�?, fontCN);

fprintf('[5/6] 生成混淆矩阵（DTW-MultiTemplate-CNN�?..\n');

make_confusion_chart( ...
    fullfile(matlabDataDir,'cm_dtw_multi_cnn_data.mat'), ...
    fullfile(outDir,'cm_dtw_multi_cnn.png'), ...
    '离线混淆矩阵（DTW-MultiTemplate-CNN，K_MT*=4�?, fontCN);

%% ========================================================================
%  5) K_MT 扫描曲线（ablation_K.png�?% ========================================================================

fprintf('[6/6] 生成 K_{MT} 扫描曲线...\n');

D = load(fullfile(matlabDataDir, 'ablation_K_data.mat'));
KMT_list = double(D.K_list(:));
valF1    = 100 * double(D.val_f1(:));
testF1   = 100 * double(D.test_f1(:));
KMT_best = double(D.K_best(1));

fig = figure('Color','w','Position',[100 100 900 520]);
h1 = plot(KMT_list, valF1, '-o', 'LineWidth', 1.8, 'MarkerSize', 7); hold on;
h2 = plot(KMT_list, testF1,'-s', 'LineWidth', 1.8, 'MarkerSize', 7);
h3 = xline(KMT_best, '--', 'LineWidth', 1.4);

grid on; box on;
xticks(KMT_list.');
xlim([min(KMT_list)-0.2, max(KMT_list)+0.2]);

xlabel('每类模板�?K_{MT}');
ylabel('Macro-F_1�?�?);
legend([h1,h2,h3], {'验证�?Macro-F_1','测试�?Macro-F_1',sprintf('K_{MT}^*=%d',KMT_best)}, 'Location','best');

title('K_{MT} 取值消融（验证集选择，测试集报告�?);

save_png(fig, fullfile(outDir, 'ablation_K.png'), 300);
close(fig);

fprintf('\n=== 全部完成 ===\n');
fprintf('输出目录�?s\n', outDir);

%% =============================== 函数�?================================

function fontCN = pick_chinese_font()
% 更稳的中文字体选择：同时兼容英文名与中文名
    cand = { ...
        'Microsoft YaHei','Microsoft YaHei UI','微软雅黑', ...
        'SimHei','黑体', ...
        'SimSun','宋体', ...
        'STSong','华文宋体', ...
        'Arial Unicode MS' ...
    };

    fs = listfonts;
    fontCN = '';

    % 精确匹配
    for i = 1:numel(cand)
        if any(strcmpi(fs, cand{i}))
            fontCN = cand{i};
            return;
        end
    end

    % 模糊匹配：优先找 YaHei/微软雅黑/宋体/黑体
    fsLower = lower(string(fs));
    idx = find(contains(fsLower, "yahei") | contains(string(fs), "微软雅黑"), 1);
    if ~isempty(idx), fontCN = fs{idx}; return; end

    idx = find(contains(string(fs), "�?) | contains(fsLower, "simsun"), 1);
    if ~isempty(idx), fontCN = fs{idx}; return; end

    idx = find(contains(string(fs), "�?) | contains(fsLower, "simhei"), 1);
    if ~isempty(idx), fontCN = fs{idx}; return; end

    % 最后兜底：别用 Arial（容易不支持中文），用系统列表第一�?    fontCN = fs{1};
end

function set_plot_defaults(fontCN)
% MATLAB 风格统一设置（尽量不引入"�?Python"的外观）
    set(groot, 'defaultFigureColor', 'w');
    set(groot, 'defaultAxesFontName', fontCN);
    set(groot, 'defaultTextFontName', fontCN);
    set(groot, 'defaultAxesFontSize', 12);
    set(groot, 'defaultAxesLineWidth', 1.0);
    set(groot, 'defaultLineLineWidth', 1.5);
    set(groot, 'defaultAxesBox', 'on');

    % 关键：统一解释器，保证 \Delta、下标等�?tex 解析
    set(groot, 'defaultTextInterpreter', 'tex');
    set(groot, 'defaultAxesTickLabelInterpreter', 'tex');
    set(groot, 'defaultLegendInterpreter', 'tex');
end

function save_png(figHandle, filePath, dpi)
% 优先使用 exportgraphics，保证分辨率与边界裁剪效�?% 参�?MathWorks 文档：exportgraphics 支持 Resolution 参数�?    if exist('exportgraphics','file') == 2
        exportgraphics(figHandle, filePath, 'Resolution', dpi);
    else
        print(figHandle, filePath, '-dpng', sprintf('-r%d', dpi));
    end
end

function [B, dB, b, t] = extract_event(Bpad, N, fs, N0)
% 提取事件有效段并进行基线扣除
% - Bpad: [512 x 3] padding 后的原始序列
% - N   : 有效长度
% - 基线：用事件片段开�?N0 点估计（与论文“到达前窗口”口径一致）
    N = double(N);
    B = double(Bpad(1:N, :));
    n0 = min(N0, N);
    B0 = mean(B(1:n0, :), 1);
    dB = B - B0;
    b  = sqrt(sum(dB.^2, 2));
    t  = (0:N-1)' / fs;
end

function idxUse = crop_by_mag(b, frac, margin)
% 用模值序�?b[n] 的相对阈值裁剪有效段，并在两侧保�?margin �?    if isempty(b)
        idxUse = [];
        return;
    end
    thr = frac * max(b);
    pos = find(b >= thr);
    if isempty(pos)
        idxUse = (1:numel(b)).';
        return;
    end
    s = max(1, pos(1) - margin);
    e = min(numel(b), pos(end) + margin);
    idxUse = (s:e).';
end

function y = resample_linear(x, L)
% 线性插值重采样到长�?L
    x = x(:);
    N = numel(x);
    if N == L
        y = x;
        return;
    end
    t1 = linspace(0, 1, N);
    t2 = linspace(0, 1, L);
    y  = interp1(t1, x, t2, 'linear');
    y  = y(:);
end

function [repIdx, repLen, repScore] = select_representative_samples(ProcessedData, targetLength, fs, N0, L)
% 选择每一类的代表性样本（用于波形示例图）�?%   1) 对每个样本提取三�?dB，各轴线性重采样到长�?L 并做 z-score�?%   2) 拼接�?3L 维特征向量，以类均值模板作为参考计算相关性；
%   3) 加入能量阈值与长度分位约束，避免过弱样本与明显离群样本�?%   4) 选择相关性最大的样本�?
    nClass = numel(ProcessedData);
    repIdx   = zeros(1, nClass);
    repLen   = zeros(1, nClass);
    repScore = zeros(1, nClass);

    for c = 1:nClass
        lenVec = double(targetLength{c}(:));
        nSamp  = numel(lenVec);

        feat = zeros(nSamp, 3*L);
        energy = zeros(nSamp, 1);

        for i = 1:nSamp
            [~, dB, b, ~] = extract_event(ProcessedData{c}{i}, lenVec(i), fs, N0);
            energy(i) = sum(b.^2);

            X = zeros(L, 3);
            for k = 1:3
                xk = resample_linear(dB(:,k), L);
                xk = (xk - mean(xk)) / (std(xk) + eps);
                X(:,k) = xk;
            end
            feat(i,:) = reshape(X, 1, []);
        end

        tmpl = mean(feat, 1);
        tmpl = (tmpl - mean(tmpl)) / (std(tmpl) + eps);

        corrScore = (feat * tmpl.') / (3*L);

        eThr = prctile(energy, 60);
        lenLo = prctile(lenVec, 25);
        lenHi = prctile(lenVec, 75);
        mask = (energy >= eThr) & (lenVec >= lenLo) & (lenVec <= lenHi);
        if sum(mask) < 5
            mask = (energy >= eThr);
        end

        corrScore(~mask) = -inf;
        [bestScore, bestIdx] = max(corrScore);

        repIdx(c) = bestIdx;
        repLen(c) = lenVec(bestIdx);
        repScore(c) = bestScore;
    end
end

function idx = pick_max_energy(classCell, lenVec, idxPool, fs, N0)
% 在给定索引池中选择 b[n] 能量最大的样本
    bestE = -inf;
    idx = idxPool(1);
    for ii = idxPool(:)'
        [~, ~, b, ~] = extract_event(classCell{ii}, lenVec(ii), fs, N0);
        E = sum(b.^2);
        if E > bestE
            bestE = E;
            idx = ii;
        end
    end
end

function make_confusion_chart(matFile, outPng, titleStr, fontCN)
% 目标：生成与 MATLAB confusionchart 类似的版式（主矩�?�?列汇总）
%      并在主矩阵中�?0 显式写出来，避免"空白像出图错�?�?
    assert(exist(matFile,'file')==2, '缺少混淆矩阵数据�?s', matFile);
    D = load(matFile);

    cm = double(D.cm);
    labelsRaw = D.labels;

    % labels 统一�?cellstr�?xN�?    if iscell(labelsRaw)
        labels = labelsRaw(:).';
    elseif isstring(labelsRaw)
        labels = cellstr(labelsRaw(:)).';
    else
        labels = cellstr(labelsRaw(:)).';
    end

    nClass = size(cm,1);

    fig = figure('Color','w','Position',[100 100 950 720]);

    % --- �?confusionchart 生成你图里那种版�?---
    cc = confusionchart(cm, labels);
    cc.Normalization = 'absolute';           % 主矩阵显示计�?    cc.RowSummary    = 'row-normalized';     % 右侧：每行正�?错误百分比（召回视角�?    cc.ColumnSummary = 'column-normalized';  % 底部：每列正�?错误百分比（精确视角�?
    cc.Title  = titleStr;
    cc.XLabel = '预测�?;
    cc.YLabel = '真实�?;

    % 字体（避免中文变方块�?    try
        cc.FontName = fontCN;
        cc.FontSize = 12;
    catch
    end

    % 让数字以整数显示（部分版本仍会对 0 隐藏，所以后面再兜底�?0�?    try
        cc.CellLabelFormat = '%d';
    catch
    end

    % --- 兜底：主矩阵中把 0 补出�?---
    ax = find_main_confusion_axes(fig, nClass);
    if ~isempty(ax)
        hold(ax,'on');
        for i = 1:nClass
            for j = 1:nClass
                if cm(i,j) == 0
                    text(ax, j, i, '0', ...
                        'HorizontalAlignment','center', ...
                        'VerticalAlignment','middle', ...
                        'FontName', fontCN, ...
                        'FontSize', 12, ...
                        'Color', [0 0 0]);
                end
            end
        end
        hold(ax,'off');
    end

    save_png(fig, outPng, 300);
    close(fig);
end

function ax = find_main_confusion_axes(fig, nClass)
% confusionchart 会产生多�?Axes（主矩阵、行汇总、列汇总）
% �?XTick �?YTick 都是 1:nClass"的规则找主矩阵轴
    ax = [];
    axs = findall(fig, 'Type', 'Axes');
    for k = 1:numel(axs)
        try
            if numel(axs(k).XTick) == nClass && numel(axs(k).YTick) == nClass
                ax = axs(k);
                return;
            end
        catch
        end
    end
end

function [path, D] = dtw_path_4ch(X, Y, w, lambda)
% 四通道 DTW（带 Sakoe--Chiba 约束与步进惩罚项�?%   D(i,j) = d(i,j) + min{ D(i-1,j-1), D(i-1,j)+lambda, D(i,j-1)+lambda }
%   d(i,j) = ||X(i,:)-Y(j,:)||_2^2

    N = size(X,1);
    M = size(Y,1);
    D = inf(N, M);

    D(1,1) = sum((X(1,:) - Y(1,:)).^2);

    for i = 2:N
        if abs(i-1) <= w
            d = sum((X(i,:) - Y(1,:)).^2);
            D(i,1) = d + D(i-1,1) + lambda;
        end
    end

    for j = 2:M
        if abs(1-j) <= w
            d = sum((X(1,:) - Y(j,:)).^2);
            D(1,j) = d + D(1,j-1) + lambda;
        end
    end

    for i = 2:N
        jStart = max(2, i - w);
        jEnd   = min(M, i + w);
        for j = jStart:jEnd
            d = sum((X(i,:) - Y(j,:)).^2);
            D(i,j) = d + min([ ...
                D(i-1,j-1), ...
                D(i-1,j) + lambda, ...
                D(i,  j-1) + lambda]);
        end
    end

    % 回溯获得路径
    i = N; j = M;
    path = [i, j];
    while ~(i==1 && j==1)
        cand = [inf, inf, inf];  % diag, up, left
        if i>1 && j>1; cand(1) = D(i-1,j-1); end
        if i>1;        cand(2) = D(i-1,j) + lambda; end
        if j>1;        cand(3) = D(i,j-1) + lambda; end
        [~, k] = min(cand);
        if k==1
            i=i-1; j=j-1;
        elseif k==2
            i=i-1;
        else
            j=j-1;
        end
        path = [[i,j]; path]; %#ok<AGROW>
    end
end

function X_aligned = warp_to_ref_axis(X, path, M)
% 将序�?X �?DTW 路径映射到参考轴（长�?M）：对多对一位置取均�?    C = size(X,2);
    X_aligned = zeros(M, C);
    for j = 1:M
        iList = path(path(:,2) == j, 1);
        if isempty(iList)
            [~, k] = min(abs(path(:,2) - j));
            iList = path(k,1);
        end
        X_aligned(j,:) = mean(X(iList,:), 1);
    end
end
