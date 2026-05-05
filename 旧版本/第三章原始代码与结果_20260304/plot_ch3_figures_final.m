clc; clearvars; clear functions; close all;

%% -------------------------- ·�������Ŀ¼ ------------------------------
thisFile = mfilename('fullpath');
if isempty(thisFile)
    rootDir = pwd;
else
    rootDir = fileparts(thisFile);
end

rawDataFile   = fullfile(rootDir, 'processedVehicleData_3class_REAL (2).mat');
assetsDir     = fullfile(rootDir, 'ch3_assets_plus_matlab');
matlabDataDir = fullfile(assetsDir, 'matlab_data');
outDir        = fullfile(rootDir, 'figures');
if ~exist(outDir, 'dir'); mkdir(outDir); end

assert(exist(rawDataFile, 'file')==2, 'δ�ҵ�ԭʼ�����ļ���%s', rawDataFile);

%% -------------------------- ȫ�ֻ�ͼ��� --------------------------------
fontCN = pick_chinese_font();
fprintf('ѡ������ fontCN = %s\n', fontCN);
set_plot_defaults(fontCN);

%% -------------------------- �ؼ�������������һ�£� -----------------------
fs      = 50;     % ������ (Hz)
N0      = 10;     % ���߹��ƴ��ڳ��ȣ�����������
wR      = 0.15;   % Sakoe--Chiba ���ڱ���
lambda  = 0.05;   % DTW �����ͷ���
L       = 176;    % ����ͳһ��������

classNames = {'С�ͳ�','���ͳ�','���ͳ�'};

fprintf('=== ������ͼƬ���ɿ�ʼ ===\n');
fprintf('rootDir = %s\n', rootDir);
fprintf('outDir  = %s\n\n', outDir);

%% -------------------------- ����ԭʼ���� --------------------------------
S = load(rawDataFile);
ProcessedData = S.ProcessedData;
targetLength  = S.targetLength;

%% ========================================================================
%  1) ????????????????????
%% ========================================================================
fprintf('[1/6] ?????????????...\n');

[repIdx, repLen, repScore] = select_grouped_comparison_samples(ProcessedData, targetLength, fs, N0, L);

for c = 1:3
    fprintf('  %s?????1-based??%d, %d, %d\n', classNames{c}, repIdx(c,1), repIdx(c,2), repIdx(c,3));
    fprintf('  %s???? N?        %d, %d, %d\n', classNames{c}, repLen(c,1), repLen(c,2), repLen(c,3));
end

cropCfg.frac   = 0.10;
cropCfg.margin = 5;
axisColors = lines(3);

fig = figure('Color','w','Position',[100 100 1180 460]);
ax = axes('Parent', fig);
hold(ax, 'on');
grid(ax, 'on');
box(ax, 'on');
set(ax, 'Layer', 'top');
xlabel(ax, '?? / s');
ylabel(ax, '???? / nT');

waveformExamples = struct();
tCursor   = 0;
gapEvent  = 0.18;
gapGroup  = 0.70;
evtNo     = 0;
yMinAll   = inf;
yMaxAll   = -inf;
groupInfo = struct('x0', {}, 'x1', {}, 'y0', {}, 'y1', {}, 'name', {});

for c = 1:3
    groupStart = tCursor;
    groupYMin = inf;
    groupYMax = -inf;

    for j = 1:3
        evtNo = evtNo + 1;
        idx = repIdx(c, j);
        N   = repLen(c, j);

        [~, dB, b, t] = extract_event(ProcessedData{c}{idx}, N, fs, N0);
        useIdx = crop_by_mag(b, cropCfg.frac, cropCfg.margin);

        t_use  = t(useIdx) - t(useIdx(1));
        dB_use = dB(useIdx, :);
        b_use  = b(useIdx);
        t_plot = tCursor + t_use;

        h1 = plot(ax, t_plot, dB_use(:,1), 'Color', axisColors(1,:));
        h2 = plot(ax, t_plot, dB_use(:,2), 'Color', axisColors(2,:));
        h3 = plot(ax, t_plot, dB_use(:,3), 'Color', axisColors(3,:));

        if evtNo == 1
            legend(ax, [h1 h2 h3], {'X?','Y?','Z?'}, 'Location','northwest', 'Box','off');
        end

        waveformExamples(evtNo).className = classNames{c};
        waveformExamples(evtNo).classID    = c;
        waveformExamples(evtNo).eventID    = evtNo;
        waveformExamples(evtNo).index      = idx;
        waveformExamples(evtNo).N          = N;
        waveformExamples(evtNo).score      = repScore(c, j);
        waveformExamples(evtNo).t_local    = t_use;
        waveformExamples(evtNo).t_plot     = t_plot;
        waveformExamples(evtNo).dB         = dB_use;
        waveformExamples(evtNo).b          = b_use;

        T = table(t_use, t_plot, dB_use(:,1), dB_use(:,2), dB_use(:,3), b_use, ...
            'VariableNames', {'t_local_s','t_plot_s','dBx_nT','dBy_nT','dBz_nT','b_nT'});
        writetable(T, fullfile(outDir, sprintf('waveform_group_class%d_evt%d_idx%d.csv', c, evtNo, idx)));

        localTop = max(dB_use(:));
        localBot = min(dB_use(:));
        localX   = mean([t_plot(1), t_plot(end)]);

        yMinAll = min(yMinAll, localBot);
        yMaxAll = max(yMaxAll, localTop);
        groupYMin = min(groupYMin, localBot);
        groupYMax = max(groupYMax, localTop);

        waveformExamples(evtNo).labelX = localX;
        waveformExamples(evtNo).labelY = localTop;

        tCursor = t_plot(end) + gapEvent;
    end

    groupEnd = tCursor - gapEvent;
    groupInfo(c).x0 = groupStart;
    groupInfo(c).x1 = groupEnd;
    groupInfo(c).y0 = groupYMin;
    groupInfo(c).y1 = groupYMax;
    groupInfo(c).name = classNames{c};

    tCursor = groupEnd + gapGroup;
end

yRange = yMaxAll - yMinAll;
if yRange < eps
    yRange = max(1, abs(yMaxAll));
end
yPad = 0.18 * yRange;
yBoxPad = 0.08 * yRange;
yLow = yMinAll - yPad;
yHigh = yMaxAll + yPad;
xlim(ax, [0, tCursor - gapGroup + 0.25]);
ylim(ax, [yLow, yHigh]);

for c = 1:3
    x0 = groupInfo(c).x0 - 0.08;
    x1 = groupInfo(c).x1 + 0.08;
    y0 = groupInfo(c).y0 - yBoxPad;
    y1 = groupInfo(c).y1 + yBoxPad;
    rectangle(ax, 'Position', [x0, y0, x1-x0, y1-y0], ...
        'LineStyle', ':', 'LineWidth', 1.2, 'EdgeColor', [0.35 0.35 0.35]);
    text(ax, mean([x0, x1]), yLow + 0.10 * yRange, groupInfo(c).name, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontSize', 14);
end

for i = 1:numel(waveformExamples)
    labelY = min(waveformExamples(i).labelY + 0.07 * yRange, yHigh - 0.05 * yRange);
    text(ax, waveformExamples(i).labelX, labelY, sprintf('%d', waveformExamples(i).eventID), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'FontWeight', 'bold', 'FontSize', 13);
end

pngOut = fullfile(outDir, 'ch3_waveform_by_class.png');
save_png(fig, pngOut, 300);
close(fig);
save(fullfile(outDir, 'ch3_waveform_examples.mat'), 'waveformExamples', 'cropCfg', 'fs', 'N0');

repoRoot = fileparts(fileparts(fileparts(rootDir)));
thesisImageDir = fullfile(repoRoot, 'images');
if exist(thesisImageDir, 'dir') == 7
    copyfile(pngOut, fullfile(thesisImageDir, 'ch3_waveform_by_class.png'));
end

%% ========================================================================
%  2) ���ٱ仯�����ʱ����������ͼ
% ========================================================================
fprintf('[2/6] ����ʱ����������ͼ...\n');

cStretch = 2;
lenVec = double(targetLength{cStretch}(:));

q = 0.15;
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

xlabel('ʱ�� / s');
ylabel('ģֵ���� b[n] / nT');
legend({sprintf('��ʱ��������N=%d��', lenVec(idxShort)), ...
        sprintf('��ʱ��������N=%d��', lenVec(idxLong))}, 'Location','best');
title(sprintf('���ٱ仯�����ʱ������ʾ�⣨%s��', classNames{cStretch}), 'FontWeight','normal');

save_png(fig, fullfile(outDir, 'fig_motivation_speed.png'), 300);
close(fig);

%% ========================================================================
%  3) DTW ����ʾ��ͼ��չʾ ��Bz��
% ========================================================================
fprintf('[3/6] ���� DTW ����ʾ��ͼ...\n');

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
w = max(floor(wR * max(N,M)), abs(N-M));

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
xlabel(ax1,'ʱ�� / s');
ylabel(ax1,'\Delta B_z[n] / nT');
legend(ax1, {sprintf('����A��N=%d��', N), sprintf('����B��N=%d��', M)}, 'Location','best');
title(ax1, '����ǰ��ʱ��������ֲ���λ��', 'FontWeight','normal');

ax2 = nexttile(tl,2);
plot(ax2, tRef, zY, 'LineWidth', 1.6); hold(ax2,'on');
plot(ax2, tRef, zX_aligned, 'LineWidth', 1.6);
grid(ax2,'on'); box(ax2,'on');
xlabel(ax2,'�ο�ʱ���� / s');
ylabel(ax2,'\Delta B_z[n] / nT');
legend(ax2, {'�ο�����B','����������A'}, 'Location','best');
title(ax2, 'DTW �����ӳ�䵽�ο�ʱ���ᣩ', 'FontWeight','normal');

save_png(fig, fullfile(outDir, 'fig_motivation_dtw_align_z.png'), 300);
close(fig);

%% ========================================================================
%  4) �������󣨸Ĳ��֣����� confusionchart ���
% ========================================================================
fprintf('[4/6] ���ɻ�������CNN Baseline��...\n');
make_confusion_chart( ...
    fullfile(matlabDataDir,'cm_cnn_baseline_data.mat'), ...
    fullfile(outDir,'cm_cnn_baseline.png'), ...
    '���߻�������CNN Baseline��', fontCN);

fprintf('[5/6] ���ɻ�������DTW-MultiTemplate-CNN��...\n');
make_confusion_chart( ...
    fullfile(matlabDataDir,'cm_dtw_multi_cnn_data.mat'), ...
    fullfile(outDir,'cm_dtw_multi_cnn.png'), ...
    '���߻�������DTW-MultiTemplate-CNN��K_{MT}^*=4��', fontCN);

%% ========================================================================
%  5) K_MT ȡֵ���ڣ��� Python ���0-1���������񣻱��� K* ���ߣ�
% ========================================================================
fprintf('[6/6] ���� K_{MT} ɨ������...\n');

D = load(fullfile(matlabDataDir, 'ablation_K_data.mat'));

% --- �����ֶ�����Python=K����ɰ�=K_list ---
if isfield(D,'K')
    KMT_list = double(D.K(:));
elseif isfield(D,'K_list')
    KMT_list = double(D.K_list(:));
else
    error('ablation_K_data.mat ȱ�� K / K_list �ֶ�');
end

valF1  = double(D.val_f1(:));
testF1 = double(D.test_f1(:));

% --- ͳһ�� 0~1������ǰٷ��ƾͳ��� 100��---
if max(valF1)  > 1.2, valF1  = valF1  / 100; end
if max(testF1) > 1.2, testF1 = testF1 / 100; end

% --- K*�����ȶ��ļ���������֤�����ֵѡ ---
if isfield(D,'K_best')
    KMT_best = double(D.K_best(1));
elseif isfield(D,'best_k')
    KMT_best = double(D.best_k(1));
else
    [~, idBest] = max(valF1);
    KMT_best = KMT_list(idBest);
end

fig = figure('Color','w','Position',[100 100 900 520]);

% Ϊ�������㡰�ڶ���ͼ������=Test����=Val
hTest = plot(KMT_list, testF1, '-o', 'LineWidth', 2.0, 'MarkerSize', 8, ...
    'MarkerFaceColor','auto'); hold on;
hVal  = plot(KMT_list, valF1,  '-s', 'LineWidth', 2.0, 'MarkerSize', 8, ...
    'MarkerFaceColor','auto');

% ��������Ҫ�� K* ��ֱ����
hBest = xline(KMT_best, '--', 'LineWidth', 1.6);

grid on; box on;
ax = gca;
ax.GridLineStyle = '--';
ax.GridAlpha     = 0.40;
ax.FontName      = fontCN;
ax.FontSize      = 14;

xticks(KMT_list.');
xlim([min(KMT_list)-0.2, max(KMT_list)+0.2]);

ylim([0.0, 1.0]);
yticks(0:0.2:1.0);

xlabel('ÿ��ģ���� K_{MT}');
ylabel('Macro-F1');
title('K_{MT} ȡֵ����', 'FontWeight','normal');

legend([hTest,hVal,hBest], ...
    {'���Լ� Macro-F1','��֤�� Macro-F1', sprintf('K_{MT}^*=%d', KMT_best)}, ...
    'Location','southwest');

save_png(fig, fullfile(outDir, 'ablation_K.png'), 300);
close(fig);

fprintf('\n=== ȫ����� ===\n');
fprintf('���Ŀ¼��%s\n', outDir);

%% =============================== ������ ================================

function fontCN = pick_chinese_font()
    cand = { ...
        'Microsoft YaHei','Microsoft YaHei UI','΢���ź�', ...
        'SimHei','����', ...
        'SimSun','����', ...
        'STSong','��������', ...
        'Arial Unicode MS' ...
    };
    fs = listfonts;
    for i = 1:numel(cand)
        if any(strcmpi(fs, cand{i}))
            fontCN = cand{i};
            return;
        end
    end
    fsLower = lower(string(fs));
    idx = find(contains(fsLower, "yahei") | contains(string(fs), "΢���ź�"), 1);
    if ~isempty(idx), fontCN = fs{idx}; return; end
    idx = find(contains(string(fs), "��") | contains(fsLower, "simsun"), 1);
    if ~isempty(idx), fontCN = fs{idx}; return; end
    idx = find(contains(string(fs), "��") | contains(fsLower, "simhei"), 1);
    if ~isempty(idx), fontCN = fs{idx}; return; end
    fontCN = fs{1};
end

function set_plot_defaults(fontCN)
    set(groot, 'defaultFigureColor', 'w');
    set(groot, 'defaultAxesFontName', fontCN);
    set(groot, 'defaultTextFontName', fontCN);
    set(groot, 'defaultAxesFontSize', 12);
    set(groot, 'defaultAxesLineWidth', 1.0);
    set(groot, 'defaultLineLineWidth', 1.5);
    set(groot, 'defaultAxesBox', 'on');
    set(groot, 'defaultTextInterpreter', 'tex');
    set(groot, 'defaultAxesTickLabelInterpreter', 'tex');
    set(groot, 'defaultLegendInterpreter', 'tex');
end

function save_png(figHandle, filePath, dpi)
    if exist('exportgraphics','file') == 2
        exportgraphics(figHandle, filePath, 'Resolution', dpi);
    else
        print(figHandle, filePath, '-dpng', sprintf('-r%d', dpi));
    end
end

function [B, dB, b, t] = extract_event(Bpad, N, fs, N0)
    N = double(N);
    B = double(Bpad(1:N, :));
    n0 = min(N0, N);
    B0 = mean(B(1:n0, :), 1);
    dB = B - B0;
    b  = sqrt(sum(dB.^2, 2));
    t  = (0:N-1)' / fs;
end

function idxUse = crop_by_mag(b, frac, margin)
    if isempty(b), idxUse = []; return; end
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
    x = x(:);
    N = numel(x);
    if N == L
        y = x; return;
    end
    t1 = linspace(0, 1, N);
    t2 = linspace(0, 1, L);
    y  = interp1(t1, x, t2, 'linear');
    y  = y(:);
end

function qv = pct(x, p)
    if exist('prctile','file') == 2
        qv = prctile(x, p);
        return;
    end
    x = sort(double(x(:)));
    if isempty(x), qv = NaN; return; end
    pos = 1 + (numel(x)-1) * (p/100);
    lo = floor(pos); hi = ceil(pos);
    if lo == hi
        qv = x(lo);
    else
        qv = x(lo) + (pos-lo)*(x(hi)-x(lo));
    end
end

function [repIdx, repLen, repScore] = select_representative_samples(ProcessedData, targetLength, fs, N0, L)
    nClass = numel(ProcessedData);
    repIdx   = zeros(1, nClass);
    repLen   = zeros(1, nClass);
    repScore = zeros(1, nClass);

    for c = 1:nClass
        lenVec = double(targetLength{c}(:));
        nSamp  = numel(lenVec);

        feat   = zeros(nSamp, 3*L);
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

        eThr  = pct(energy, 60);
        lenLo = pct(lenVec, 25);
        lenHi = pct(lenVec, 75);

        mask = (energy >= eThr) & (lenVec >= lenLo) & (lenVec <= lenHi);
        if sum(mask) < 5
            mask = (energy >= eThr);
        end

        corrScore(~mask) = -inf;
        [bestScore, bestIdx] = max(corrScore);

        repIdx(c)   = bestIdx;
        repLen(c)   = lenVec(bestIdx);
        repScore(c) = bestScore;
    end
end

function [repIdx, repLen, repScore] = select_grouped_comparison_samples(ProcessedData, targetLength, fs, N0, L)
    nClass = numel(ProcessedData);
    repIdx   = zeros(nClass, 3);
    repLen   = zeros(nClass, 3);
    repScore = zeros(nClass, 3);

    featCell   = cell(nClass, 1);
    energyCell = cell(nClass, 1);
    ampCell    = cell(nClass, 1);
    lenCell    = cell(nClass, 1);

    tmpl = zeros(nClass, 3 * L);

    for c = 1:nClass
        lenVec = double(targetLength{c}(:));
        nSamp  = numel(lenVec);
        feat   = zeros(nSamp, 3 * L);
        energy = zeros(nSamp, 1);
        ampVal = zeros(nSamp, 1);

        for i = 1:nSamp
            [~, dB, b, ~] = extract_event(ProcessedData{c}{i}, lenVec(i), fs, N0);
            energy(i) = sum(b.^2);
            ampVal(i) = max(abs(dB(:)));

            Xi = zeros(L, 3);
            for k = 1:3
                xk = resample_linear(dB(:,k), L);
                xk = (xk - mean(xk)) / (std(xk) + eps);
                Xi(:,k) = xk;
            end
            feat(i,:) = reshape(Xi, 1, []);
        end

        featCell{c} = feat;
        energyCell{c} = energy;
        ampCell{c} = ampVal;
        lenCell{c} = lenVec;

        tmpl(c,:) = mean(feat, 1);
        tmpl(c,:) = (tmpl(c,:) - mean(tmpl(c,:))) / (std(tmpl(c,:)) + eps);
    end

    qList = [30, 50, 70];

    for c = 1:nClass
        feat   = featCell{c};
        energy = energyCell{c};
        ampVal = ampCell{c};
        lenVec = lenCell{c};
        nSamp  = size(feat, 1);

        ownScore = (feat * tmpl(c,:).') / (3 * L);
        otherScore = -inf(nSamp, 1);
        for oc = 1:nClass
            if oc == c
                continue;
            end
            ocScore = (feat * tmpl(oc,:).') / (3 * L);
            otherScore = max(otherScore, ocScore);
        end

        marginScore = ownScore - otherScore;
        energyScore = scale01(log(energy + 1));
        ampScore    = scale01(ampVal);
        baseScore   = ownScore + 0.80 * marginScore + 0.10 * energyScore + 0.10 * ampScore;

        mask = (energy >= pct(energy, 40)) & (ampVal >= pct(ampVal, 40));
        if sum(mask) < 9
            mask = (energy >= pct(energy, 25));
        end
        if sum(mask) < 6
            mask = true(size(mask));
        end

        used = false(nSamp, 1);
        lenStd = std(lenVec) + eps;

        for j = 1:3
            lenTarget = pct(lenVec, qList(j));
            lenScore = -abs(lenVec - lenTarget) / lenStd;
            totalScore = baseScore + 0.20 * lenScore;
            totalScore(~mask) = -inf;
            totalScore(used) = -inf;

            [bestScore, bestIdx] = max(totalScore);
            if ~isfinite(bestScore)
                fallbackScore = baseScore + 0.10 * lenScore;
                fallbackScore(used) = -inf;
                [bestScore, bestIdx] = max(fallbackScore);
            end

            used(bestIdx) = true;
            repIdx(c, j) = bestIdx;
            repLen(c, j) = lenVec(bestIdx);
            repScore(c, j) = bestScore;
        end
    end
end

function y = scale01(x)
    x = double(x(:));
    lo = min(x);
    hi = max(x);
    if ~isfinite(lo) || ~isfinite(hi) || (hi - lo) < eps
        y = 0.5 * ones(size(x));
    else
        y = (x - lo) / (hi - lo);
    end
end

function idx = pick_max_energy(classCell, lenVec, idxPool, fs, N0)
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

%% ===================== ���������ֶ����֣�����ԭ confusionchart�� =====================

function make_confusion_chart(matFile, outPng, titleStr, fontCN)
    assert(exist(matFile,'file')==2, 'ȱ�ٻ����������ݣ�%s', matFile);
    D = load(matFile);

    cm = double(D.cm);
    labels = to_cellstr_row(D.labels);

    desired = {'С�ͳ�','���ͳ�','���ͳ�'};
    [cm, labels] = reorder_cm_labels(cm, labels, desired);

    n = size(cm,1);
    assert(n==3 && size(cm,2)==3, '��ǰ���ư�������ʵ�֣�����⵽ cm=%dx%d', n, size(cm,2));

    % ��/�л��ܣ������ٷֱȣ�0% ����գ�
    tp = diag(cm);
    rowSum = sum(cm,2);
    colSum = sum(cm,1).';

    rowCorrect = zeros(n,1);
    colCorrect = zeros(n,1);
    for i = 1:n
        if rowSum(i) > 0
            rowCorrect(i) = round(100 * tp(i) / rowSum(i));
        else
            rowCorrect(i) = 0;
        end
        if colSum(i) > 0
            colCorrect(i) = round(100 * tp(i) / colSum(i));
        else
            colCorrect(i) = 0;
        end
    end
    rowWrong = 100 - rowCorrect;
    colWrong = 100 - colCorrect;

    Prow = [rowCorrect, rowWrong];          % 3x2���Ҳࣩ
    Pcol = [colCorrect.'; colWrong.'];      % 2x3���ײ���

    cmap = confusion_like_cmap(256);

    fig = figure('Color','w','Position',[100 100 950 720]);

    % ---- �ֶ����֣��� confusionchart �ӽ����ؼ���----
    posMain = [0.10 0.34 0.56 0.56];   % [x y w h]
    posRow  = [0.70 0.34 0.22 0.56];
    posCol  = [0.10 0.10 0.56 0.18];

    axMain = axes('Parent', fig, 'Position', posMain);
    axRow  = axes('Parent', fig, 'Position', posRow);
    axCol  = axes('Parent', fig, 'Position', posCol);

    % �����󣺱��ַ���۸�
    cm_plot_matrix(axMain, cm, cmap, [0 max(cm(:))], 'count',   fontCN, true);
    axMain.XTick = [];                         % ��������ʾ x ���
    axMain.YTickLabel = labels;
    ylabel(axMain, '��ʵ��');

    % �л��ܣ�����������������ǿ�Ʒ��񣬱�����ף�
    cm_plot_matrix(axRow, Prow, cmap, [0 100], 'percent', fontCN, false);
    axRow.XTick = []; axRow.YTick = [];        % ����ʾ�κο̶�

    % �л��ܣ�����������������ǿ�Ʒ���
    cm_plot_matrix(axCol, Pcol, cmap, [0 100], 'percent', fontCN, false);
    axCol.YTick = [];
    axCol.XTickLabel = labels;
    xlabel(axCol, 'Ԥ����');

    sgtitle(titleStr, 'FontName', fontCN, 'FontWeight','normal', 'Interpreter','tex');

    save_png(fig, outPng, 300);
    close(fig);
end

function labels = to_cellstr_row(x)
    if iscell(x)
        labels = x(:).';
    elseif isstring(x)
        labels = cellstr(x(:)).';
    elseif ischar(x)
        labels = cellstr(x).';
    else
        labels = cellstr(string(x(:))).';
    end
end

function s = normlbl(s)
    s = string(s);
    s = regexprep(s, '\s+', '');
    s = strrep(s, char(160), '');
    s = strrep(s, char(12288), '');
end

function [cm2, labels2] = reorder_cm_labels(cm, labels, desired)
    lbl = normlbl(labels);
    des = normlbl(desired);

    perm = zeros(1, numel(desired));
    for k = 1:numel(desired)
        idx = find(lbl == des(k), 1);
        if isempty(idx)
            idx = find(contains(lbl, des(k)), 1);
        end
        if isempty(idx), perm(k) = 0; else, perm(k) = idx; end
    end

    if any(perm==0) || numel(unique(perm)) < numel(perm)
        warning('labels �޷��ȶ�ƥ�䵽 {С�ͳ�,���ͳ�,���ͳ�}��������ԭ˳��labels=%s', strjoin(string(labels), ','));
        cm2 = cm;
        labels2 = labels;
        return;
    end

    cm2 = cm(perm, perm);
    labels2 = desired;
end

function cmap = confusion_like_cmap(n)
    cLow  = [0.9569 0.8353 0.8078]; % ǳ��
    cHigh = [0.0000 0.4470 0.7410]; % ��
    cmap = [linspace(cLow(1), cHigh(1), n).', ...
            linspace(cLow(2), cHigh(2), n).', ...
            linspace(cLow(3), cHigh(3), n).'];
end

function cm_plot_matrix(ax, M, cmap, clim, mode, fontCN, squareCells)
    imagesc(ax, M);
    colormap(ax, cmap);
    caxis(ax, clim);

    [R,C] = size(M);
    set(ax, 'YDir','reverse');
    set(ax, 'XLim', [0.5 C+0.5], 'YLim', [0.5 R+0.5]);
    set(ax, 'FontName', fontCN, 'FontSize', 12);
    set(ax, 'TickLength', [0 0]);
    box(ax,'on');

    if squareCells
        axis(ax, 'image');
    else
        axis(ax, 'normal');
    end

    ax.XTick = 1:C;
    ax.YTick = 1:R;

    hold(ax,'on');
    for x = 0.5:1:(C+0.5)
        plot(ax, [x x], [0.5 R+0.5], 'k-', 'LineWidth', 0.8);
    end
    for y = 0.5:1:(R+0.5)
        plot(ax, [0.5 C+0.5], [y y], 'k-', 'LineWidth', 0.8);
    end

    vmax = clim(2);
    for i = 1:R
        for j = 1:C
            v = M(i,j);
            if strcmp(mode,'percent')
                txt = sprintf('%d%%', round(v));
                useWhite = (v >= 60);
            else
                txt = sprintf('%d', round(v));
                useWhite = (vmax > 0) && (v >= 0.65*vmax);
            end
            if useWhite, tcolor = 'w'; else, tcolor = 'k'; end
            text(ax, j, i, txt, 'HorizontalAlignment','center', ...
                'VerticalAlignment','middle', 'FontName', fontCN, ...
                'FontSize', 12, 'Color', tcolor);
        end
    end
    hold(ax,'off');
end

%% ===================== DTW ��� =====================

function [path, D] = dtw_path_4ch(X, Y, w, lambda)
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

    i = N; j = M;
    path = [i, j];
    while ~(i==1 && j==1)
        cand = [inf, inf, inf];
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
    C = size(X,2);
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
