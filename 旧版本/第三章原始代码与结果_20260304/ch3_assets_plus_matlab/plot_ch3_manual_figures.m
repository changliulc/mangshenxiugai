%==========================================================================
% Chapter 3 (vehicle 3-class) : MATLAB plotting utilities (manual figures)
%
% Data file: processedVehicleData_3class_REAL (2).mat
%
% Consistent notation with thesis:
%   Bx,By,Bz           ->  B_x[n], B_y[n], B_z[n]  (raw field components)
%   dBx,dBy,dBz        ->  ΔB_x[n], ΔB_y[n], ΔB_z[n] (baseline removed)
%   b                  ->  b[n] = ||ΔB[n]||_2
%   fs                 ->  sampling rate (50 Hz)
%
% Outputs (place into thesis figures/):
%   figures/fig_motivation_speed.png
%   figures/fig_motivation_dtw_align_z.png
% Optional:
%   figures/ch3_waveform_3class.png
%==========================================================================
clear; clc;

%% -------------------------- configuration -------------------------------
fs  = 50;          % sampling frequency (Hz)
N0  = 10;          % baseline estimation points (use head+tail each N0 points)
wR  = 0.15;        % Sakoe--Chiba relative window ratio
lambda = 0.05;     % step penalty in DTW recursion (same symbol as thesis)

dataPath = 'processedVehicleData_3class_REAL (2).mat';   % adjust if needed
outDir   = fullfile(pwd, 'figures');
if ~exist(outDir, 'dir'); mkdir(outDir); end

%% ---------------------------- load data --------------------------------
S = load(dataPath);
ProcessedData = S.ProcessedData;      % 1x3 cell, each: 1xN cell, each cell: [512x3] padded
targetLength  = S.targetLength;       % 1x3 cell, each: [1xN] original length

classNames = {'小型车','中型车','大型车'};

% sanity check
assert(iscell(ProcessedData) && numel(ProcessedData)==3, 'ProcessedData format unexpected.');
assert(iscell(targetLength)  && numel(targetLength)==3,  'targetLength format unexpected.');

%% =======================================================================
%  Figure A (optional): three-class representative waveforms (b[n])
%  - Select one representative sample per class:
%    length near median, then pick the one with largest energy.
% =======================================================================
tolLen = 3;      % tolerance around median length (samples)
repIdx = zeros(1,3);
repLen = zeros(1,3);

for c = 1:3
    lenVec = double(targetLength{c}(:));
    medLen = round(median(lenVec));
    cand = find(abs(lenVec - medLen) <= tolLen);

    if isempty(cand)
        cand = 1:numel(lenVec);
    end

    bestE = -inf;
    bestI = cand(1);
    for ii = cand(:)'
        [~, ~, b, ~] = extract_event(ProcessedData{c}{ii}, lenVec(ii), fs, N0);
        E = sum(b.^2);       % energy on b[n]
        if E > bestE
            bestE = E;
            bestI = ii;
        end
    end

    repIdx(c) = bestI;
    repLen(c) = lenVec(bestI);
end

% plot
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
print(fig1, fullfile(outDir, 'ch3_waveform_3class.png'), '-dpng', '-r300');

%% =======================================================================
%  Figure 3.?.1: motivation - speed stretch (same class, different lengths)
%  Output name MUST match the thesis:
%    figures/fig_motivation_speed.png
% =======================================================================
% use medium class to show larger length variation
cStretch = 2;   % 1=small, 2=medium, 3=large
lenVec = double(targetLength{cStretch}(:));

% choose short sample from bottom q, long sample from top q (by length),
% within each quantile pick the one with largest energy.
q = 0.15;
idxSort = sortrows([(1:numel(lenVec))', lenVec], 2);  % [index, length]
nAll = size(idxSort,1);
shortPool = idxSort(1: max(1, round(q*nAll)), 1);
longPool  = idxSort(max(1, round((1-q)*nAll)) : nAll, 1);

idxShort = pick_max_energy(ProcessedData{cStretch}, lenVec, shortPool, fs, N0);
idxLong  = pick_max_energy(ProcessedData{cStretch}, lenVec, longPool,  fs, N0);

[~, dB_s, b_s, t_s] = extract_event(ProcessedData{cStretch}{idxShort}, lenVec(idxShort), fs, N0);
[~, dB_l, b_l, t_l] = extract_event(ProcessedData{cStretch}{idxLong},  lenVec(idxLong),  fs, N0);

fig2 = figure('Color','w','Position',[100 100 900 420]);
plot(t_s, b_s, 'LineWidth', 1.2); hold on;
plot(t_l, b_l, 'LineWidth', 1.2);
grid on;
xlabel('时间 / s');
ylabel('b[n]');
legend({sprintf('短时长样本（N=%d）', lenVec(idxShort)), ...
        sprintf('长时长样本（N=%d）', lenVec(idxLong))}, ...
        'Location','best');
title(sprintf('车速变化引起的时间伸缩示意（%s）', classNames{cStretch}));
print(fig2, fullfile(outDir, 'fig_motivation_speed.png'), '-dpng', '-r300');

%% =======================================================================
%  Figure 3.?.2: motivation - DTW alignment demo (plot z-channel before/after)
%  Output name MUST match the thesis:
%    figures/fig_motivation_dtw_align_z.png
% =======================================================================
% build 4-channel sequences for DTW distance, but visualize only ΔB_z
X4 = [dB_s, b_s];   % [N_short x 4]
Y4 = [dB_l, b_l];   % [N_long  x 4]
zX = dB_s(:,3);     % ΔB_z (short)
zY = dB_l(:,3);     % ΔB_z (long)

N = size(X4,1);
M = size(Y4,1);
w = max(floor(wR * max(N,M)), abs(N-M));  % ensure feasible constraint

[path, ~] = dtw_path_4ch(X4, Y4, w, lambda);    % path: [Lw x 2] (i,j)
X4_aligned = warp_to_ref_axis(X4, path, M);
zX_aligned = X4_aligned(:,3);
tRef = (0:M-1)'/fs;

fig3 = figure('Color','w','Position',[100 100 900 600]);

subplot(2,1,1);
plot((0:N-1)'/fs, zX, 'LineWidth', 1.2); hold on;
plot((0:M-1)'/fs, zY, 'LineWidth', 1.2);
grid on;
xlabel('时间 / s');
ylabel('\Delta B_z[n]');
title('对齐前（同类样本存在时间伸缩与局部错位）');
legend({sprintf('样本A（N=%d）', N), sprintf('样本B（N=%d）', M)}, 'Location','best');

subplot(2,1,2);
plot(tRef, zY, 'LineWidth', 1.2); hold on;
plot(tRef, zX_aligned, 'LineWidth', 1.2);
grid on;
xlabel('参考时间轴 / s');
ylabel('\Delta B_z[n]');
title('DTW 对齐后（映射到参考时间轴）');
legend({'参考样本B','对齐后的样本A'}, 'Location','best');

print(fig3, fullfile(outDir, 'fig_motivation_dtw_align_z.png'), '-dpng', '-r300');

%% save the selected indices for record (optional)
save(fullfile(outDir, 'motivation_selected_indices.mat'), ...
    'repIdx','repLen','cStretch','idxShort','idxLong','lenVec','fs','N0','wR','lambda');

disp('Done. Figures saved to:');
disp(outDir);

%% ============================= functions ===============================
function [B, dB, b, t] = extract_event(Bpad, N, fs, N0)
% Extract valid segment and apply baseline removal:
%   Bpad : [512 x 3] padded raw field
%   N    : valid length
    N = double(N);
    B = double(Bpad(1:N, :));                 % raw B_x,B_y,B_z
    n0 = min(N0, N);
    B0 = mean([B(1:n0,:); B(N-n0+1:N,:)], 1); % approximate baseline (head+tail)
    dB = B - B0;                              % ΔB_x, ΔB_y, ΔB_z
    b  = sqrt(sum(dB.^2, 2));                 % b[n]
    t  = (0:N-1)'/fs;
end

function idx = pick_max_energy(classCell, lenVec, idxPool, fs, N0)
% From a pool of indices, pick the one with maximum energy on b[n].
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

function [path, D] = dtw_path_4ch(X, Y, w, lambda)
% DTW with Sakoe--Chiba window and step penalty:
%   D(i,j) = d(i,j) + min{ D(i-1,j-1), D(i-1,j)+lambda, D(i,j-1)+lambda }.
% Local distance uses squared Euclidean distance in 4D.
%
% Inputs:
%   X: [N x 4], Y: [M x 4]
%   w: window half-width (integer)
%   lambda: step penalty (>=0)
%
% Outputs:
%   path: [Lw x 2] alignment path (i,j), 1-based
%   D: [N x M] accumulated cost matrix

    N = size(X,1);
    M = size(Y,1);
    D = inf(N, M);

    % initialize (1,1)
    D(1,1) = sum((X(1,:)-Y(1,:)).^2);

    % first column
    for i = 2:N
        if abs(i-1) <= w
            d = sum((X(i,:)-Y(1,:)).^2);
            D(i,1) = d + D(i-1,1) + lambda;
        end
    end

    % first row
    for j = 2:M
        if abs(1-j) <= w
            d = sum((X(1,:)-Y(j,:)).^2);
            D(1,j) = d + D(1,j-1) + lambda;
        end
    end

    % main DP
    for i = 2:N
        jStart = max(2, i - w);
        jEnd   = min(M, i + w);
        for j = jStart:jEnd
            d = sum((X(i,:)-Y(j,:)).^2);
            D(i,j) = d + min([ ...
                D(i-1,j-1), ...
                D(i-1,j) + lambda, ...
                D(i,  j-1) + lambda ]);
        end
    end

    % backtrack
    i = N; j = M;
    path = [i, j];
    while ~(i == 1 && j == 1)
        candidates = [inf, inf, inf]; % diag, up, left
        if i > 1 && j > 1
            candidates(1) = D(i-1, j-1);
        end
        if i > 1
            candidates(2) = D(i-1, j) + lambda;
        end
        if j > 1
            candidates(3) = D(i, j-1) + lambda;
        end
        [~, k] = min(candidates);
        if k == 1
            i = i - 1; j = j - 1;
        elseif k == 2
            i = i - 1;
        else
            j = j - 1;
        end
        path = [[i, j]; path]; %#ok<AGROW>
    end
end

function X_aligned = warp_to_ref_axis(X, path, M)
% Warp X to reference axis of length M using DTW path:
%   For each reference index j, average all X(i,:) aligned to j.
    C = size(X,2);
    X_aligned = zeros(M, C);
    for j = 1:M
        iList = path(path(:,2) == j, 1);
        if isempty(iList)
            % fallback: nearest j in path (rare with standard DTW)
            [~, k] = min(abs(path(:,2) - j));
            iList = path(k,1);
        end
        X_aligned(j,:) = mean(X(iList,:), 1);
    end
end
