%% plot_two_confmats_thesis.m
% 版式：左=混淆矩阵(计数) 右=召回率  下=精确率（参考你第3张图）
clc; clear; close all;

%% ========== 0) 类别与顺序（匹配参考图：中型、大型、小型） ==========
labels_show = {'中型车','大型车','小型车'};   % 参考图的显示顺序
order = [1 2 3];  % 你的原始矩阵按{小型,中型,大型}给出时，重排到{中型,大型,小型}

%% ========== 1) CNN Baseline（原始顺序：小/中/大） ==========
C_cnn_raw = [ 94  9  0;   % 真小 -> 预测(小/中/大)
             11 45  5;   % 真中
              0  3 38];  % 真大
C_cnn = C_cnn_raw(order, order);  % 重排到参考图顺序（中/大/小）

fig1 = plot_confmat_thesis(C_cnn, labels_show, '离线混淆矩阵（CNN Baseline）');

%% ========== 2) DTW-MultiTemplate-CNN（原始顺序：小/中/大） ==========
C_dtw_raw = [ 99  4  0;   % 真小
              4 57  0;   % 真中
              0  2 39];  % 真大
C_dtw = C_dtw_raw(order, order);

fig2 = plot_confmat_thesis(C_dtw, labels_show, '离线混淆矩阵（DTW-MultiTemplate-CNN）');

%% ========== 3) 导出（可选：论文建议 600 dpi） ==========
exportgraphics(fig1, 'cm_cnn_baseline.png', 'Resolution', 600);
exportgraphics(fig1, 'cm_cnn_baseline.pdf', 'ContentType','image', 'Resolution', 600);

exportgraphics(fig2, 'cm_dtw_multitemplate.png', 'Resolution', 600);
exportgraphics(fig2, 'cm_dtw_multitemplate.pdf', 'ContentType','image', 'Resolution', 600);


%% ========================================================================
function fig = plot_confmat_thesis(C, classNames, mainTitle)
% C: n×n，行=真实类，列=预测类（已按展示顺序排好）
    if isstring(classNames), classNames = cellstr(classNames); end
    n = size(C,1);

    % —— 字体（Windows 下推荐微软雅黑；若无该字体可改为 SimHei）——
    fontCN = 'Microsoft YaHei';
    FS_TIT = 18;
    FS_AX  = 14;
    FS_TXT = 16;

    % —— 颜色：0=白(0值) 1=蓝(对角正确) 2=浅橙(非对角错误)——
    cmap = [1 1 1;
            0.16 0.52 0.78;
            0.95 0.80 0.72];

    % —— 指标（召回/精确率）——
    rowSum = sum(C,2);
    colSum = sum(C,1)';
    recall = diag(C) ./ max(rowSum, 1);
    prec   = diag(C) ./ max(colSum, 1);

    r_pct = round(100*recall);
    p_pct = round(100*prec);

    % —— 画布与布局：2×4，左侧主图占3列，右侧占1列——
    fig = figure('Color','w','Units','centimeters','Position',[2 2 24 16]);
    t = tiledlayout(fig, 2, 4, 'TileSpacing','compact', 'Padding','compact');

    ax_cm = nexttile(t, 1);
    ax_cm.Layout.TileSpan = [1 3];

    ax_rec = nexttile(t, 4);

    ax_pr = nexttile(t, 5);
    ax_pr.Layout.TileSpan = [1 3];

    ax_blank = nexttile(t, 8);
    axis(ax_blank,'off');

    %% --------- (A) 混淆矩阵（计数）---------
    code = zeros(n,n);
    for i = 1:n
        for j = 1:n
            if C(i,j) == 0
                code(i,j) = 0;
            elseif i == j
                code(i,j) = 1;
            else
                code(i,j) = 2;
            end
        end
    end

    imagesc(ax_cm, code);
    colormap(ax_cm, cmap);
    caxis(ax_cm, [-0.5 2.5]);
    axis(ax_cm, 'image');
    set(ax_cm, 'YDir','reverse', ...
        'XTick',1:n, 'YTick',1:n, ...
        'XTickLabel',[], ...
        'YTickLabel',classNames, ...
        'TickLength',[0 0], ...
        'FontName',fontCN, 'FontSize',FS_AX, ...
        'Box','on', 'LineWidth',1.0);
    ylabel(ax_cm, '真实类', 'FontName',fontCN, 'FontSize',FS_AX+1);
    title(ax_cm, mainTitle, 'FontName',fontCN, 'FontSize',FS_TIT, 'Interpreter','none');

    hold(ax_cm,'on');
    for k = 0.5:1:(n+0.5)
        xline(ax_cm, k, 'k', 'LineWidth', 0.8);
        yline(ax_cm, k, 'k', 'LineWidth', 0.8);
    end
    for i = 1:n
        for j = 1:n
            if C(i,j) > 0
                if i == j
                    txtColor = [1 1 1];
                else
                    txtColor = [0 0 0];
                end
                text(ax_cm, j, i, sprintf('%d', C(i,j)), ...
                    'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
                    'FontName',fontCN, 'FontSize',FS_TXT, 'FontWeight','bold', ...
                    'Color',txtColor);
            end
        end
    end
    hold(ax_cm,'off');

    %% --------- (B) 右侧：召回率（按真实类）---------
    recCode = [ones(n,1), 2*ones(n,1)];  % 左蓝=正确，右浅橙=错误
    imagesc(ax_rec, recCode);
    colormap(ax_rec, cmap);
    caxis(ax_rec, [-0.5 2.5]);
    axis(ax_rec, 'image');
    set(ax_rec, 'YDir','reverse', ...
        'XTick',[], 'YTick',[], ...
        'TickLength',[0 0], ...
        'Box','on', 'LineWidth',1.0);

    hold(ax_rec,'on');
    for k = 0.5:1:(n+0.5)
        yline(ax_rec, k, 'k', 'LineWidth', 0.8);
    end
    xline(ax_rec, 0.5, 'k', 'LineWidth', 0.8);
    xline(ax_rec, 1.5, 'k', 'LineWidth', 0.8);
    xline(ax_rec, 2.5, 'k', 'LineWidth', 0.8);

    for i = 1:n
        text(ax_rec, 1, i, sprintf('%d%%', r_pct(i)), ...
            'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
            'FontName',fontCN, 'FontSize',FS_TXT, 'FontWeight','bold', ...
            'Color',[1 1 1]);
        text(ax_rec, 2, i, sprintf('%d%%', 100-r_pct(i)), ...
            'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
            'FontName',fontCN, 'FontSize',FS_TXT, 'FontWeight','bold', ...
            'Color',[0 0 0]);
    end
    hold(ax_rec,'off');

    %% --------- (C) 下方：精确率（按预测类）---------
    prCode = [ones(1,n); 2*ones(1,n)];  % 上蓝=正确，下浅橙=错误
    imagesc(ax_pr, prCode);
    colormap(ax_pr, cmap);
    caxis(ax_pr, [-0.5 2.5]);
    axis(ax_pr, 'image');
    set(ax_pr, 'YDir','reverse', ...
        'XTick',1:n, 'XTickLabel',classNames, ...
        'YTick',[], ...
        'TickLength',[0 0], ...
        'FontName',fontCN, 'FontSize',FS_AX, ...
        'Box','on', 'LineWidth',1.0);
    xlabel(ax_pr, '预测类', 'FontName',fontCN, 'FontSize',FS_AX+1);

    hold(ax_pr,'on');
    for k = 0.5:1:(n+0.5)
        xline(ax_pr, k, 'k', 'LineWidth', 0.8);
    end
    yline(ax_pr, 0.5, 'k', 'LineWidth', 0.8);
    yline(ax_pr, 1.5, 'k', 'LineWidth', 0.8);
    yline(ax_pr, 2.5, 'k', 'LineWidth', 0.8);

    for j = 1:n
        text(ax_pr, j, 1, sprintf('%d%%', p_pct(j)), ...
            'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
            'FontName',fontCN, 'FontSize',FS_TXT, 'FontWeight','bold', ...
            'Color',[1 1 1]);
        text(ax_pr, j, 2, sprintf('%d%%', 100-p_pct(j)), ...
            'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
            'FontName',fontCN, 'FontSize',FS_TXT, 'FontWeight','bold', ...
            'Color',[0 0 0]);
    end
    hold(ax_pr,'off');
end