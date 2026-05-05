function make_confusion_chart(matFile, outPng, titleStr, fontCN)
% 目标：生成与 MATLAB confusionchart 类似的版式（主矩阵+行/列汇总）
%      并在主矩阵中把 0 显式写出来，避免"空白像出图错误"。

    assert(exist(matFile,'file')==2, '缺少混淆矩阵数据：%s', matFile);
    D = load(matFile);

    cm = double(D.cm);
    labelsRaw = D.labels;

    % labels 统一为 cellstr（1xN）
    if iscell(labelsRaw)
        labels = labelsRaw(:).';
    elseif isstring(labelsRaw)
        labels = cellstr(labelsRaw(:)).';
    else
        labels = cellstr(labelsRaw(:)).';
    end

    nClass = size(cm,1);

    fig = figure('Color','w','Position',[100 100 950 720]);

    % --- 用 confusionchart 生成你图里那种版式 ---
    cc = confusionchart(cm, labels);
    cc.Normalization = 'absolute';           % 主矩阵显示计数
    cc.RowSummary    = 'row-normalized';     % 右侧：每行正确/错误百分比（召回视角）
    cc.ColumnSummary = 'column-normalized';  % 底部：每列正确/错误百分比（精确视角）

    cc.Title  = titleStr;
    cc.XLabel = '预测类';
    cc.YLabel = '真实类';

    % 字体（避免中文变方块）
    try
        cc.FontName = fontCN;
        cc.FontSize = 12;
    catch
    end

    % 让数字以整数显示（部分版本仍会对 0 隐藏，所以后面再兜底补 0）
    try
        cc.CellLabelFormat = '%d';
    catch
    end

    % --- 兜底：主矩阵中把 0 补出来 ---
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
% confusionchart 会产生多个 Axes（主矩阵、行汇总、列汇总）
% 用"XTick 和 YTick 都是 1:nClass"的规则找主矩阵轴
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
