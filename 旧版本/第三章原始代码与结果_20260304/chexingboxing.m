% 你需要把 data_mat_path 改成自己的 RealPack 三分类数据文件路径
data_mat_path = 'D:\xidian_Master\研究生论文\毕业论文\实验数据\第三章\数据\processedVehicleData_3class_REAL (2).mat';
A = load(data_mat_path);

ProcessedData = A.ProcessedData;   % 1x3 cell，每类一个 cell，内部是事件序列
targetLength  = A.targetLength;    % 1x3 cell，每类每个事件的长度

labels = ["小型车","中型车","大型车"];

figure; hold on; grid on;
for c = 1:3
    Ls = double(targetLength{c}(:));
    Lmed = median(Ls);

    % 选一个长度最接近中位数的样本作为“典型”
    [~, idx] = min(abs(Ls - Lmed));

    xyz = double(ProcessedData{c}{idx});   % [T x 3]
    b = sqrt(sum(xyz.^2, 2));              % 模值 b[n]

    % 可选：z-score，和你 DTW 对齐示例保持一致
    b = (b - mean(b)) / (std(b) + 1e-12);

    plot(b, 'DisplayName', labels(c));
end
xlabel('采样点 n');
ylabel('b[n] (z-score)');
legend('Location','best');