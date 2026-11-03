%% BEvIS connectivity
%
% Compares connectivity between the Baby Einstein (BE) and Inscapes (IS) tasks:
%   - Plots the group-average connectivity matrix for each task
%   - Compares average connectivity across tasks at the edge level
%   - Compares global connectivity across tasks for matched subjects
%   - Finds the strongest connections in each task and plots them in
%     BrainNet Viewer
%
% Input:  Subject information from opm_study_config() and a connectivity
%         matrix for each scan. p.subject_data.task_io is 0 for BE and 1 for IS.
%         Region coordinates, labels, and categories.
% Output: Figures, statistics printed to the command window, and BrainNet Viewer
%         node and edge files.
%
% Update the paths and settings below to match your data.

close all
clear all
clc

%% Paths and settings

% Toolboxes
addpath('/path/to/toolbox/for/brainnetviewer/ver/');
addpath('/path/to/toolbox/for/brain-connectivity-toolbox/ver/');

% Region information
node_coords_path = '/path/to/coord.txt';
node_labels_path = '/path/to/reg_labels.txt';
plot_labels_path = '/path/to/reg_plot_labels.txt';
node_cat_path    = '/path/to/node/categories.txt';

% BrainNet Viewer files
outputpath = '/path/to/output';
options    = '/path/to/brainnetviewer/options.mat';

band = '1';           % Frequency band to analyze
top_percent = 0.05;   % Proportion of strongest connections to keep

%% Load connectivity matrices

p = opm_study_config();

subjectID_categorical = categorical(p.subject_data.subject);
subjectID = double(subjectID_categorical);

% Iterate over the subjects
for ss = 1:size(p.subject_data, 1)

    disp(num2str(ss))
    temp = load(p.conenctivity.aec_mat_ve_coordinates(p.subject(ss), p.session(ss), p.run(ss), p.task(ss), ...
        'all', band));

    % Take the absolute values and remove the diagonal
    n_regions = size(temp.adjmat, 1);
    temp.adjmat = abs(temp.adjmat) .* (1 - eye(n_regions));

    % Normalize by the RMS of the upper triangle
    connMatrices(:,:,ss) = abs(temp.adjmat)./rms(temp.adjmat(find(triu(temp.adjmat,1))),'all');
end

idx_roi = 1:n_regions;   % Regions included when finding the top connections

%% Plot average connectivity matrices

nodelabels4mat = readtable(plot_labels_path, 'Delimiter', '\t', 'ReadVariableNames', false);

% Regions to label on the axes. Use a subset if the labels are too crowded
labels2plot = 1:height(nodelabels4mat);

% Scans for each task
BE_idx = find(p.subject_data.task_io == 0);
IS_idx = find(p.subject_data.task_io == 1);

% Group-average connectivity for each task
BE_avg = mean(connMatrices(:, :, BE_idx), 3);
IS_avg = mean(connMatrices(:, :, IS_idx), 3);

% BE average, using a colour scale shared across tasks
figure();
colorlim = [min([BE_avg(:); IS_avg(:)]), max([BE_avg(:); IS_avg(:)])];

imagesc(BE_avg)
colorbar
yticks(labels2plot)
yticklabels(nodelabels4mat.Var1(labels2plot))
xticks(labels2plot)
xticklabels(nodelabels4mat.Var1(labels2plot))
caxis(colorlim)
axis image;
set(gca, 'FontSize', 20, 'Box', 'off');

% IS average, using a colour scale shared across tasks
figure();
colorlim = [min([BE_avg(:); IS_avg(:)]), max([BE_avg(:); IS_avg(:)])];

imagesc(IS_avg)
colorbar
yticks(labels2plot)
yticklabels(nodelabels4mat.Var1(labels2plot))
xticks(labels2plot)
xticklabels(nodelabels4mat.Var1(labels2plot))
caxis(colorlim)
axis image;
set(gca, 'FontSize', 20, 'Box', 'off');

%% Edge-level BE vs IS connectivity plot

% Use each unique edge once
ut = triu(true(n_regions), 1);
x  = IS_avg(ut);
y  = BE_avg(ut);

% Remove edges with zero connectivity
mask = x ~= 0 & y ~= 0;
x = x(mask);
y = y(mask);

% --- correlation stats -------------------------------------------
[r_edge, p_edge] = corr(x, y);
n_edge = numel(x);
[log10p_edge, p_str] = corr_log10p(r_edge, n_edge);

fprintf('\nEdge-level network agreement (BE_avg vs IS_avg):\n');
fprintf('  n = %d edges, r = %.3f, p = %s  (log10 p = %.2f)\n', ...
    n_edge, r_edge, strrep(p_str, '^{', 'e'), log10p_edge);

% Regression
p_fit = polyfit(x, y, 1);
x_line = linspace(min(x), max(x), 100);

figure;
set(gcf, 'Color', 'w', 'Units', 'inches', 'Position', [1 1 5.5 5.5]);

plot(x, y, '.', 'Color', [0.6 0.6 0.6], 'MarkerSize', 12); hold on;
plot(x_line, polyval(p_fit, x_line), '-', 'LineWidth', 3, 'Color', [0.2 0.2 0.2]);
if p_edge < 0.001
    stat_str = sprintf('r = %.2f, p < .001', r_edge);
else
    stat_str = sprintf('r = %.2f, p = %.3f', r_edge, p_edge);
end
text(0.05, 0.95, stat_str, 'Units','normalized', ...
    'FontSize', 18, 'VerticalAlignment','top', 'Color', [0.15 0.15 0.15]);

% Axis limits
pad = 0.05 * (max(x) - min(x));
xlim([min(x)-pad, max(x)+pad]);
ylim([min(y)-pad, max(y)+pad]);

xlabel('IS Connectivity', 'FontSize', 20);
ylabel('BE Connectivity', 'FontSize', 20);

ax = gca;
ax.FontSize        = 12;
ax.LineWidth       = 1.2;
ax.TickDir         = 'out';
ax.Box             = 'off';
ax.XColor          = [0.15 0.15 0.15];
ax.YColor          = [0.15 0.15 0.15];
set(gca, 'FontSize', 20, 'Box', 'off');

%% Global BE vs IS connectivity plot

% Global connectivity: sum of the upper triangle, not RMS normalized
global_conn = zeros(size(p.subject_data,1), 1);

for ss = 1:size(p.subject_data, 1)
    disp(num2str(ss))
    temp = load(p.conenctivity.aec_mat_ve_coordinates(p.subject(ss), p.session(ss), ...
        p.run(ss), p.task(ss), 'all', band));
    global_conn(ss) = sum(abs(temp.adjmat(ut)));
end

% Match subjects across tasks
be_subj = subjectID(BE_idx);
is_subj = subjectID(IS_idx);
[~, ia, ib] = intersect(be_subj, is_subj, 'stable');

y = global_conn(BE_idx(ia));   % BabyEinstein, matched
x = global_conn(IS_idx(ib));   % Inscapes,     matched

% --- stats -------------------------------------------------------
[r_glob, p_glob] = corr(x, y);
n_glob = numel(x);
p_fit  = polyfit(x, y, 1);

d = y - x;                                   % BE - IS, for agreement
fprintf('\nGlobal connectivity, BE vs IS across participants:\n');
fprintf('  n = %d, r(%d) = %.3f, p = %.3g\n', n_glob, n_glob-2, r_glob, p_glob);
fprintf('  fit: BE = %.3f*IS %+.3f\n', p_fit(1), p_fit(2));
fprintf('  mean BE-IS diff = %.3g (SD %.3g), 95%% LoA [%.3g, %.3g]\n', ...
    mean(d), std(d), mean(d)-1.96*std(d), mean(d)+1.96*std(d));

% --- plot --------------------------------------------------------
ax_lim = [0, max([x; y])];
pad    = 0.05 * diff(ax_lim);
ax_lim = ax_lim + [-pad pad];
x_line = linspace(ax_lim(1), ax_lim(2), 100);

figure('Color','w','Units','inches','Position',[1 1 5.5 5.5]);
hold on;

plot(x_line, polyval(p_fit, x_line), '-', 'LineWidth', 2, 'Color', [0.2 0.2 0.2]);
plot(x, y, 'o', 'MarkerSize', 10, ...
    'MarkerFaceColor', [0.6 0.6 0.6], ...
    'MarkerEdgeColor', [0.35 0.35 0.35], ...
    'LineWidth', 0.5, 'HandleVisibility','off');

xlim(ax_lim); ylim(ax_lim);
axis square;
xlabel('IS Global Connectivity');
ylabel('BE Global Connectivity');

ax = gca;
ax.FontSize  = 20;
ax.LineWidth = 1.2;
ax.TickDir   = 'out';
ax.Box       = 'off';
ax.XColor    = [0.15 0.15 0.15];
ax.YColor    = [0.15 0.15 0.15];
hold off;

%% Find and plot the top connections

mask_BE = get_top_connections(BE_avg, idx_roi, top_percent);
mask_IS = get_top_connections(IS_avg, idx_roi, top_percent);

figure();
colorlim = [min(mask_BE(:)), max(mask_BE(:))];

imagesc(mask_BE)
colorbar
caxis(colorlim)
yticks(labels2plot)
yticklabels(nodelabels4mat.Var1(labels2plot))
xticks(labels2plot)
xticklabels(nodelabels4mat.Var1(labels2plot))
axis image;
title('top BE')

figure();
colorlim = [min(mask_IS(:)), max(mask_IS(:))];

imagesc(mask_IS)
colorbar
yticks(labels2plot)
yticklabels(nodelabels4mat.Var1(labels2plot))
xticks(labels2plot)
xticklabels(nodelabels4mat.Var1(labels2plot))
caxis(colorlim)
axis image;
title('top IS')

%% Save node and edge files

% Define input files for node and edge file creation
node_coords = load(node_coords_path);
node_labels = readtable(node_labels_path, 'Delimiter', '\t', 'ReadVariableNames', false);
node_labels = node_labels.Var1;
node_cat = load(node_cat_path);

% Save edge files
save([outputpath '/top_BE_edges.edge'], 'mask_BE','-ascii');
save([outputpath '/top_IS_edges.edge'], 'mask_IS', '-ascii');

% Save node files
create_node_file([outputpath '/top_BE_nodes.node'], mask_BE, node_coords, node_cat, node_labels);
create_node_file([outputpath '/top_IS_nodes.node'], mask_IS, node_coords, node_cat, node_labels);

%% Plot in BrainNet Viewer

% Plots the IS network. Change to the top_BE files to plot the BE network
edgepath = [outputpath '/top_IS_edges.edge'];
nodepath = [outputpath '/top_IS_nodes.node'];
plotInBrainNet(edgepath, nodepath, 'options', options);


%% Local functions

function mask = get_top_connections(data, idx_roi, top_ratio)
% Returns a symmetric binary mask of the strongest connections in data

    % Get the submatrix
    conn_filtered = data(idx_roi, :);

    % Find duplicate connections and set to zero
    for i = 1:length(idx_roi)
        for j = 1:i  % includes diagonal
            conn_filtered(i, idx_roi(j)) = 0;
        end
    end

    % Extract the nonzero values
    [row_idx, col_idx, val] = find(conn_filtered);

    % Sort and find top values
    n_top = ceil(top_ratio * numel(val));
    [~, sort_idx] = sort(abs(val), 'descend');
    top_idx = sort_idx(1:n_top);

    % Resulting top coordinates
    top_rows = row_idx(top_idx);
    top_cols = col_idx(top_idx);

    submask = zeros(size(conn_filtered));
    for i = 1:length(top_rows)
        submask(top_rows(i), top_cols(i)) = 1;
    end

    % Reset to original size
    mask = zeros(size(data));
    mask(idx_roi, :) = submask;

    % Make sure it's symmetric
    mask = mask | mask';
    mask = double(mask);

end

function create_node_file(nodepath, mask, node_coords, node_cat, node_labels)
% Writes a BrainNet Viewer node file, with node size set by degree in mask

    node_size = degrees_und(mask)';

    % Construct node file
    t = table();
    t.Var1 = node_coords(:,1);
    t.Var2 = node_coords(:,2);
    t.Var3 = node_coords(:,3);
    t.Var4 = reshape(node_cat,[],1);
    t.Var5 = node_size;
    t.Var6 = reshape(node_labels,[],1);

    writetable(t, nodepath, 'FileType', 'text', 'Delimiter', '\t', 'WriteVariableNames', false);

end

function [] = plotInBrainNet(edgepath, nodepath, varargin)
% Plots node and edge files on a surface using BrainNet Viewer

iP = inputParser;
addParameter(iP, 'options', []);
addParameter(iP, 'surface', '/path/to/brainnetviewer/Data/SurfTemplate/BrainMesh_ICBM152_smoothed.nv');
parse(iP, varargin{:});

% Set up the parameters that will go into BrainNet Viewer
brainnet_params = {iP.Results.surface, edgepath, nodepath};
if ~isempty(iP.Results.options)
  brainnet_params{end+1} = iP.Results.options;
end

% Call BrainNet Viewer
BrainNet_MapCfg(brainnet_params{:});
end

function [log10p, str] = corr_log10p(r, n)
% Exact two-sided p-value for Pearson r, computed in log space so it
% doesn't underflow. p = I_x(df/2, 1/2) with x = 1 - r^2.
df = n - 2;
a  = df/2;  b = 0.5;
x  = 1 - r^2;

% Hypergeometric series correction term (converges fast for small x)
S = 1; k = 0; term = 1;
while term > 1e-16*S && k < 1e4
    term = exp(betaln(a+1, k+1) - betaln(a+b, k+1) + (k+1)*log(x));
    S = S + term;  k = k + 1;
end

lnp    = a*log(x) + b*log(r^2) - log(a) - betaln(a, b) + log(S);
log10p = lnp / log(10);

e   = floor(log10p);
m   = 10^(log10p - e);
str = sprintf('%.1f x 10^{%d}', m, e);
end