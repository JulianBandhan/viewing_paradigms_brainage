% BEvIS Bayes factor connectivity matrix plot
%
% Plots the paired Baby Einstein vs Inscapes Bayes factors as a region x region
% matrix, then lists the edges with BF10 above a chosen threshold.
%
% Input:  BEvIS_BF10.csv and BEvIS_log10BF10.csv, saved by connmat_bayes_analysis.r
%         A tab-delimited label file with one region name per line, in the
%         same order as the matrix
% Output: Matrix figure and a table of edges printed to the command window
%
% Run from the repository root. Update the paths below to match your data.

%% Paths

results_dir = 'results';
label_file  = fullfile('data', 'region_labels.txt');

%% Load data

BF10      = readmatrix(fullfile(results_dir, 'BEvIS_BF10.csv'));
log10BF10 = readmatrix(fullfile(results_dir, 'BEvIS_log10BF10.csv'));

nodelabels4mat = readtable(label_file, ...
    'Delimiter', '\t', 'ReadVariableNames', false);

% Regions to label on the axes. Use a subset if the labels are too crowded
labels2plot = 1:height(nodelabels4mat);

% Largest absolute value, used to centre the colour scale on 0
L = max(abs(log10BF10(:)), [], 'omitnan');

%% Plot matrix

figure();
h = imagesc(log10BF10, [-L L]);
set(h, 'AlphaData', ~isnan(log10BF10))  % show NaN cells as transparent
set(gca, 'Color', [1 1 1])              % white background behind NaN cells
colormap(divergent_cmap(256))
cb = colorbar;

% Label the selected regions
yticks(labels2plot)
yticklabels(nodelabels4mat.Var1(labels2plot))
xticks(labels2plot)
xticklabels(nodelabels4mat.Var1(labels2plot))
axis image;
set(gca, 'FontSize', 20, 'Box', 'off');

% Show colorbar ticks in original BF10 units
bfTicks = [0.1 0.33 1 3 10 30];           % round BF values spanning your range
bfTicks = bfTicks(bfTicks >= 10^(-L) & bfTicks <= 10^L);  % keep ticks inside plotted range
cb.Ticks = log10(bfTicks);                % position on the log10 axis
cb.TickLabels = compose('%.2g', bfTicks); % display as original BF values
cb.Label.String = 'BF_{10}';

%% List edges above threshold

% List all edges with BF10 > threshold
thresh = 1;
n = size(BF10, 1);

symBF = max(BF10, BF10.');   % ignores NaNs, so works if only one triangle is filled
[r, c] = find(symBF > thresh & tril(true(n), -1));   % each edge once, no diagonal
vals = symBF(sub2ind([n n], r, c));

% Build a table of edges, sorted from strongest to weakest evidence
edgeTable = table(nodelabels4mat.Var1(r), nodelabels4mat.Var1(c), r, c, vals, log10(vals), ...
    'VariableNames', {'Region1', 'Region2', 'Index1', 'Index2', 'BF10', 'log10BF10'});
edgeTable = sortrows(edgeTable, 'BF10', 'descend');

fprintf('%d of %d edges have BF10 > %g\n', height(edgeTable), n*(n-1)/2, thresh);
disp(edgeTable)

%% Local functions

function cmap = divergent_cmap(n)
% Blue-grey-red colormap with n colours, matching the R plots
if nargin < 1, n = 256; end
low  = [33 102 172]/255;   % #2166AC
mid  = [242 242 242]/255;  % grey95
high = [178 24 43]/255;    % #B2182B
half = floor(n/2);
cmap = [interp1([0 1], [low; mid], linspace(0,1,half)); ...
    interp1([0 1], [mid; high], linspace(0,1,n-half))];
end