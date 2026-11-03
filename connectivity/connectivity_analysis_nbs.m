%% BEvIS connectivity and cross-task NBS comparison
%
% Part 1 builds a normalized connectivity matrix for each scan and saves the
% matrices and NBS design matrix for each task and frequency band. Run it once
% per task.
%
% Part 2 uses those matrices and the saved NBS outputs to compare the Baby
% Einstein (BE) and Inscapes (IS) tasks. NBS itself is run separately. For each
% task it produces:
%   - Node and edge files and a BrainNet Viewer plot of the NBS network
% Across tasks it produces:
%   - A matched connectivity file for BEvIS_paired_BF.R
%   - Network strength vs age plots for the BE-derived and IS-derived networks,
%     with connectivity over each network's edges computed from both tasks
%   - A linear model of connectivity on age for each task, and a linear mixed
%     effects model testing the task by age interaction
%
% Author: Julian Bandhan
% Updated: 2026-11-03
%
% Update the paths and settings in each part to match your data.

%% PART 1: BUILD CONNECTIVITY MATRICES

close all
clear
clc

num_bands = 1;         % Number of frequency bands
task_f = 'inscapes';   % Task to process. Also change the task in the config file

for jj = 1:num_bands
    % Load the subject information and keep the scans for this task
    p = opm_study_config();
    band = num2str(jj);
    p.subject_data = p.subject_data(strcmp(p.subject_data.task_label, task_f), :);

    n = height(p.subject_data);

    % NBS design matrix: intercept and age
    design = [ones(n,1), p.subject_data.age];

    tic

    % Iterate over the subjects
    for ss = 1:size(p.subject_data, 1)

        disp(num2str(ss))
        fpath = p.conenctivity.aec_mat_ve_coordinates(p.subject(ss), p.session(ss), p.run(ss), p.task(ss), 'all', band);
        temp = load(fpath);
        fprintf('ss=%d  task_f=%s  p.task(ss)=%s  file=%s\n', ss, task_f, string(p.task(ss)), fpath);

        % Take the absolute values and normalize by the RMS of the upper triangle
        connMatrices(:,:,ss) = abs(temp.adjmat)./rms(temp.adjmat(find(triu(temp.adjmat,1))),'all');
    end

    % Save the connectivity matrices and design matrix
    filename = sprintf('%s_%s_BEvIS_ConnMat.mat', task_f, band);
    save(filename, 'connMatrices');
    writematrix(design, 'design_mat.txt', 'Delimiter', 'tab');
    toc
end


%% PART 2: CROSS-TASK NBS COMPARISON
% Requires the connectivity matrices from Part 1 and the NBS outputs for both tasks

close all; clear; clc;
addpath(genpath('/path/to/toolbox/NBS1.2/'));

%% -------------------- EDIT THESE PATHS --------------------
band = '1';   % Frequency band to compare

% Per-task file locations
tasks(1).name          = 'babyeinstein';
tasks(1).conn_mat_path = sprintf('%s_%s_BEvIS_ConnMat.mat', tasks(1).name, band);
tasks(1).nbs_out_path  = sprintf('BE_%s_int_NBSoutput.mat', band);
tasks(1).color         = [0.20 0.40 0.75];   % blue

tasks(2).name          = 'inscapes';
tasks(2).conn_mat_path = sprintf('%s_%s_BEvIS_ConnMat.mat', tasks(2).name, band);
tasks(2).nbs_out_path  = sprintf('IS_%s_int_NBSoutput.mat', band);
tasks(2).color         = [0.85 0.40 0.20];   % orange

% Node metadata, shared across tasks
node_coords_path = '/path/to/coord.txt';
node_labels_path = '/path/to/reg_labels.txt';

node_cat_path    = '/path/to/node/categories.txt';   % Region category vector
brainnet_options = '/path/to/brainnetviewer/options.mat';


%% SECTION 1: Reload the subject information and matrices for each task
%  Uses the same filtering and order as when connMatrices was built

for tt = 1:numel(tasks)
    p = opm_study_config();
    p.subject_data = p.subject_data(strcmp(p.subject_data.task_label, tasks(tt).name), :);

    tasks(tt).subjectID = string(p.subject_data.subject);   % string ID for matching
    tasks(tt).age       = p.subject_data.age;

    % Subjects are matched by ID, so each subject must appear once per task
    assert(numel(unique(tasks(tt).subjectID)) == numel(tasks(tt).subjectID), ...
        '%s: subject IDs are repeated, so subjects cannot be matched across tasks.', tasks(tt).name);

    ld = load(tasks(tt).conn_mat_path); fn = fieldnames(ld);
    tasks(tt).connMatrices = ld.(fn{1});

    assert(size(tasks(tt).connMatrices,3) == numel(tasks(tt).subjectID), ...
        '%s: connMatrices depth does not match the number of subjects. Check that the filtering matches what built the .mat file.', tasks(tt).name);

    tasks(tt).nbsdata = load(tasks(tt).nbs_out_path);
end

n_regions = size(tasks(1).connMatrices, 1);

%% SECTION 2: Extract the NBS network edge mask for each task
%  Combines all significant networks into one symmetric mask

for tt = 1:numel(tasks)
    nbsdata = tasks(tt).nbsdata;
    num_networks = numel(nbsdata.nbs.NBS.con_mat);
    comb = false(size(full(nbsdata.nbs.NBS.con_mat{1})));
    for kk = 1:num_networks
        comb = comb | logical(full(nbsdata.nbs.NBS.con_mat{kk}));
    end
    comb = comb | comb';
    if nnz(comb) == 0
        warning('%s: NBS network is empty, no significant edges.', tasks(tt).name);
    end
    tasks(tt).net_mask = comb;
end

%% SECTION 3: Brain plots for each task (node and edge files, BrainNet Viewer)

addpath('/path/to/toolbox/for/brainnetviewer/ver/');
addpath('/path/to/toolbox/for/brain-connectivity-toolbox/ver/');
node_coords = load(node_coords_path);
node_labels = readtable(node_labels_path, 'Delimiter','\t', 'ReadVariableNames', false).Var1;
node_cat    = load(node_cat_path);

for tt = 1:numel(tasks)
    net_mask = tasks(tt).net_mask;
    if nnz(net_mask) == 0, continue; end

    connMatrices = tasks(tt).connMatrices;
    age_v        = tasks(tt).age;

    % Correlate each edge with age to use as edge weights
    rMat = zeros(n_regions);
    [r, c] = find(triu(ones(n_regions),1));
    for e = 1:numel(r)
        rMat(r(e),c(e)) = corr(squeeze(connMatrices(r(e),c(e),:)), age_v, 'rows','complete');
    end
    rMat = rMat + rMat';

    % Keep the edges in the NBS network
    edge = rMat .* net_mask;
    edge(isnan(edge)) = 0;

    edge_out_path = sprintf('%s_BEvIS_edge.edge', tasks(tt).name);
    node_out_path = sprintf('%s_BEvIS_node.node', tasks(tt).name);
    save(edge_out_path, '-ascii', 'edge');

    % Node file: coordinates, category, size (degree), and label
    node_size = degrees_und(double(net_mask))';
    t = table();
    t.Var1 = node_coords(:,1); t.Var2 = node_coords(:,2); t.Var3 = node_coords(:,3);
    t.Var4 = reshape(node_cat, [], 1);
    t.Var5 = node_size;
    t.Var6 = node_labels;
    writetable(t, node_out_path, 'FileType','text', 'Delimiter','\t', 'WriteVariableNames', false);

    fprintf('%s: edge/node files saved (%d edges in network)\n', tasks(tt).name, nnz(triu(net_mask,1)));

    if ~isempty(brainnet_options)
        plotInBrainNet(edge_out_path, node_out_path, 'options', brainnet_options);
    end
end

%% SECTION 4: Match subjects common to both tasks

common_id = intersect(tasks(1).subjectID, tasks(2).subjectID);
fprintf('Common subjects across both tasks: %d\n', numel(common_id));

for tt = 1:numel(tasks)
    [~, idx] = ismember(common_id, tasks(tt).subjectID);
    assert(all(idx > 0), 'Subject matching failed for %s.', tasks(tt).name);
    tasks(tt).common_idx = idx;
end

% Age should match within subject across tasks. Use the age from task 1 and
% check it against task 2
age_common = tasks(1).age(tasks(1).common_idx);
age_check  = tasks(2).age(tasks(2).common_idx);
if any(abs(age_common - age_check) > 0.5)
    warning('Age mismatch >0.5 between tasks for some common subjects. Check alignment and session dates.');
end

% Save the matched matrices for BEvIS_paired_BF.R, with BE first and IS second
connMatrices = cat(3, tasks(1).connMatrices(:,:,tasks(1).common_idx), ...
    tasks(2).connMatrices(:,:,tasks(2).common_idx));
if ~exist('data', 'dir'), mkdir('data'); end
save(fullfile('data', 'BEvIS_ConnMat_file.mat'), 'connMatrices', '-v7.3');

%% SECTION 5: Network strength vs age
%  For each NBS-derived network (BE-derived, IS-derived), extract mean
%  connectivity over that network's edges from both tasks' matrices, then fit
%  conn ~ age separately per task and conn ~ task*age + (1|subject) jointly.

plot_titles = {'BE network', 'IS network'};
results     = struct();

nSub = numel(common_id);
age  = age_common(:);

for pp = 1:2
    edge_mask = tasks(pp).net_mask;
    if nnz(edge_mask) == 0
        warning('Skipping %s: network is empty.', plot_titles{pp});
        continue;
    end

    % net_mask is symmetric, so use the upper triangle to count each edge once
    edge_lin = find(triu(edge_mask, 1));

    strength = nan(nSub, 2);          % col 1 = BE conn, col 2 = IS conn
    for tt = 1:2
        cm = tasks(tt).connMatrices(:,:,tasks(tt).common_idx);
        for ss = 1:nSub
            m = cm(:,:,ss);
            strength(ss,tt) = mean(m(edge_lin));
        end
    end

    conn_BE = strength(:,1);
    conn_IS = strength(:,2);

    % --- per-task linear models -------------------------------------
    mdl_BE = fitlm(table(age, conn_BE, 'VariableNames',{'age','conn'}), 'conn ~ age');
    mdl_IS = fitlm(table(age, conn_IS, 'VariableNames',{'age','conn'}), 'conn ~ age');

    % --- combined LME: conn ~ task + age + task:age + (1|subject) ----
    tbl_all = table( ...
        [conn_BE; conn_IS], ...
        categorical([repmat("BE",nSub,1); repmat("IS",nSub,1)]), ...
        [age; age], ...
        categorical([common_id(:); common_id(:)]), ...
        'VariableNames', {'conn','task','age','subject'});
    tbl_all.task = reordercats(tbl_all.task, {'BE','IS'});   % BE = reference

    mdl_all = fitlme(tbl_all, 'conn ~ task*age + (1|subject)');

    % Pull out the task by age interaction
    ix_int = find(contains(mdl_all.CoefficientNames, ':'), 1);
    p_int  = mdl_all.Coefficients.pValue(ix_int);
    b_int  = mdl_all.Coefficients.Estimate(ix_int);

    % --- plot -------------------------------------------------------
    plot_task_regressions(age, conn_BE, conn_IS, mdl_BE, mdl_IS, p_int, ...
        tasks(1).color, tasks(2).color, ...
        sprintf('%s', plot_titles{pp}));

    % --- report -----------------------------------------------------
    fprintf('\n=== %s (%d edges, n=%d subjects) ===\n', ...
        plot_titles{pp}, numel(edge_lin), nSub);
    fprintf('BE  conn~age : slope=%+.4g, t=%.2f, p=%.4g, R2=%.3f\n', ...
        mdl_BE.Coefficients.Estimate(2), mdl_BE.Coefficients.tStat(2), ...
        mdl_BE.Coefficients.pValue(2), mdl_BE.Rsquared.Ordinary);
    fprintf('IS  conn~age : slope=%+.4g, t=%.2f, p=%.4g, R2=%.3f\n', ...
        mdl_IS.Coefficients.Estimate(2), mdl_IS.Coefficients.tStat(2), ...
        mdl_IS.Coefficients.pValue(2), mdl_IS.Rsquared.Ordinary);
    fprintf('LME task:age : delta-slope=%+.4g, p=%.4g\n', b_int, p_int);
    disp(mdl_all);

    results(pp).name    = plot_titles{pp};
    results(pp).mdl_BE  = mdl_BE;
    results(pp).mdl_IS  = mdl_IS;
    results(pp).mdl_all = mdl_all;
    results(pp).conn    = strength;
end


%% =========================================================================
%  Local functions
% ==========================================================================
function plot_task_regressions(age, yBEdat, yISdat, mdl_BE, mdl_IS, p_int, cBE, cIS, ylab)
% Plots connectivity vs age for both tasks with each task's regression line

% Signed Pearson r: for a single-predictor model this equals corr(age, y)
r_BE = sign(mdl_BE.Coefficients.Estimate(2)) * sqrt(mdl_BE.Rsquared.Ordinary);
r_IS = sign(mdl_IS.Coefficients.Estimate(2)) * sqrt(mdl_IS.Rsquared.Ordinary);

figure('Color','w','Units','inches','Position',[1 1 5.5 5.5]); hold on;

scatter(age, yBEdat, 100, cBE, 'filled', 'MarkerFaceAlpha', 0.6, ...
    'HandleVisibility','off');
scatter(age, yISdat, 100, cIS, 'filled', 'MarkerFaceAlpha', 0.6, ...
    'HandleVisibility','off');

xq  = linspace(min(age), max(age), 100)';
yBE = predict(mdl_BE, table(xq, 'VariableNames', {'age'}));
yIS = predict(mdl_IS, table(xq, 'VariableNames', {'age'}));

hBE  = plot(xq, yBE, '-', 'Color', cBE, 'LineWidth', 3);
hIS  = plot(xq, yIS, '-', 'Color', cIS, 'LineWidth', 3);
hInt = plot(nan, nan, 'LineStyle','none');   % legend-only entry

xlabel('Age (years)');
ylabel(ylab);

legend([hBE hIS hInt], { ...
    sprintf('BE (r = %.2f, p = %.3f)', r_BE, mdl_BE.Coefficients.pValue(2)), ...
    sprintf('IS (r = %.2f, p = %.3f)', r_IS, mdl_IS.Coefficients.pValue(2)), ...
    sprintf('Interaction (p = %.3f)', p_int)}, ...
    'Location','best');

set(gca, 'FontSize', 20, 'Box', 'off');
pbaspect([1 1 1]);
hold off;
end