%% BEvIS global connectivity vs age
% Compares the relationship between global connectivity and age across the Baby
% Einstein (BE) and Inscapes (IS) tasks. Global connectivity is the sum of the
% upper triangle of each scan's connectivity matrix.
%
% Fits conn ~ age separately for each task, then tests the task by age
% interaction using conn ~ task*age + (1|subject).
%
% Input:  Subject information from opm_study_config() and a connectivity matrix
%         for each scan. The first half of the rows in p.subject_data must be
%         BE and the second half IS, with the same subjects in the same order.
% Output: Model results printed to the command window and a figure of global
%         connectivity vs age for both tasks.

close all; clear; clc;

band = '1';   % Frequency band to analyze

p = opm_study_config();

% Number of subjects per task
n_scans = height(p.subject_data);
assert(mod(n_scans, 2) == 0, 'Expected the same number of BE and IS scans.');
nSub = n_scans / 2;

%% Load BE connectivity matrices

for ss = 1:nSub
    fpath = p.conenctivity.aec_mat_ve_coordinates( ...
        p.subject(ss), p.session(ss), p.run(ss), p.task(ss), 'all', band);
    temp  = load(fpath);

    % Preallocate using the dimensions of the first scan
    if ss == 1
        n_regions = size(temp.adjmat, 1);
        BE_connMatrices = zeros(n_regions, n_regions, nSub);
        IS_connMatrices = zeros(n_regions, n_regions, nSub);
    end

    BE_connMatrices(:,:,ss) = abs(temp.adjmat);   % not RMS normalized
end

%% Load IS connectivity matrices

for ss = 1:nSub
    gi    = ss + nSub;   % row of this subject's IS scan in p.subject_data
    fpath = p.conenctivity.aec_mat_ve_coordinates( ...
        p.subject(gi), p.session(gi), p.run(gi), p.task(gi), 'all', band);
    temp  = load(fpath);
    IS_connMatrices(:,:,ss) = abs(temp.adjmat); 
end

%% Global connectivity: sum of the upper triangle

ut = triu(true(n_regions), 1);   % upper triangle mask, excluding the diagonal

BE_flat = reshape(BE_connMatrices, n_regions^2, nSub);
IS_flat = reshape(IS_connMatrices, n_regions^2, nSub);
BE_global = sum(BE_flat(ut(:), :), 1)';   % nSub x 1
IS_global = sum(IS_flat(ut(:), :), 1)';

%% Subject information

age_BE  = p.subject_data.age(1:nSub);
age_IS  = p.subject_data.age(nSub+1:2*nSub);
sub_BE  = p.subject_data.subject(1:nSub);
sub_IS  = p.subject_data.subject(nSub+1:2*nSub);

% Check that the two blocks have the same subjects in the same order
assert(isequal(string(sub_BE), string(sub_IS)), ...
    'Subject order differs between the BE and IS blocks.');
assert(max(abs(age_BE - age_IS)) < 1e-6, ...
    'Ages differ between the BE and IS blocks for the same subjects.');

age      = age_BE(:);
subjects = string(sub_BE(:));

%% Model 1: BE conn ~ age

tbl_BE = table(BE_global, age, 'VariableNames', {'conn','age'});
mdl_BE = fitlm(tbl_BE, 'conn ~ age');
disp('=== BabyEinstein: conn ~ age ===');  disp(mdl_BE);

%% Model 2: IS conn ~ age

tbl_IS = table(IS_global, age, 'VariableNames', {'conn','age'});
mdl_IS = fitlm(tbl_IS, 'conn ~ age');
disp('=== Inscapes: conn ~ age ===');      disp(mdl_IS);

%% Model 3: combined conn ~ task*age + (1|subject)

conn_all = [BE_global; IS_global];
age_all  = [age; age];
task_all = categorical([repmat("BE", nSub, 1); repmat("IS", nSub, 1)]);
task_all = reordercats(task_all, {'BE','IS'});          % BE = reference
sub_all  = categorical([subjects; subjects]);

tbl_all = table(conn_all, task_all, age_all, sub_all, ...
    'VariableNames', {'conn','task','age','subject'});

mdl_all = fitlme(tbl_all, 'conn ~ task*age + (1|subject)');
disp('=== Combined: conn ~ task + age + task:age + (1|subject) ===');
disp(mdl_all);

%% Plot

% Signed Pearson r: for a single-predictor model this equals corr(age, y)
r_BE = sign(mdl_BE.Coefficients.Estimate(2)) * sqrt(mdl_BE.Rsquared.Ordinary);
r_IS = sign(mdl_IS.Coefficients.Estimate(2)) * sqrt(mdl_IS.Rsquared.Ordinary);

% Pull out the p-value for the task by age interaction
ix_int = find(contains(mdl_all.CoefficientNames, ':'), 1);
p_int  = mdl_all.Coefficients.pValue(ix_int);

figure('Color','w','Units','inches','Position',[1 1 5.5 5.5]); hold on;
cBE = [0.20 0.40 0.75];
cIS = [0.85 0.40 0.20];

% Plot each subject's global connectivity for both tasks
scatter(age, BE_global, 100, cBE, 'filled', 'MarkerFaceAlpha', 0.6, ...
    'HandleVisibility','off');
scatter(age, IS_global, 100, cIS, 'filled', 'MarkerFaceAlpha', 0.6, ...
    'HandleVisibility','off');

% Regression lines from the per-task models
xq  = linspace(min(age), max(age), 100)';
yBE = predict(mdl_BE, table(xq, 'VariableNames', {'age'}));
yIS = predict(mdl_IS, table(xq, 'VariableNames', {'age'}));

hBE  = plot(xq, yBE, '-', 'Color', cBE, 'LineWidth', 3);
hIS  = plot(xq, yIS, '-', 'Color', cIS, 'LineWidth', 3);
hInt = plot(nan, nan, 'LineStyle','none');   % legend-only entry

xlabel('Age (years)');
ylabel('Global connectivity');

legend([hBE hIS hInt], { ...
    sprintf('BE (r = %.2f, p = %.3f)', r_BE, mdl_BE.Coefficients.pValue(2)), ...
    sprintf('IS (r = %.2f, p = %.3f)', r_IS, mdl_IS.Coefficients.pValue(2)), ...
    sprintf('Interaction (p = %.3f)', p_int)}, ...
    'Location','best');

set(gca, 'FontSize', 20, 'Box', 'off');
pbaspect([1 1 1]);
hold off;