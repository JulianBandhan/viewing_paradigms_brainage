% BEvIS power spectral density computation
%
% Computes the PSD of each source-modelled region for every scan using Welch's
% method, averages across trials, and saves the result for analysis in Python.
%
% Input:  Subject information from opm_study_config() and a virtual electrode
%         .mat file for each scan containing:
%             ts    : timeseries, region x time x trial
%             srate : sampling rate in Hz
% Output: BEvIS_power.mat containing the structure data, with fields:
%             subject, session, run, age, task
%             psd  : subject x frequency x region
%             freq : frequency values for the PSD bins
%
% Author: Julian Bandhan
% Update: 2026-11-03
%


close all
clear all
clc

%% Load subject information

% Load in config of subject data
p = opm_study_config();
sub_len = length(p.subject_data.subject);

%% Create data structure

data = struct(); % Holds subject information and power
data.subject = cellstr(p.subject_data.subject);
data.session = p.subject_data.session;
data.run = p.subject_data.run;
data.age = p.subject_data.age;
data.task = cellstr(p.subject_data.task);

%% Main loop

% Loop over participants
for ss = 1:sub_len
    fprintf('Loading VE: %s %s %s\n', p.subject(ss), p.session(ss), p.run(ss));

    % Get the virtual electrode file for this scan
    VE_filename = p.sourcemodel.coordinates.ve(p.subject(ss), p.session(ss), p.run(ss), p.task(ss));

    fprintf('Processing VE: %s %s %s\n%s\n', p.subject(ss), p.session(ss), p.run(ss), VE_filename);
    ve = load(VE_filename);
    timeseries = ve.ts; % Region x time x trial

    % Get the PSD frequency values
    [~, psd_freq] = pwelch(zscore(timeseries(1,:,1)), ve.srate, [], [], ve.srate, 'psd');

    % Preallocate using the dimensions of the first scan
    if ss == 1
        data.psd = zeros(sub_len, length(psd_freq), size(timeseries,1));
    end

    % Holds the trial-averaged power for each region
    pxx = zeros(length(psd_freq), size(timeseries,1));

    % Loop over each region
    for ii = 1:size(timeseries,1)

        % Holds the power for each trial
        pxx_t = zeros(size(pxx,1), size(timeseries,3));

        % Loop over each trial
        for tt = 1:size(timeseries,3)
            % Compute the PSD using a one-second window
            [pxx_t(:,tt), ~] = pwelch(zscore(timeseries(ii,:,tt)), ve.srate, [], [], ve.srate, 'psd');
        end

        % Average the power across trials
        pxx(:,ii) = mean(pxx_t, 2);
        %pxx(:,ii) = pxx(:,ii)/sum(pxx(:,ii)); % Normalizing (if zscore isn't used)
    end

    data.psd(ss,:,:) = pxx(:,:);
end

%% Save the data for analysis in Python

data.freq = psd_freq;
psdmat_filename = 'BEvIS_power.mat';
save(psdmat_filename, 'data');
fprintf('Saved: %s\n', psdmat_filename)