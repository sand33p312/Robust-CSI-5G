%% =========================================================================
%  Smart MIMO 3x2 Dataset — Main Script
%
%  Key design principles (vs old dataset):
%   1. NO normalisation in MATLAB — raw complex saved, Python normalises
%   2. Model-aware splits: CDL-A/C/D = train/val/test, CDL-B/E = gen-model
%   3. Condition-aware splits: extreme Doppler/DS/SCS = gen-cond track
%   4. SNR range extended to -5 dB (low-SNR stress test)
%   5. Per-sample metadata: every sample carries its full condition label
%
%  Output folders:
%   smart_cdl_mimo3x2/
%     train/          <- CDL-A/C/D, seen conditions, 70% stratified
%     val/            <- CDL-A/C/D, seen conditions, 20% stratified
%     test/           <- CDL-A/C/D, seen conditions, 10% stratified
%     gen_model/      <- CDL-B/E,   same conditions as train (model OOD)
%     gen_cond/       <- CDL-A/C/D, extreme Doppler/DS/SCS (condition OOD)
%
%  Total: ~200,000 samples across all splits
%% =========================================================================

clear all; close all; clc;
rng(42);

%% ── Antenna + grid (fixed) ───────────────────────────────────────────────
config.numTxAntennas    = 3;
config.numRxAntennas    = 2;
config.numSubcarriers   = 12;    % 1 RB = 12 SC
config.numSymbols       = 14;    % full NR slot
config.carrierFrequency = 3.5e9; % FR1 sub-6 GHz standard band

%% ── Training/val/test split ratios ───────────────────────────────────────
config.trainRatio = 0.70;
config.valRatio   = 0.20;
config.testRatio  = 0.10;

%% ── Samples per combination ───────────────────────────────────────────────
config.samplesPerCombo = 500;

%% ── TRAINING CONDITIONS (seen during training) ───────────────────────────
% CDL models: A (NLOS dense urban), C (NLOS suburban), D (LOS)
config.train.channelModels     = {'CDL-A', 'CDL-C', 'CDL-D'};
% Delay spread: low / medium / moderate
config.train.delaySpreadValues = [30e-9, 100e-9, 200e-9];   % ns
% Doppler: pedestrian / slow vehicle / urban vehicle
config.train.dopplerShifts     = [5, 30, 100];               % Hz
% Subcarrier spacing: NR numerologies 0 and 1
config.train.subcarrierSpacings= [15e3, 30e3];               % Hz
% No SNR here — Python adds AWGN at any SNR at training time

%% ── GENERALISATION-MODEL CONDITIONS (unseen CDL type) ────────────────────
config.gen_model.channelModels     = {'CDL-B', 'CDL-E'};
config.gen_model.delaySpreadValues = config.train.delaySpreadValues;
config.gen_model.dopplerShifts     = config.train.dopplerShifts;
config.gen_model.subcarrierSpacings= config.train.subcarrierSpacings;
% No SNR — Python adds noise

%% ── GENERALISATION-CONDITION CONDITIONS (unseen extreme conditions) ───────
config.gen_cond.channelModels      = {'CDL-A', 'CDL-C', 'CDL-D'};
config.gen_cond.delaySpreadValues  = [300e-9, 400e-9];
config.gen_cond.dopplerShifts      = [200, 500];
config.gen_cond.subcarrierSpacings = [60e3];
% No SNR — Python adds noise

%% ── Summary ──────────────────────────────────────────────────────────────
nTrain_combos = numel(config.train.channelModels)      * ...
               numel(config.train.delaySpreadValues)   * ...
               numel(config.train.dopplerShifts)       * ...
               numel(config.train.subcarrierSpacings);
nGenM_combos  = numel(config.gen_model.channelModels)      * ...
               numel(config.gen_model.delaySpreadValues)   * ...
               numel(config.gen_model.dopplerShifts)       * ...
               numel(config.gen_model.subcarrierSpacings);
nGenC_combos  = numel(config.gen_cond.channelModels)      * ...
               numel(config.gen_cond.delaySpreadValues)   * ...
               numel(config.gen_cond.dopplerShifts)       * ...
               numel(config.gen_cond.subcarrierSpacings);

nTotal = (nTrain_combos + nGenM_combos + nGenC_combos) * config.samplesPerCombo;

fprintf('============================================================\n');
fprintf('  Smart MIMO 3x2 Dataset — Clean Physics, No Noise\n');
fprintf('============================================================\n');
fprintf('  NOISE         : NOT added in MATLAB — Python adds AWGN\n');
fprintf('  NORMALISATION : NOT applied — Python normalises\n');
fprintf('  Carrier freq  : %.1f GHz\n', config.carrierFrequency/1e9);
fprintf('\n  [TRAIN/VAL/TEST conditions]\n');
fprintf('    CDL models  : %s\n', strjoin(config.train.channelModels, ', '));
fprintf('    Delay spread: %s ns\n', mat2str(config.train.delaySpreadValues*1e9));
fprintf('    Doppler     : %s Hz\n', mat2str(config.train.dopplerShifts));
fprintf('    SCS         : %s kHz\n', mat2str(config.train.subcarrierSpacings/1e3));
fprintf('    Combos      : %d  x %d = %d samples\n', ...
        nTrain_combos, config.samplesPerCombo, nTrain_combos*config.samplesPerCombo);
fprintf('\n  [GEN-MODEL: unseen CDL type]\n');
fprintf('    CDL models  : %s\n', strjoin(config.gen_model.channelModels, ', '));
fprintf('    Combos      : %d  x %d = %d samples\n', ...
        nGenM_combos, config.samplesPerCombo, nGenM_combos*config.samplesPerCombo);
fprintf('\n  [GEN-COND: unseen extreme conditions]\n');
fprintf('    CDL models  : %s\n', strjoin(config.gen_cond.channelModels, ', '));
fprintf('    Delay spread: %s ns\n', mat2str(config.gen_cond.delaySpreadValues*1e9));
fprintf('    Doppler     : %s Hz\n', mat2str(config.gen_cond.dopplerShifts));
fprintf('    SCS         : %s kHz\n', mat2str(config.gen_cond.subcarrierSpacings/1e3));
fprintf('    Combos      : %d  x %d = %d samples\n', ...
        nGenC_combos, config.samplesPerCombo, nGenC_combos*config.samplesPerCombo);
fprintf('\n  TOTAL SAMPLES : %d\n', nTotal);
fprintf('  (multiply by any number of SNR levels in Python at no extra cost)\n');
fprintf('============================================================\n\n');

%% ── Create output directories ────────────────────────────────────────────
config.outputFolder = 'smart_cdl_mimo3x2';
subdirs = {'train', 'val', 'test', 'gen_model', 'gen_cond'};
if ~exist(config.outputFolder, 'dir'), mkdir(config.outputFolder); end
for k = 1:numel(subdirs)
    d = fullfile(config.outputFolder, subdirs{k});
    if ~exist(d,'dir'), mkdir(d); end
end

%% ── Generate + save each track ───────────────────────────────────────────
ticTotal = tic;

%% Track 1: Training pool (CDL-A/C/D, seen conditions)
fprintf('\n[1/3] Generating TRAIN pool (CDL-A/C/D, seen conditions)...\n');
trainPool = g_smart_cdl_mimo3x2(config, config.train);
[trainData, valData, testData] = splitByRatio(trainPool, config);
saveRawDataset(trainData, fullfile(config.outputFolder,'train'), 'train', config);
saveRawDataset(valData,   fullfile(config.outputFolder,'val'),   'val',   config);
saveRawDataset(testData,  fullfile(config.outputFolder,'test'),  'test',  config);
clear trainPool;  % free memory

%% Track 2: Generalisation-Model (CDL-B/E, same conditions)
fprintf('\n[2/3] Generating GEN-MODEL pool (CDL-B/E, same conditions)...\n');
genModelData = g_smart_cdl_mimo3x2(config, config.gen_model);
saveRawDataset(genModelData, fullfile(config.outputFolder,'gen_model'), 'gen_model', config);
clear genModelData;

%% Track 3: Generalisation-Condition (CDL-A/C/D, extreme conditions)
fprintf('\n[3/3] Generating GEN-COND pool (CDL-A/C/D, extreme conditions)...\n');
genCondData = g_smart_cdl_mimo3x2(config, config.gen_cond);
saveRawDataset(genCondData, fullfile(config.outputFolder,'gen_cond'), 'gen_cond', config);
clear genCondData;

%% ── Save config ──────────────────────────────────────────────────────────
save(fullfile(config.outputFolder, 'config.mat'), 'config', '-v7.3');

fprintf('\n============================================================\n');
fprintf('  ALL DONE in %.1f minutes\n', toc(ticTotal)/60);
fprintf('  Output: %s/\n', config.outputFolder);
fprintf('  Splits: train/ val/ test/ gen_model/ gen_cond/\n');
fprintf('  Clean physics saved — NO noise, NO normalisation\n');
fprintf('  Python recipe:\n');
fprintf('    noise_power = 10^(-SNR_dB/10)\n');
fprintf('    Y_noisy = Y_clean + sqrt(noise_power/2)*(randn+1j*randn)\n');
fprintf('    then z-score X, Y_noisy, H per sample\n');
fprintf('============================================================\n');