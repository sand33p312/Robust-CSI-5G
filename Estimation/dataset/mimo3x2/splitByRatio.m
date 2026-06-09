function [trainData, valData, testData] = splitByRatio(dataset, config)
%% splitByRatio  — Stratified split across all condition dimensions
%
%  Splits the training pool (CDL-A/C/D, seen conditions) into
%  train / val / test by stratifying across every combination cell.
%  Each combo block gets the same ratio split so every condition appears
%  in all three splits.
%
%  This is called ONLY on the training pool.
%  gen_model and gen_cond are already separate — do NOT split them.

    numSamples  = size(dataset.H_freq, 1);
    sPC         = config.samplesPerCombo;
    nCombos     = numSamples / sPC;

    rng(42);  % reproducible

    trainIdx = [];
    valIdx   = [];
    testIdx  = [];

    for c = 1:nCombos
        blockIdx = ((c-1)*sPC + 1):(c*sPC);
        shuffled = blockIdx(randperm(numel(blockIdx)));

        nTr = round(sPC * config.trainRatio);
        nVl = round(sPC * config.valRatio);
        % nTe = remainder

        trainIdx = [trainIdx, shuffled(1:nTr)];
        valIdx   = [valIdx,   shuffled(nTr+1 : nTr+nVl)];
        testIdx  = [testIdx,  shuffled(nTr+nVl+1 : end)];
    end

    trainData = extractSubset(dataset, trainIdx);
    valData   = extractSubset(dataset, valIdx);
    testData  = extractSubset(dataset, testIdx);

    fprintf('  Stratified split:\n');
    fprintf('    train : %d samples\n', numel(trainIdx));
    fprintf('    val   : %d samples\n', numel(valIdx));
    fprintf('    test  : %d samples\n', numel(testIdx));

    %% Verify all conditions present in each split
    fprintf('  DS  in train  : %s ns\n',  mat2str(unique(trainData.delaySpread)'*1e9));
    fprintf('  Dop in train  : %s Hz\n',  mat2str(unique(trainData.dopplerShift)'));
    fprintf('  SCS in train  : %s kHz\n', mat2str(unique(trainData.scs_kHz)'));
    uModels = unique(trainData.channelModel);
    fprintf('  CDL in train  : %s\n',     strjoin(uModels', ', '));
    fprintf('  NOTE: No SNR field — noise added in Python at runtime\n');
end


%% =========================================================================
function sub = extractSubset(ds, idx)
    sub = struct();
    sub.X_grid        = ds.X_grid(idx,:,:,:);
    sub.Y_clean       = ds.Y_clean(idx,:,:,:);
    sub.H_freq        = ds.H_freq(idx,:,:,:);
    sub.channelModel  = ds.channelModel(idx);
    sub.delaySpread   = ds.delaySpread(idx);
    sub.dopplerShift  = ds.dopplerShift(idx);
    sub.scs_kHz       = ds.scs_kHz(idx);
    sub.H_power_dB    = ds.H_power_dB(idx);
    sub.sig_power     = ds.sig_power(idx);
    sub.pilotSymbols  = ds.pilotSymbols;
    sub.pilotSCs_Tx1  = ds.pilotSCs_Tx1;
    sub.pilotSCs_Tx2  = ds.pilotSCs_Tx2;
    sub.pilotSCs_Tx3  = ds.pilotSCs_Tx3;
end