function saveRawDataset(data, outputFolder, splitName, config)
%% saveRawDataset  — Save clean physics dataset to HDF5 + MAT metadata
%
%  Saves CLEAN data only — no noise, no normalisation.
%  Python adds noise and normalises at training time.
%
%  HDF5 layout:
%    /X_grid   [N, nTx, nSC, nSym, 2]  float32 — clean pilot grid (Re/Im)
%    /Y_clean  [N, nRx, nSC, nSym, 2]  float32 — clean received, power-normed (Re/Im)
%    /H_freq   [N, nCh, nSC, nSym, 2]  float32 — true channel (Re/Im)
%    /delay_spread, /doppler_shift, /scs_kHz, /H_power_dB, /sig_power
%  Attributes: noise_in_matlab=0, normalised=0
%
%  Python noise recipe (using saved sig_power):
%    noise_power = sig_power / 10^(SNR_dB/10)   [if not power-normed]
%    OR simply: noise_power = 10^(-SNR_dB/10)   [Y_clean is already power-normed]
%    noise  = sqrt(noise_power/2) * (randn+1j*randn)
%    Y_noisy = Y_clean + noise

    if ~exist(outputFolder,'dir'), mkdir(outputFolder); end

    h5path  = fullfile(outputFolder, [splitName '.h5']);
    matpath = fullfile(outputFolder, [splitName '_meta.mat']);

    N   = size(data.H_freq, 1);
    nTx = size(data.X_grid,  2);
    nRx = size(data.Y_clean, 2);
    nCh = size(data.H_freq,  2);
    nSC = size(data.H_freq,  3);
    nSy = size(data.H_freq,  4);

    fprintf('  Saving %s -> %s\n', splitName, outputFolder);
    fprintf('    Samples : %d | H_freq: [%d x %d x %d x %d]\n', N, N, nCh, nSC, nSy);
    fprintf('    Noise   : NOT in data — Python adds at runtime\n');
    fprintf('    Norm    : NOT applied — Python normalises at runtime\n');

    %% Convert complex → real+imag last dim: [N, Ch, nSC, nSym, 2]
    X_ri = cat(5, real(data.X_grid),  imag(data.X_grid));   % [N,nTx,nSC,nSy,2]
    Y_ri = cat(5, real(data.Y_clean), imag(data.Y_clean));  % [N,nRx,nSC,nSy,2]
    H_ri = cat(5, real(data.H_freq),  imag(data.H_freq));   % [N,nCh,nSC,nSy,2]

    %% Write HDF5
    if exist(h5path, 'file'), delete(h5path); end

    h5create(h5path, '/X_grid',  size(X_ri), 'Datatype','single', 'ChunkSize',[min(N,512),nTx,nSC,nSy,2]);
    h5create(h5path, '/Y_clean', size(Y_ri), 'Datatype','single', 'ChunkSize',[min(N,512),nRx,nSC,nSy,2]);
    h5create(h5path, '/H_freq',  size(H_ri), 'Datatype','single', 'ChunkSize',[min(N,512),nCh,nSC,nSy,2]);

    h5write(h5path, '/X_grid',  X_ri);
    h5write(h5path, '/Y_clean', Y_ri);
    h5write(h5path, '/H_freq',  H_ri);

    %% HDF5 attributes
    h5writeatt(h5path, '/', 'split',             splitName);
    h5writeatt(h5path, '/', 'num_samples',        N);
    h5writeatt(h5path, '/', 'nTx',               nTx);
    h5writeatt(h5path, '/', 'nRx',               nRx);
    h5writeatt(h5path, '/', 'nCh',               nCh);
    h5writeatt(h5path, '/', 'nSC',               nSC);
    h5writeatt(h5path, '/', 'nSym',              nSy);
    h5writeatt(h5path, '/', 'noise_in_matlab',    0);   % FLAG: 0 = no noise saved
    h5writeatt(h5path, '/', 'normalised',         0);   % FLAG: 0 = raw
    h5writeatt(h5path, '/', 'carrier_freq_GHz',  config.carrierFrequency/1e9);
    h5writeatt(h5path, '/', 'pilot_symbols',     data.pilotSymbols);
    h5writeatt(h5path, '/', 'pilot_SCs_Tx1',     data.pilotSCs_Tx1);
    h5writeatt(h5path, '/', 'pilot_SCs_Tx2',     data.pilotSCs_Tx2);
    h5writeatt(h5path, '/', 'pilot_SCs_Tx3',     data.pilotSCs_Tx3);

    %% Scalar condition arrays in HDF5
    h5create(h5path, '/delay_spread',  [N,1], 'Datatype','single');
    h5create(h5path, '/doppler_shift', [N,1], 'Datatype','single');
    h5create(h5path, '/scs_kHz',       [N,1], 'Datatype','single');
    h5create(h5path, '/H_power_dB',    [N,1], 'Datatype','single');
    h5create(h5path, '/sig_power',     [N,1], 'Datatype','single');

    h5write(h5path, '/delay_spread',  data.delaySpread);
    h5write(h5path, '/doppler_shift', data.dopplerShift);
    h5write(h5path, '/scs_kHz',       data.scs_kHz);
    h5write(h5path, '/H_power_dB',    data.H_power_dB);
    h5write(h5path, '/sig_power',     data.sig_power);

    %% MAT metadata
    metadata = struct();
    metadata.channelModel  = data.channelModel;   % cell array — not HDF5-friendly
    metadata.delaySpread   = data.delaySpread;
    metadata.dopplerShift  = data.dopplerShift;
    metadata.scs_kHz       = data.scs_kHz;
    metadata.H_power_dB    = data.H_power_dB;
    metadata.sig_power     = data.sig_power;
    metadata.pilotSymbols  = data.pilotSymbols;
    metadata.pilotSCs_Tx1  = data.pilotSCs_Tx1;
    metadata.pilotSCs_Tx2  = data.pilotSCs_Tx2;
    metadata.pilotSCs_Tx3  = data.pilotSCs_Tx3;
    metadata.splitName     = splitName;
    metadata.numSamples    = N;
    metadata.noise_in_matlab = false;
    metadata.normalised    = false;
    metadata.config        = config;
    save(matpath, 'metadata', '-v7.3');

    %% Summary
    Hpow_mean = mean(data.H_power_dB);
    Hpow_std  = std(data.H_power_dB);
    uMdl = unique(data.channelModel);
    fprintf('    H_power : mean=%.1f dB  std=%.1f dB\n', Hpow_mean, Hpow_std);
    fprintf('    CDL     : %s\n', strjoin(uMdl', ', '));
    fprintf('    HDF5    : %s  (%.1f MB)\n', h5path, (dir(h5path).bytes)/1e6);
end