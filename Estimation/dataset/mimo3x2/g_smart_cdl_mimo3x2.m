function dataset = g_smart_cdl_mimo3x2(config, trackConfig)
%% generateCDLDataset  — Clean physics CDL MIMO 3x2 generator
%
%  NO noise added. NO normalisation applied.
%  MATLAB saves the clean physics only.
%  Python adds AWGN, normalises, and does any other preprocessing.
%
%  This means the dataset is generated ONCE and can be used with:
%    - any SNR (add noise in Python)
%    - any noise type (AWGN, phase noise, impulsive — all in Python)
%    - any normalisation scheme (z-score, maxVal, per-antenna — in Python)
%    - online SNR randomisation per batch
%
%  Outputs per sample:
%    X_grid   [nTx x nSC x nSym]  — sparse pilot grid (complex, raw, clean)
%    Y_clean  [nRx x nSC x nSym]  — clean received grid (NO noise, power-normed)
%    H_freq   [nCh x nSC x nSym]  — true channel (complex, raw, clean)
%    channelModel, delaySpread, dopplerShift, scs_kHz — condition labels
%    H_power_dB — per-sample mean channel power in dB
%    sig_power  — received signal power before normalisation (for Python SNR ref)

    models   = trackConfig.channelModels;
    DSvals   = trackConfig.delaySpreadValues;
    Dopvals  = trackConfig.dopplerShifts;
    SCSvals  = trackConfig.subcarrierSpacings;

    nModels  = numel(models);
    nDS      = numel(DSvals);
    nDop     = numel(Dopvals);
    nSCS     = numel(SCSvals);
    sPC      = config.samplesPerCombo;

    nTx      = config.numTxAntennas;   % 3
    nRx      = config.numRxAntennas;   % 2
    nCh      = nTx * nRx;              % 6
    nSC      = config.numSubcarriers;  % 12
    nSym     = config.numSymbols;      % 14

    % No SNR loop — noise added in Python
    totalCombos  = nModels * nDS * nDop * nSCS;
    totalSamples = totalCombos * sPC;

    % Orthogonal pilot layout — DMRS-like
    % Symbol positions 1 and 8 (1-indexed), staggered SCs per Tx
    pilotSymbols  = [1, 8];
    pilotSCs_all  = {[1,4,7,10], [2,5,8,11], [3,6,9,12]};
    % Each Tx occupies 4 out of 12 SCs per pilot symbol → 14.3% density

    %% Allocate (single-precision to save memory)
    dataset.X_grid       = zeros(totalSamples, nTx, nSC, nSym, 'single');
    dataset.Y_clean      = zeros(totalSamples, nRx, nSC, nSym, 'single'); % clean, no noise
    dataset.H_freq       = zeros(totalSamples, nCh, nSC, nSym, 'single');
    dataset.channelModel = cell(totalSamples, 1);
    dataset.delaySpread  = zeros(totalSamples, 1, 'single');
    dataset.dopplerShift = zeros(totalSamples, 1, 'single');
    dataset.scs_kHz      = zeros(totalSamples, 1, 'single');
    dataset.H_power_dB   = zeros(totalSamples, 1, 'single');
    dataset.sig_power    = zeros(totalSamples, 1, 'single'); % received power ref
    % Pilot pattern (same for all samples)
    dataset.pilotSymbols  = pilotSymbols;
    dataset.pilotSCs_Tx1  = pilotSCs_all{1};
    dataset.pilotSCs_Tx2  = pilotSCs_all{2};
    dataset.pilotSCs_Tx3  = pilotSCs_all{3};

    fprintf('  Combinations : %d  (no SNR loop — noise added in Python)\n', totalCombos);
    fprintf('  Samples      : %d\n', totalSamples);
    fprintf('  Noise        : NONE saved — Python adds AWGN at runtime\n');
    fprintf('  Normalisation: NONE — raw complex\n');

    sampleCount = 0;
    ticStart    = tic;
    progressBar = waitbar(0, 'Generating...', 'Name', 'CDL Dataset');

    for mIdx = 1:nModels
      for dsIdx = 1:nDS
        for dopIdx = 1:nDop
          for scsIdx = 1:nSCS

              mdl  = models{mIdx};
              ds   = DSvals(dsIdx);
              dop  = Dopvals(dopIdx);
              scs  = SCSvals(scsIdx);   % Hz

              for smp = 1:sPC
                sampleCount = sampleCount + 1;

                if mod(sampleCount, 500) == 0
                    elapsed = toc(ticStart);
                    rate    = sampleCount / max(elapsed, 1);
                    remain  = (totalSamples - sampleCount) / rate;
                    waitbar(sampleCount/totalSamples, progressBar, ...
                        sprintf('%s | %d/%d | ETA %.0f min', ...
                        mdl, sampleCount, totalSamples, remain/60));
                end

                [Xg, Yc, Hf, hpow_dB, sp] = generateOneSample(...
                    mdl, ds, dop, scs, config, ...
                    pilotSymbols, pilotSCs_all);

                dataset.X_grid(sampleCount,:,:,:)  = single(Xg);
                dataset.Y_clean(sampleCount,:,:,:) = single(Yc);
                dataset.H_freq(sampleCount,:,:,:)  = single(Hf);
                dataset.channelModel{sampleCount}   = mdl;
                dataset.delaySpread(sampleCount)    = single(ds);
                dataset.dopplerShift(sampleCount)   = single(dop);
                dataset.scs_kHz(sampleCount)        = single(scs/1e3);
                dataset.H_power_dB(sampleCount)     = single(hpow_dB);
                dataset.sig_power(sampleCount)      = single(sp);

              end % smp
          end % scs
        end % dop
      end % ds
    end % model

    close(progressBar);
    elapsed = toc(ticStart);
    fprintf('  Done: %d samples in %.1f min\n', totalSamples, elapsed/60);

    %% Sanity check — report raw ranges
    Hmag_max  = max(abs(dataset.H_freq(:)));
    Hmag_mean = mean(abs(dataset.H_freq(:)));
    Yclean_max= max(abs(dataset.Y_clean(:)));
    fprintf('  Raw |H_freq|  max=%.4f  mean=%.4f\n', Hmag_max, Hmag_mean);
    fprintf('  Raw |Y_clean| max=%.4f  (clean, no noise)\n', Yclean_max);
    fprintf('  H_power range : [%.1f, %.1f] dB\n', ...
            min(dataset.H_power_dB), max(dataset.H_power_dB));
    fprintf('  sig_power range: [%.4f, %.4f]\n', ...
            min(dataset.sig_power), max(dataset.sig_power));
    fprintf('  NOTE: No noise, no normalisation — Python handles both\n');
end


%% =========================================================================
function [X_grid, Y_clean, H_freq, H_power_dB, sig_power] = ...
    generateOneSample(channelModel, delaySpread, dopplerShift, ...
                       scs_hz, config, pilotSymbols, pilotSCs_all)
%% generateOneSample — generates ONE clean sample, NO noise
%
%  Returns:
%    X_grid    [nTx x nSC x nSym]  clean pilot grid (raw complex)
%    Y_clean   [nRx x nSC x nSym]  clean received signal (power-normed, NO noise)
%    H_freq    [nCh x nSC x nSym]  true channel (raw complex, clean)
%    H_power_dB  scalar — mean channel power in dB
%    sig_power   scalar — received signal power before normalisation
%
%  Python uses sig_power as SNR reference:
%    noise_power = sig_power / 10^(SNR_dB/10)
%    noise = sqrt(noise_power/2) * (randn + 1j*randn)
%    Y_noisy = Y_clean + noise

    nTx  = config.numTxAntennas;
    nRx  = config.numRxAntennas;
    nCh  = nTx * nRx;
    nSC  = config.numSubcarriers;
    nSym = config.numSymbols;

    %% Carrier config
    carrier = nrCarrierConfig;
    carrier.NSizeGrid         = nSC / 12;
    carrier.SubcarrierSpacing = scs_hz / 1e3;   % kHz

    ofdmInfo = nrOFDMInfo(carrier);

    %% CDL channel
    cdl = nrCDLChannel;
    cdl.DelayProfile        = channelModel;
    cdl.DelaySpread         = delaySpread;
    cdl.MaximumDopplerShift = dopplerShift;
    cdl.SampleRate          = ofdmInfo.SampleRate;
    cdl.CarrierFrequency    = config.carrierFrequency;
    cdl.TransmitAntennaArray.Size = [nTx, 1, 1, 1, 1];
    cdl.ReceiveAntennaArray.Size  = [nRx, 1, 1, 1, 1];
    % CDL-D/E are LOS — nrCDLChannel handles K-factor automatically

    %% TX pilot grid — random QPSK pilots
    txGrid = zeros(nSC, nSym, nTx);
    for txIdx = 1:nTx
        pSCs = pilotSCs_all{txIdx};
        for symIdx = pilotSymbols
            pilots = (sign(randn(numel(pSCs),1)) + ...
                      1i*sign(randn(numel(pSCs),1))) / sqrt(2);  % QPSK
            txGrid(pSCs, symIdx, txIdx) = pilots;
        end
    end

    %% Modulate + propagate through CDL channel
    txWaveform = nrOFDMModulate(carrier, txGrid);
    [rxWaveform, pathGains, sampleTimes] = cdl(txWaveform);
    pathFilters = getPathFilters(cdl);

    %% Perfect time-varying channel — [nSC x nSym x nRx x nTx]
    H_perfect = nrPerfectChannelEstimate(carrier, pathGains, ...
                                          pathFilters, 0, sampleTimes);

    %% Demodulate — clean received grid, NO noise yet
    rxGrid = nrOFDMDemodulate(carrier, rxWaveform);
    % rxGrid: [nSC x nSym x nRx]

    %% Power normalise received signal
    %  This makes sig_power=1 after normalisation.
    %  Python adds noise relative to this normalised power:
    %    noise_power = 10^(-SNR_dB/10)
    sig_power = mean(abs(rxGrid(:)).^2);   % save BEFORE normalising
    if sig_power > 0
        rxGrid = rxGrid / sqrt(sig_power);
    end
    % After this: E[|rxGrid|^2] = 1.0
    % Python: noise_power = 10^(-SNR_dB/10), then add noise

    %% Channel power in dB
    H_power_dB = 10 * log10(mean(abs(H_perfect(:)).^2) + 1e-12);

    %% Pack H_freq [nCh x nSC x nSym] — raw, no normalisation
    H_freq = zeros(nCh, nSC, nSym);
    chIdx  = 0;
    for rxIdx = 1:nRx
        for txIdx = 1:nTx
            chIdx = chIdx + 1;
            H_freq(chIdx,:,:) = squeeze(H_perfect(:,:,rxIdx,txIdx));
        end
    end

    %% Pack X_grid [nTx x nSC x nSym] — clean pilots, raw
    X_grid = zeros(nTx, nSC, nSym);
    for txIdx = 1:nTx
        pSCs = pilotSCs_all{txIdx};
        for symIdx = pilotSymbols
            X_grid(txIdx, pSCs, symIdx) = txGrid(pSCs, symIdx, txIdx);
        end
    end

    %% Pack Y_clean [nRx x nSC x nSym] — clean, power-normed, NO noise
    Y_clean = zeros(nRx, nSC, nSym);
    for rxIdx = 1:nRx
        Y_clean(rxIdx,:,:) = squeeze(rxGrid(:,:,rxIdx));
    end

end