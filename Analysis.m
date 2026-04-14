% Data analysis in Tian & Doiron (2026)
close all; clear; clc

set(groot, 'defaultLineLineWidth', 2)
set(groot, 'defaultAxesLineWidth', 2)
set(groot, 'defaultAxesFontSize', 20)
set(groot, 'defaultAxesTickDir', 'out')
set(groot,  'defaultAxesTickDirMode', 'manual')

% The data for this analysis comes from Duszkiewicz et al. (2024) and can 
% be downloaded from https://doi.org/10.6084/m9.figshare.24921252
% The link also includes the MATLAB analysis code used in the original 
% paper. Our analysis uses some of the dependencies in the code, in 
% particular the TSToolbox (https://github.com/PeyracheLab/TStoolbox).

% Make_TSToolbox
% addpath .../Code/Dependencies
% dataPath = '.../Dataset_1/'

%% Preprocess data
nBin = 36; % Number of bins (between -pi and pi) for tuning curve
winSize = 0.1; % Time bin size to calculate spike counts

sessions = dir([dataPath, '*-*']); nSession = length(sessions);
allData = struct(); 

h = waitbar(0);
for iSession = 1 : nSession
    fileName = sessions(iSession).name;
    [R0, Xall, R, V, theta0] = preprocsession(dataPath, fileName, nBin, winSize);
    allData(iSession).R0 = R0; % Tuning curves with finer bins for shape analysis
    allData(iSession).Xall = Xall; % Spike counts
    allData(iSession).R = R; % Tuning curves with coarser bins for variability analysis
    allData(iSession).V = V; % Variances calculated using the same bins as R
    allData(iSession).theta0 = theta0; % Angles corresponding to R0's binning
    waitbar(iSession/nSession, h)
end
delete(h)

%% Tuning curve shape (Fig. 4d)
ratioThresh = 0.2; % Threshold of bimodality
rsqThresh = 0.9; % Only look at good fits to rule out tuning curves with 3+ modes

peakRatioAll = []; % Ratio between the smaller and larger peak to measure bimodality
dPhiAll = []; % Phase difference between the two peaks

h = waitbar(0);
for iSession = 1 : nSession
    R0 = allData(iSession).R0;
    theta0 = allData(iSession).theta0;
    [dPhi, peakRatio, rsq] = fitbivmpop(R0, theta0);
    dPhiAll = [dPhiAll; dPhi(rsq > rsqThresh)];
    peakRatioAll = [peakRatioAll; peakRatio(rsq > rsqThresh)];
    waitbar(iSession/nSession, h)
end
delete(h)

% Plot
x = 0 : 0.1 : 1;
figure; histogram(peakRatioAll, x, 'Normalization', 'pdf')
xlabel('Peak magnitude ratio'); ylabel('Density')

x = linspace(0, pi, 9);
figure; histogram(dPhiAll(peakRatioAll > ratioThresh), x, 'Normalization', 'pdf')
xlabel('Peak phase difference'); ylabel('Density')

%% PCA (Fig. 4f)
iSession = 1; 
R0 = allData(iSession).R0; 
theta0 = allData(iSession).theta0;

[y0, u, v] = fitpop(R0, theta0);
[Rp, Yp] = socpop(y0, u, v, theta0);
Yn = sqrt(R0) / 2;
Y = sign(Yp) .* Yn; % Only use the sign of the fitted model

[coeffR, scoreR, latentR] = pca(R0'); 
[coeffY, scoreY, latentY] = pca(Y');
latentR = cumsum(latentR / sum(latentR));
latentY = cumsum(latentY / sum(latentY));

figure; plot(latentR(1:20)); hold on; plot(latentY(1:20))
xlabel('Eigenmode'); ylabel('Cumulative explained variance')
yline(0.8, '--', 'LineWidth', 2); legend('r coordinates', 'y coordinates')

figure; plot(scoreR(:, 1), scoreR(:, 2)); axis equal; xlabel('PC 1'); ylabel('PC 2')
figure; plot(scoreY(:, 1), scoreY(:, 2)); axis equal; xlabel('PC 1'); ylabel('PC 2')

%% Correlation between two sigmas (Fig. 5c-d)
ratioThresh = 0.2; % Threshold of bimodality
rsqThresh = 0.9; % Only look at good fits
nShuffle = 10; % Number of shuffle data per neuron
cAll = zeros(1, nSession); % Correlation values between the sigmas
cShuffle = zeros(nShuffle, nSession); % Shuffled correlations

for iSession = 1 : nSession
    disp(['Processing session ', num2str(iSession)])

    R0 = allData(iSession).R0;
    theta0 = allData(iSession).theta0;
    
    nNeu = size(R0, 1); goodfit = false(nNeu, 1);
    sigma = zeros(nNeu, 1); sigma0 = zeros(nNeu, 1);
    h = waitbar(0);
    for iNeu = 1 : nNeu
        % Only use unimodal tuning curves
        r = allData(iSession).R0(iNeu, :);
        [ratio, dPhi, rsq] = fitbivmneu(r, theta);
        if ratio > ratioThresh || rsq < rsqThresh 
            continue
        end
            
        % sigma_tun
        fitted = gffit(r, theta0);
        sigma(iNeu) = fitted(4);

        % sigma_var
        X = cell(1, nBin);
        for iBin = 1 : nBin
            X{iBin} = allData(iSession).Xall{iBin}(:, iNeu);
        end
        sigma0(iNeu) = fitnbin(X); 
        if sigma0(iNeu) > 0 % sigma0 = 0 if NB fitting failed
            goodfit(iNeu) = true;
        end
    
        waitbar(iNeu/nNeu, h)
    end
    delete(h)

    sigma0 = sigma0(goodfit); sigma = sigma(goodfit);
    allData(iSession).sigma0 = sigma0; allData(iSession).sigma = sigma;
    
    c = corrcoef(sigma0, sigma); cAll(iSession) = c(1, 2);
    nGF = length(sigma0);
    for iShuffle = 1 : nShuffle
        neuShuffle = randperm(nGF);
        c = corrcoef(sigma0, sigma(neuShuffle));
        cShuffle(iShuffle, iSession) = c(1, 2);
    end
end

iSession = 4;
sigma0 = allData(iSession).sigma0;
sigma0 = (sigma0-min(sigma0))/(max(sigma0)-min(sigma0));
sigma = allData(iSession).sigma;
sigma = (sigma-min(sigma))/(max(sigma)-min(sigma));
figure('position', [0 0 560 420]); scatter(sigma0, sigma, 80, 'filled'); 
xlabel('Normalized super-Poissonness \sigma_{var}')
ylabel('Normalized tuning sharpness \sigma_{tun}')

x = -1 : 0.1 : 1; 
figure; histogram(cAll, x, 'Normalization', 'pdf')
hold on; histogram(cShuffle, x, 'Normalization', 'pdf')
xlabel('Corr. between \sigma_{var} and \sigma_{tun}')
legend('Data', 'Shuffle'); ylabel('Density')

%% Functions
function [R0, Xall, R, V, theta0] = preprocsession(dataPath, fileName, nBin, winSize)
% Preprocess data from one session
load(fullfile(dataPath, fileName, 'Data', 'BehavEpochs'));
load(fullfile(dataPath, fileName, 'Data', 'SpikeData'), 'S');
load(fullfile(dataPath, fileName, 'Data', 'Angle'));
load(fullfile(dataPath, fileName, 'Data', 'Velocity'));
load(fullfile(dataPath, fileName, 'Data', 'CellTypes'), 'hd');
load(fullfile(dataPath, fileName, 'Analysis', 'HdTuning_moveEp'), 'hAll');

% Fine-grained tuning curve using the method in Duszkiewicz et al. (2024)
R0 = squeeze(hAll(:, hd, 4))';
nBin0 = size(R0, 2);
binEdges0 = linspace(-pi, pi, nBin0 + 1);
theta0 = (binEdges0(1:end-1) + binEdges0(2:end))/2;

% Parameters and initialization
velTh = 2; % Velocity threshold
S = S(hd); % Focus on head direction neurons
nNeu = length(S); % Number of neurons
Xall = cell(1, nBin); % Spike count matrices
R = zeros(nNeu, nBin); % Firing rate vectors (tuning curves)
Call = zeros(nNeu, nNeu, nBin); % Covariance matrices
V = zeros(nNeu, nBin); % Variances

% Restrict to epoch
if ~exist('wake1Ep', 'var')   
    wake1Ep = wakeEp;
end
ep = wake1Ep;
angR = Restrict(ang, ep);
angrange = Range(angR);  % Make sure only tracked times are used
ep = intervalSet(angrange(1), angrange(end));
    
% Restrict ep to times with movement (as in the original paper)
epVel = thresholdIntervals(vel, velTh, 'Direction', 'Above');
ep = intersect(ep, epVel);
angR = Restrict(angR, ep); % Restrict angle to movement epoch
binEdges = linspace(-pi, pi, nBin + 1);

for iBin = 1 : nBin
    epBin = boundaryIntervals(angR, binEdges(iBin), binEdges(iBin+1));
    maxWin = Data(max(length(epBin)));
    nWin = floor(maxWin/winSize);
    tStart = [];
    tEnd = [];
    for iWin = 1 : nWin
        epBin = dropShortIntervals(epBin, winSize*iWin);
        temp = Start(epBin) + (iWin-1)*winSize;
        tStart = [tStart; temp];
        tEnd = [tEnd; temp + winSize];
    end
    epBin = intervalSet(sort(tStart), sort(tEnd));
    nWin = length(tStart);
    X = zeros(nWin, nNeu);
    for iNeu = 1 : nNeu
        X(:, iNeu) = Data(intervalCount(S{iNeu}, epBin));
    end
    Xall{iBin} = X;
    R(:, iBin) = mean(X);
    C = cov(X);
    Call(:, :, iBin) = C;
    V(:, iBin) = diag(C);
end

% Eliminate components of the variance tangent to the tuning curve
for iBin = 1 : nBin
    C = squeeze(Call(:, :, iBin));
    if iBin == 1
        i1 = nBin; i2 = 2;
    elseif iBin == nBin
        i1 = nBin - 1; i2 = 1;
    else
        i1 = iBin - 1; i2 = iBin + 1;
    end
    r1 = R(:, i1);
    r2 = R(:, i2);
    dr = r2 - r1;
    a = sum(triu_ele(C.*(dr*dr')))/sum(triu_ele((dr*dr').^2));
    D = diag(C);
    D0 = D - a * (dr.^2);
    D0(D0 < 0) = 0;
    V(:, iBin) = D0;
end

end


