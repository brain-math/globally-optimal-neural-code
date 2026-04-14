% Simulations in Tian & Doiron (2026)
close all; clear; clc

set(groot, 'defaultLineLineWidth', 2)
set(groot, 'defaultAxesLineWidth', 2)
set(groot, 'defaultAxesFontSize', 20)
set(groot, 'defaultAxesTickDir', 'out')
set(groot,  'defaultAxesTickDirMode', 'manual')

cmap = colororder;

%% Neurometric curve of (2D) ellipses with isometric Gaussian noise (Fig. 1b)
kAll = [1, 3, 5]; nk = length(kAll); % Aspect ratio b/a
a0 = 5; % Radius of circle
S = pi * a0 / 2; % Quarter of length
nt = 1000; nx = 10; % Parameters for Monte Carlo MI calculation
dThetaAllT = 0:0.05:(pi/2); ndThetaT = length(dThetaAllT); % Delta theta's for theory
dThetaAllN = 0.1:0.2:(pi/2); ndThetaN = length(dThetaAllN); % Delta theta's for numerics
nPair = 100; % Pairs of theta's for each delta theta in numerics
nTrial = 1e4; % Number of classification trials

MI = zeros(1, nk);
lTheory = zeros(nk, ndThetaT); 
lNum = zeros(nk, ndThetaN, nPair);
h = waitbar(0);
for ik = 1 : nk
    k = kAll(ik); m = 1 - k^(-2); % Square of eccentricity
    b = S / ellipticE(m); % Calculate width of ellipse with length 4S
    a = b / k; % Height
    
    % % Monte Carlo for mutual information (takes some time to run)
    % Theta = rand(nt, 1) * 2 * pi;
    % T = arrayfun(@(x) altot(x, m), Theta);
    % Rx = a*cos(T); Ry = b*sin(T);
    % X = randn(nt, nx) + Rx;
    % Y = randn(nt, nx) + Ry;
    % x = X(:); y = Y(:); 
    % MI(ik) = -mean(log(mean(exp(-((x-Rx').^2 + (y-Ry').^2)/2), 2))) - 1;

    % Neurometric curve (theoretical)
    for idTheta = 1 : ndThetaT
        dTheta = dThetaAllT(idTheta);
        lTheory(ik, idTheta) = integral(@(x) hfun(elleucd(altot(x, m), altot(x+dTheta, m), a, b)), -dTheta/2, pi/2-dTheta/2)*2/pi;
    end

    % Neurometric curve (numerical)
    for idTheta = 1 : ndThetaN
        dTheta = dThetaAllN(idTheta);
        for iPair = 1 : nPair
            theta = rand * 2 * pi - dTheta;
            % Calculate centers
            t1 = altot(theta, m); rx1 = a*cos(t1); ry1 = b*sin(t1);
            t2 = altot(theta+dTheta, m); rx2 = a*cos(t2); ry2 = b*sin(t2);

            % Generate response data
            C = rand(nTrial, 1) > 0.5; % Correct category (1 for r1, 0 for r2)
            R = zeros(nTrial, 2); R(C, 1) = rx1; R(C, 2) = ry1;
            R(~C, 1) = rx2; R(~C, 2) = ry2;
            X = R + randn(nTrial, 2); % Response data

            % Optimal decoder under isotropic Gaussian noise is nearest neighbor
            Cpred = pdist2(X, [rx1, ry1]) < pdist2(X, [rx2, ry2]);
            lNum(ik, idTheta, iPair) = mean(C ~= Cpred); % Classification error
        end

        waitbar(((ik-1)*ndThetaN+idTheta) / (nk*ndThetaN), h)
    end
end
delete(h)

% Plot
figure
for ik = 1 : nk
    plot(dThetaAllT, lTheory(ik, :), 'Color', (ik-1)*[0.4,0.4,0.4]); hold on
    lNumi = squeeze(lNum(ik, :, :));
    errorbar(dThetaAllN, mean(lNumi, 2), std(lNumi, [], 2)/sqrt(nPair), 'o', 'Color', (ik-1)*[0.4,0.4,0.4], 'LineWidth', 2, 'MarkerSize', 8);
end
yscale log; xlabel('\Delta\theta'); ylabel('Classification error')
legend('\rho_1/\rho_2=1', '', '\rho_1/\rho_2=3', '', '\rho_1/\rho_2=5')

%% Visualizing tuning curves and calculate selectivity (Fig. 3a)
beta = 4; T = 500; kappa = 2.4; g0 = 3; rho = 10; edges = 0 : 0.05 : 1;

% Von Mises
N = 20; [rVM, t] = genvmr(N, T, kappa, g0); zVM = neusel(t, rVM);

% Square-of-circle
N = 100; [y0, a, b] = gencirc(N, beta, rho);
[y, t] = circy(T, y0, a, b); rC = y.^2; zC = neusel(t, rC); 

figure; plot(t, rVM', 'color', [0, 0, 0, 0.2])
xlabel('\theta'); ylabel('Firing rate')
xticks([0, pi, 2*pi]); xticklabels({'0', '\pi', '2\pi'})

figure('Position', [1 1 560 210])
histogram(zVM, edges, 'Normalization', 'percentage')
xlabel('Selectivity'); ylabel('Percentage')

figure; plot(t, rC', 'color', [0, 0, 0, 0.2]); hold on
xlabel('\theta'); xticks([0, pi, 2*pi]); xticklabels({'0', '\pi', '2\pi'})

figure('Position', [1 1 560 210])
histogram(zC, edges, 'Normalization', 'percentage'); xlabel('Selectivity')

% PCA in y coordinates
yVM = sqrt(rVM); 
[coeffYVM, scoreYVM, latentYVM] = pca(yVM');
[coeffY, scoreY, latentY] = pca(y'); 

figure; plot3([scoreYVM(:, 1); scoreYVM(1, 1)], ...
    [scoreYVM(:, 2); scoreYVM(1, 2)], ...
    [scoreYVM(:, 3); scoreYVM(1, 3)], 'color', cmap(2, :))
axis equal; xlim([-15, 15]); ylim([-15, 15])
xticks([]); yticks([]); zticks([]); view(-24, 29)
xlabel('PC 1'); ylabel('PC 2'); zlabel('PC 3')

figure; plot3([scoreY(:, 1); scoreY(1, 1)], ...
    [scoreY(:, 2); scoreY(1, 2)], ...
    [scoreY(:, 3); scoreY(1, 3)], 'color', cmap(2, :))
axis equal; zlim([-2.8, 2.8]); xticks([]); yticks([]); zticks([]); view(-24, 29)
xlabel('PC 1'); ylabel('PC 2'); zlabel('PC 3'); 

%% Binary classification performance (Fig. 3b)
N = 100; % Number of neurons
dThetaAllT = 0:0.01:pi; ndThetaT = length(dThetaAllT); % Delta theta's for theory
dThetaAllN = 0.1:0.2:pi; ndThetaN = length(dThetaAllN); % Delta theta's for numerics
nPair = 100; % Pairs of theta's for each delta theta in numerics
nTrial = 2e4; % Dataset size for each pair

% Nonnegative circle
rho = 3.5;
[y0, a, b] = gencircnneg(N, rho);
cErrorC = getnmc(@(x) sqc(x, y0, a, b), @poissrnd, nPair, dThetaAllN, nTrial);
ceTheoryC = hfun(2*rho*sin(dThetaAllT/2)); % Theoretical prediction

% Von Mises
kappa = 2.4;
theta = linspace(0, 2*pi, N+1)'; theta = theta(2:end); % Preferred theta's
g0 = rho / sqrt(N*kappa*besseli(1, kappa)); % Match FI
cErrorV = getnmc(@(x) g0^2*exp(kappa*cos(theta - x)), @poissrnd, nPair, dThetaAllN, nTrial);
ceTheoryV = erfc(rho*sqrt((besseli(0, kappa)-besseli(0, kappa*cos(dThetaAllT/2)))/(kappa*besseli(1, kappa))))/2; % Theoretical prediction

% Random circle with sign ambiguity
beta = 4; [y0, a, b] = gencirc(N, beta, rho);
cErrorR = getnmc(@(x) sqc(x, y0, a, b), @poissrnd, nPair, dThetaAllN, nTrial);
c = sin(dThetaAllT/2);
ceTheoryR = erfc(rho*c.*sqrt(1-16*c/9/beta/pi)/sqrt(2))/2; % Theoretical prediction

figure
plot(dThetaAllT, ceTheoryC, 'Color', cmap(1, :)); hold on
errorbar(dThetaAllN, mean(cErrorC), std(cErrorC), 'o', 'Color', cmap(1, :), 'MarkerSize', 8)
plot(dThetaAllT, ceTheoryR, 'Color', cmap(2, :)); hold on
errorbar(dThetaAllN, mean(cErrorR), std(cErrorR), 'o', 'Color', cmap(2, :), 'MarkerSize', 8)
plot(dThetaAllT, ceTheoryV, 'Color', cmap(3, :)); hold on
errorbar(dThetaAllN, mean(cErrorV), std(cErrorV), 'o', 'Color', cmap(3, :), 'MarkerSize', 8)
yscale log; xlabel('\Delta\theta'); ylabel('Classification error')
legend('Square-of-circle (SoC)', '', 'SoC w/ sign ambiguity', '', 'Von Mises', '')

%% Importance of matching tuning curve with noise model (Fig. 3d)
sigmaAll = 0:4; nSigma = length(sigmaAll);
N = 100; dTheta = pi/2; rho = 3.5; nPair = 30; nSample = 1e3;
cError = zeros(nSigma); K = zeros(nSigma);
h = waitbar(0);
for iSigma = 1 : nSigma % Used to construct the tuning curve
    sigmai = sigmaAll(iSigma);
    for jSigma = 1 : nSigma % Used to generate spike counts
        sigmaj = sigmaAll(jSigma);

        [y0, a, b] = gencircnneg(N, rho);
        if sigmaj ~= sigmai % Adjust rho to have constant FI
            [y0, a, b] = gencircnneg(N, rho);
            [y0, a, b, k] = adjustrho(rho, sigmai, sigmaj, y0, a, b);
            K(iSigma, jSigma) = k;            
        end
        if sigmai == 0
            rFun = @(x) sqc(x, y0, a, b);
        else
            rFun = @(x) gpc(x, y0, a, b, sigmai);
        end
        
        if sigmaj == 0
            xFun = @poissrnd;
        else
            xFun = @(x) nbinrnd(sigmaj^(-2), 1./(x*sigmaj^2+1));
        end

        cError(iSigma, jSigma) = mean(getnmclc(rFun, xFun, nPair, dTheta, nSample));
        waitbar(((iSigma-1)*nSigma+jSigma)/nSigma^2, h)
    end
end
delete(h)

% Normalize
cErrorNorm = cError;
for iSigma = 1 : nSigma
    for jSigma = 1 : nSigma
        if iSigma == jSigma
            cErrorNorm(iSigma, jSigma) = 1;
        else
            cErrorNorm(iSigma, jSigma) = cError(iSigma, jSigma) / min(cError(iSigma, iSigma), cError(jSigma, jSigma));
        end
    end
end

figure; imagesc(cErrorNorm); colormap sky; axis equal
colorbar('southoutside', 'Ticks', 1:6, 'TickLabels', {'1', '2', '3', '4', '5', '6'})

%% Linearized recurrent network performance (Fig. 6c)
N = 100; g = 1; rho = 3.5; beta = 4; [y0, a, b] = gencirc(N, beta, rho);
dTheta = pi/2; nPair = 30; nSample = 1e3; nRepeat = 100;

ceMatch = zeros(1, nRepeat); h = waitbar(0);
for iRepeat = 1 : nRepeat
    W = randn(N) * g / sqrt(N);
    rFun = @(x) wtor(x, y0, a, b, W);
    ceMatch(iRepeat) = mean(getnmcw(rFun, W, nPair, dTheta, nSample));
    waitbar(iRepeat/nRepeat, h)
end
delete(h)

ceMismatch = zeros(1, nRepeat); K = zeros(1, nRepeat); h = waitbar(0);
for iRepeat = 84 : nRepeat
    W1 = randn(N) * g / sqrt(N); W2 = randn(N) * g / sqrt(N);
    [y01, a1, b1, k] = adjustrhow(rho, W1, W2, y0, a, b); K(iRepeat) = k;
    rFun = @(x) wtor(x, y01, a1, b1, W1);
    ceMismatch(iRepeat) = mean(getnmcw(rFun, W2, nPair, dTheta, nSample));
    waitbar(iRepeat/nRepeat, h)
end
delete(h)

figure; histogram(ceMatch/mean(ceMatch), 'BinWidth', 0.05); hold on
histogram(ceMismatch/mean(ceMatch), 'BinWidth', 0.05)
xlabel('Normalized classification error'); ylabel('Density')
legend('Matched', 'Mismatched')

%% Constraining the population firing rate (Fig. 7)
rho = 5; % Use r to control classification error magnitude
l0 = rho * 2 / sqrt(3); % Length constraint (divided by 2*pi). -1 if no constraint
N = 100; % Number of neurons, which is 1 dimension larger than the sphere
K = 10; % Largest Fourier mode
T = 100; theta = linspace(0, 2*pi, T+1); theta = theta(2:end)';
B = [ones(T, 1), cos(theta*(1:K)), sin(theta*(1:K))]; % Fourier bases

% Random initialization in Fourier space
Phi0 = randn(2*K+1, N);
Y0 = B * Phi0; Rho0 = sum(Y0.^2, 2); Phi0 = Phi0 / sqrt(mean(Rho0));

options = optimoptions('fmincon', 'MaxFunctionEvaluations', 1e8,...
    'MaxIterations', 1e5, 'PlotFcn', 'optimplotfvalconstr', 'Display', 'off');
[Phi, L] = fmincon(@(x) perrors(x, B), Phi0, [], [], [], [], [], [], @(x) lconsm(x, B, rho, l0), options);

% Neurometric curve
dThetaAll = 0 : 0.01 : (pi/2);
cError0 = gettnmc(Phi0, dThetaAll);
cError2 = gettnmc(Phi2, dThetaAll);
cError4 = gettnmc(Phi4, dThetaAll);
cError = gettnmc(Phi, dThetaAll);

figure; plot(dThetaAll, hfun(2/sqrt(3)*rho*sin(dThetaAll/2))); hold on
plot(dThetaAll, cError2); plot(dThetaAll, cError4); plot(dThetaAll, cError)
legend('Largest positive circle (LPC)', 'Length = 2 x LPC',...
    'Length = 4 x GC', 'Length unconstrained')
xlabel('\theta'); ylabel('Classification error')

%% The effect of sign ambiguity (Fig. S1)
nN = 10; Nall = ceil(logspace(1, 3, nN)); nRepeat = 30; 
beta = 4; rho0 = 0.35;
ceR = zeros(nRepeat, nN); ceTheory = zeros(1, nN);

options = optimoptions(@fminunc, 'Display', 'off');
h = waitbar(0);
for iN = 1 : nN
    N = Nall(iN); rho = rho0*sqrt(N); 
    ceTheory(iN) = 8*sqrt(2)/(9*exp(1)*pi^(3/2)*beta*rho);
    for iRepeat = 1 : nRepeat
        [y0, a, b] = gencirc(N, beta, rho);
        [~, ce] = fminunc(@(x) hfun(2*rho*sin(x/2)) - calcrce(x, y0, a, b), asin(min(sqrt(2)/rho, pi/4))*2, options);
        ceR(iRepeat, iN) = -ce;
        waitbar(((iN-1)*nRepeat+iRepeat)/(nN*nRepeat), h)
    end
end
delete(h)

figure; errorbar(Nall, mean(ceR), std(ceR)/sqrt(nRepeat), 'o', 'LineWidth', 2, 'MarkerSize', 8)
hold on; plot(Nall, ceTheory, 'color', cmap(1, :))
xscale log; yscale log; xlabel('N'); ylabel('max \Delta l')
legend('Exact', 'Lower bound')

%% Functions
function d = elleucd(t1, t2, a, b)
% Euclidean distance between points on the ellipse
d = sqrt(a^2*(cos(t1)-cos(t2)).^2+b^2*(sin(t1)-sin(t2)).^2);
end

function t = altot(theta, m)
% Go from arclength (theta) to ellipse parameter (t)
options = optimoptions('fsolve', 'Display', 'off');
b = pi/2 / ellipticE(m);
if theta < pi/2
    t = fsolve(@(x) b*ellipticE(x, m)-theta, theta, options);
elseif theta < pi
    t = pi - fsolve(@(x) b*ellipticE(x, m)-(pi-theta), pi-theta, options);
elseif theta < pi*3/2
    t = fsolve(@(x) b*ellipticE(x, m)-(theta-pi), theta-pi, options) + pi;
else
    t = 2*pi - fsolve(@(x) b*ellipticE(x, m)-(2*pi-theta), 2*pi-theta, options);
end
end

function r = sqc(theta, y0, a, b)
y = y0 + a*cos(theta) + b*sin(theta); r = y.^2/4;
end

function l = calcrce(dTheta, y0, a, b)
l = integral(@(x) hfun(norm(abs(y0+a*sin(x)+b*cos(x))-abs(y0+a*sin(x+dTheta)+b*cos(x+dTheta)))), 0, 2*pi, 'ArrayValued', true)/2/pi;
end

function r = gpc(theta, y0, a, b, sigma)
y = y0 + a*cos(theta) + b*sin(theta); 
r = (sinh(sigma*y/2)/sigma).^2;
end

function [y0, a, b, k] = adjustrho(rho, sigmai, sigmaj, y0, a, b)
options = optimoptions('fsolve', 'Display', 'off');
k = fsolve(@(x) integral(@(t) sqrtfiij(t, x, y0, a, b, sigmai, sigmaj), 0, 2*pi, 'ArrayValued', true)-2*pi*rho, 1, options);
a = a * k; b = b * k; y0 = sqrt(a.^2 + b.^2);
end

function l = sqrtfiij(t, x, y0, a, b, sigmai, sigmaj)
y = y0+x*(a*cos(t)+b*sin(t)); dy = x*(-a*sin(t)+b*cos(t));
if sigmai == 0
    l = norm(dy./sqrt(1+(sigmaj*y/2).^2));
elseif sigmaj == 0
    l = norm(dy.*cosh(sigmai*y/2));
else
    l = norm(cosh(sigmai*y/2)./sqrt(1+(sigmaj/sigmai*sinh(sigmai*y/2)).^2).*dy);
end
end

function [f, J] = fyr(x, y, W)
N = length(x);
f = W*(x.^2) - 2*x + y;
J = 2*W.*(x')- 2*eye(N);
end

function f = perrors(Phi, B)
% Loss for optimization on the sphere
% B is T by 2*K+1. Phi is 2*K+1 by N.
C = abs(B * Phi);
d = pdist(C); % Use Euclidean distance
h = hfun(d);
f = mean(h);
end

function [y, yeq] = lconsm(Phi, B, rho, l0)
% Constraints for a semantically monotonic curve on the sphere
C = abs(B * Phi);
% There is a 1-1 mapping of distance and inner product on the sphere
D = squareform(pdist(C, @(xi, Xj) Xj*xi'));
T = size(D, 1); n = ceil((T-1)/2);
% Semantically monotonic
yeq = 0; M = zeros(1, n);
for t = 1 : n
    d = [diag(D, t); diag(D, T-t)];
    m = mean(d); M(t) = m;
    yeq = yeq + sum((d-m).^2)/T; % Same dTheta maps to the same distance
end
y = diff(M); % Increasing dTheta leads to increasing distance and decreasing inner product
yeq = [yeq; sum(C.^2, 2)-rho^2]; % Constrained on the sphere
if l0 < 0
    yeq = [yeq; 0];
else
    yeq = [yeq; lsp(Phi, B) - l0]; % Constrain total length
end
end

function l = lsp(Phi, B)
% Speed of the curve (compare with rho)
N = size(Phi, 2);
K = (size(Phi, 1)-1) / 2;
dPhi = [zeros(1, N); (1:K)'.*Phi(K+2:end, :); -(1:K)'.*Phi(2:K+1, :)];
dC = B * dPhi; % T by N
l = mean(vecnorm(dC, 2, 2));
end

function cError = gettnmc(Phi, dThetaAll)
% Calculate theoretical neurometric curve
K = (size(Phi, 1)-1)/2;
ndTheta = length(dThetaAll);
cError = zeros(1, ndTheta);
for idTheta = 1 : ndTheta
    dTheta = dThetaAll(idTheta);
    cError(idTheta) = integral(@(x) hfun(vecnorm(ttoy(x, Phi, K)-ttoy(x+dTheta, Phi, K), 2, 2)), 0, 2*pi, 'ArrayValued', true) / (2*pi);
end
end

function y = ttoy(t, Phi, K)
y = abs([ones(numel(t), 1), cos(t(:)*(1:K)), sin(t(:)*(1:K))] * Phi);
end

function r = wtor(theta, y0, a, b, W)
options = optimoptions('fsolve', 'SpecifyObjectiveGradient', true, 'Display', 'off');
y = y0 + a * cos(theta) + b * sin(theta);
sqrtr = fsolve(@(x) fyr(x, y, W), abs(y)/2, options);
r = sqrtr.^2;
end

function [y0, a, b, k, feval, exitflag, output] = adjustrhow(rho, Wr, Wx, y0, a, b)
options = optimoptions('fsolve', 'Display', 'off');
[k, feval, exitflag, output] = fsolve(@(x) integral(@(t) sqrtfiw(t, x, y0, a, b, Wr, Wx), 0, 2*pi, 'ArrayValued', true)-2*pi*rho, 1, options);
a = a * k; b = b * k;
end

function l = sqrtfiw(t, x, y0, a, b, Wr, Wx)
options = optimoptions('fsolve', 'SpecifyObjectiveGradient', true, 'Display', 'off');
y = y0+x*(a*cos(t)+b*sin(t)); dy = x*(-a*sin(t)+b*cos(t));
sqrtr = fsolve(@(z) fyr(z, y, Wr), abs(y)/2, options);
Linv = diag(1./sqrtr); A = Linv - Wx; B = Linv - Wr;
l = norm(A*(B\dy));
end