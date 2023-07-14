function P = OKDLFBUpdate(X, P, varargin)

%
% OKDLFBUpdate
% This function updates the profile of the dictionary according to the online
% kernel dictionary learningalgorithm, proposed in the following paper:
%     J. Lee and S. Kim, "Online kernel dictionary learning on a budget," 2016
%     50th Asilomar Conference on Signals, Systems and Computers, 2016,
%     pp. 1535-1539,doi: 10.1109/ACSSC.2016.7869635.
%
% Generated by Ghasem Alipoor (alipoor@hut.ac.ir) and Karl Skretting (karl.skretting@uis.no)
% Last modification: 13 June 2023
%

% In the refered paper, U and w variables are denoted as Omega and alpha, respectively.

pnames = {'GrowThresh', ...                       % Growing threshold
    'maxL', ...                                 % max value for profile L, ex 400
    'newL', ...                                 % Profile L after pruning, ex 300
    'lambda', ...                              % regularization parameter to trade-off the LS fit and the sparsity
    'mu', ...                                    % The parameter that adjusts the severity of the penalty
    'roh', ...
    };

dflts = {0.9, ... % GrowingTrsh
    200, 190, ...   % maxL, newL
    0.05, 0.1, .005, ... % lambda, mu, roh
    };
[GrowingTrsh, maxL, newL, lambda, mu, roh] = getarg(varargin,pnames,dflts);

[N, M] = size(X);       % Size of the input signal
L = size(P.X, 2);         % Profile size
if size(P.X, 1) ~= N
    error('Dimension of P.X is not proper.')
end
if size(P.W, 2) ~= L
    error('Dimension of P.W is not proper.')
end
if (size(P.K, 1) ~= L) || (size(P.K, 2) ~= L)
    error('Dimension of P.K is not proper.')
end
if (size(P.U, 1) ~= L) || (size(P.U, 2) ~= P.Q)
    error('Dimension of P.U is not proper.')
end
if length(P.Sigma_2) ~= L
    error('Dimension of P.Sigma_2 is not proper.')
end

% Main iteration
for i = 1: M
    % Step 1: get new signal x and calculate k and h vectors and sigma_2
    x = X(:, i);
    k = ker_eval(P.X, x, P.KernelType, P.KernelParam);                 % k(j) = <phi(P.X(:,j)),phi(x)>
    sigma_2 = ker_eval(x, x, P.KernelType, P.KernelParam);         % sigma_2 = K(x,x) = <phi(x),phi(x)>
    h = P.U'*k;              % P.U have not been updated yet, so here they are for (i-1).

    % Step 2: Check for the informativeness (the growing criterion in profile abstraction) of the new sample
    % Maximum coherency of the new data sample with data samples stored in the profile.
    % This coherency is calculated in the feature space simply using the k vector, after normalization.
    ki_normalized = k./sqrt(sigma_2*P.Sigma_2);
    GrowingCriterionMeasure = max(abs(ki_normalized));
    if GrowingCriterionMeasure < GrowingTrsh
        % Step 3: Sparse representation
        option.SCMethod = 'l1lsIP';
        option.lambda = lambda;
        w = KSRSC(P.Psi, h, sigma_2, option);

        % Step 4: Update profile, when a new data sample is included.
        % The new input sample is included in the profile if GrowingCriterionMeasure is
        % smaller than a pre-specified threshold, GrowingTrsh. If x passes this critrion,
        % continue with the next steps, otherwise ignore x and get the next sample.
        L = L + 1;

        P.X = cat(2, P.X, x);           % Note that scaling can not be done in signal space
        P.W = cat(2, P.W , w);
        P.K = cat(1, cat(2, P.K, k), cat(2, k', sigma_2));
        P.U = cat(1, P.U*((1 - mu*roh)*eye(P.Q) - roh*(w*w')), roh*w');
        P.Psi = P.U'*P.K*P.U;
        P.Sigma_2 = cat(1, P.Sigma_2, sigma_2);
    end
end

% Step 5: Profile abstraction by discarding a sample data, when the profile grows
%     beyond an acceptble size, i.e. MaxProfileSize
%     This exculsion also have effect on some other matrices and variables.
if L > maxL
    % Step 5a: Finding the entry to be discarded
    % Indices of the data sample in reverse order of the cnontribution
    [~,m] = mink(diag(P.W'*P.W), L - newL);

    % Step 5b: updating the corresponding matrices and variables
    P.X(:, m)         = [];
    P.W(:, m)        = [];
    P.K(:,m)          = [];
    P.K(m,:)          = [];
    P.U(m, :)         = [];
    P.Sigma_2(m) = [];
end
end
