function P = ProfileGrow(P, x, lambda, Thresh, Criterion, verbose)
% ProfileGrow     The growing part of profile update for Kernel RLS-DLA
%
% This function updates the profile of the dictionary according to the
% new update data based on the KRLS dictionary learning algorithm.
% This is a mini-batch way of updating the profile, where x may be a
% matrix with M columns, perhaps M=10 or perhaps M=20.
%
% use:
% Pout = ProfileGrow(Pin, x, lambda, verbose)
%    Pout, Pin    Profile in and out, this is a struct as in ProfileShow().
%    x            New incoming update data (size: NxM)
%                 also a sigle signal vector can be used, size Nx1  (M=1)
%                 note M and L are swapped from KRLSUpdate.
%    lambda       Forgetting factor, lambda <= 1.0 (applied once!), size: 1x1
%                 OBS, not the same as P.lambda stored in profile
%    verbose  Print warnings if true and warning is appropriate, (default: true)
%
% ex:
% P1 = ProfileGrow(P0, XTrain(:,31:40,1), 0.99, true);
% P2 = ProfileGrow(P1, XTrain(:,41:50,1));
%
% Generated by Ghasem Alipoor (alipoor@hut.ac.ir) and Karl Skretting (karl.skretting@uis.no)
% Last modification: 13 June 2023
%

if (nargin < 2)
    error('Must have at least 2 input arguments, see help.');
end
if (nargin < 3) || isempty(lambda)
    lambda = 1.0;
end
if (nargin < 4) || isempty(Thresh)
    Thresh = 0.95;
end
if (nargin < 5) || isempty(Criterion)
    Criterion = 'Projection';
end
if (nargin < 6) || isempty(verbose)
    verbose = true;
end

[N,M] = size(x);
if (N ~= P.N)
    error('New data x, size %ix%i, does not match profile', N, M);
end
[one1,one2] = size(lambda);  % should be scalar
if ~(isnumeric(lambda) && (one1 == 1) && (one2 == 1))
    % error('lamda is not numeric scalar');
    lambda = 1.0;
end

%
% sparse approximation, also use P.Psi as stored in profile
k = ker_eval(P.X, x, P.KernelType, P.KernelParam);    % LxM
h = P.U * k;                                          % QxM    Eq. (20)
sigmax = ker_eval(x, x, P.KernelType, P.KernelParam); % MxM
% disp([size(x), size(sigmax), size(P.Psi), size(h), size(k)]);
[w, r2] = myKOMP(P.Psi, h, diag(sigmax), P.s, 1e-6, false, false); % QxM  (r2 not used)
% Following controls on w elements are added.
if any(isnan(w(:)))
    if verbose
        fprintf('\n*** WARNING, NaN in w after SA (myKOMP) in %s\n',mfilename());
        ProfileShow(P, true);
        %         pause(5);
        w(isnan(w)) = 0.0;  % better with zero than NaN
    end
end

% Sparsification is applied by selecting the most informative nput
%         data samples based on some growing criteria.
% The new input sample is included in the profile if Growing Criterion
% Measure (Measure) is smaller than a pre-specified threshold (Thresh).
switch Criterion
    case 'Coherency'
        % Maximum coherency of the new data sample with data samples stored in the profile.
        % This coherency is calculated in the feature space simply using the k vector, after normalization.
        Measure = max(abs(k./sqrt(diag(P.K)*diag(sigmax)')));
    case 'Representation'
        Measure = 1 - r2'./sqrt(diag(sigmax));
    case 'Projection'
        Measure = zeros(M, 1);
        for j = 1:M
            Measure(j) = k(:, j)'*(P.K\k(:, j))/sigmax(j, j);
        end
end
idx = find(Measure < Thresh);   % Index of the data vectors that pass the growing criterion

% The following control is applied on the new coefficient vectors to
%       make sure that w is full-rank.
if rank(w(:, idx)) ~= length(idx)  % w is not full-rank
    jj = 1;
    while jj <= length(idx)
        if rank(w(:, idx(1:jj))) ~= jj
            idx(jj) = [];
        else
            jj = jj + 1;
        end
    end
end
M = length(idx);
x = x(:, idx);
k = k(:, idx);
w = w(:, idx);
sigmax = sigmax(idx, idx);

if ~isempty(idx)
    % some temporary variables, based on Pin
    u = P.C * w;                                      % QxM
    % alpha = eye(M)/(lambda*eye(M) + W'*u); disp(alpha); % is symmetric positive definte
    % next line may give warning: singular matrix ...
    ualpha = u/(lambda*eye(M) + w'*u);                % QxM
    ualphau = ualpha*u';                              % QxQ
    ualphau = (ualphau + ualphau')/2;                 % QxQ,  and make sure it is symmetric!
    v = repmat(P.lambda,1,M).*(P.W' * u);             % LxM,  needed for U and Psi
    %
    % grow profile
    P.L = P.L + M;
    P.X = [P.X, x];                                   % Nx(L+M)
    P.W = [P.W, w];                                   % Qx(L+M)
    P.lambda = [lambda*P.lambda; ones(M,1)];          % (L+M)x1

    if strcmpi(P.KernelType, 'Lin')
        r = x - P.D*w;                                % NxM,  r = phi - Phi*v
        P.D = P.D + r*ualpha';                        % NxQ,  Eq. (32)
    end

    P.C = (1/lambda)*(P.C - ualphau);             % QxQ,  Eq. (27)

    % Always use Eq. (33)
    uhat = P.U * (k - P.K*v);                         % QxM,  Eq. (34)
    uuh = ualpha*uhat';                               % Q*Q, includes alpha
    vk = v'*k;                                        % MxM
    vKv = (v'*P.K*v - vk - vk' + sigmax);             % MxM
    P.Psi = P.Psi + uuh + uuh' + ualpha*vKv*ualpha';  % QxQ,  Eq. (33)

    P.U = [P.U - ualpha*v', ualpha];     % Qx(L+M)       Eq. (30)
    P.K = [P.K, k; k', sigmax];          % (L+M)x(L+M)   Eq. (29)
end
if (min(diag(P.Psi)) < 1e-4) || (max(diag(P.Psi)) > 1e4)
    % Hmm dictionary too far from normalized, has happened but very rare
    if verbose
        fprintf('\nWARNING: %s, diag(P.Psi) NOT as it should be!\n',mfilename());
        %         pause(5);
    end
    P = ProfileNormalize(P, verbose);
end
end
