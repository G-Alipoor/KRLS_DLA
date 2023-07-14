function [W,r2] = myKOMP(DtD, DtX, dXtX, tnz, tre, doOMP, verbose)
% myKOMP          Sparse approximation in kernel space, my implementation.
%
% This implementation is basically the OMP/ORMP implementation presented in
% Fig. 3 in Partial Search paper presented at NORSIG 2003, by Skretting and Husøy.
% (http://www.ux.uis.no/~karlsk/proj02/norsig2003.pdf)
%
% input DtD:  Innerproducts in kernel space of dictionary atoms, size KxK
%             dictionary does not need to be normalized, i.e. DtD(i,i) = 1
%             This is the Gram  matrix of the dictionary (D) in kernel
%             space, the same as Si {\Psi} in KOMP.m, (and K here is Q there)
%       DtX:  Innerproducts in kernel space of signal(s) and dictionary.
%             actually it is not D'*X but rather D' * f_phi(X), size KxL
%             This is kernel vector/matrix h in KOMP.m
%       dXtX: Innerproducts in kernel space of each signal to itself.
%             dXtX(i) = diag(kernelFun(X,X) = f_phi(X)'*f_phi(X)
%             = ||X(:,i)||^2  (squared norm), size of dXtX should be 1xL or Lx1
%             This is Sigma in KOMP.m
%       tnz:  Number of atoms to use for each approximation, 1x1 or 1xL
%             Sparsity given as target number of non-zero coefficients.
%       tre:  target relative error squared (scalar size 1x1), default 1e-8
%             This is Eps in KOMP.m
%       doOMP:   true if OMP, otherwise (false) the ORMP variant is done.
%       verbose: true or false
% output  W:  The sparse coefficient vectors, size KxL
%         r2: norm squared of residual, size 1xL

%----------------------------------------------------------------------
% Made by Karl Skretting, University of Stavanger (UiS),
% Department of Electrical Engineering and Computer Science (IDE)
% Mail:  karl.skretting@uis.no   Homepage:  http://www.ux.uis.no/~karlsk/
%
% HISTORY:  dd.mm.yyyy
% Ver. 1.0  11.10.2018  KS: made file
%----------------------------------------------------------------------


[K,L] = size(DtX);
W = zeros(K,L);
r2 = zeros(1,L);
if (nargin < 5); tre = 1e-8*ones(L,1); end
if (nargin < 6); doOMP = true; end
if (nargin < 7); verbose = true; end

if doOMP
    textMethod = 'Orthogonal Matching Pursuit.';
else
    textMethod = 'Order Recursive Matching Pursuit.';
end
if verbose
    fprintf('%s: %s L=%i signals and K=%i dictionary atoms.\n', mfilename(), textMethod, L, K);
end

dd = diag(DtD);
if not(numel(dd) == K)
    fprintf('Error, size mismatch.\n')
    return
end
if (min(dd) < 0.999) || (max(dd) > 1.001)
    %     fprintf('Not normalized input dictionary, normalize it here.\n')
    dd = reshape(1./sqrt(dd),K,1);  % size Kx1
    DtD = DtD .* (dd*dd');
    DtX = DtX .* repmat(dd,1,L);
else
    dd = [];
end
if (numel(tnz) == 1)
    tnz = tnz*ones(1,L);
end
if not(numel(tnz) == L)
    fprintf('Error, size of tnz should be 1 or L.\n')
    return
end
if (numel(tre) == 1)
    tre = tre*ones(1,L);
end
if not(numel(tre) == L)
    fprintf('Error, size of tre should be 1 or L.\n')
    return
end

for columnNumber = 1:L        % i.e. for each data vector
    % **********************  INITIALIZE  **********************
    S = tnz(columnNumber);
    r = zeros(S,K);
    w = zeros(K,1);
    T = 1:K;
    e = ones(K,1);
    u = ones(K,1);
    c = DtX(:,columnNumber);
    n2x = dXtX(columnNumber);
    n2xLim = n2x*tre(columnNumber);
    % select the first frame vector
    [cm,km] = max(abs(c));
    s = 1;
    J = km;
    T(km) = -1;
    r(1,km) = u(km);
    n2x = n2x-cm*cm;
    % **********************  THE MAIN LOOP  **********************
    while ((s<S) && (n2x>n2xLim))
        for k=1:K
            if (T(k)>=0)
                r(s,k) = DtD(km,k);
                for n=1:(s-1)
                    r(s,k) = r(s,k)-r(n,km)*r(n,k);
                end
                if not(u(km)==0); r(s,k) = r(s,k)/u(km); end
                c(k) = c(k)*u(k)-c(km)*r(s,k);
                if doOMP  % use next line for OMP
                    w(k) = abs(c(k));  % use w here (instead of a new variable d)
                end
                e(k) = e(k)-r(s,k)*r(s,k);
                u(k) = sqrt(abs(e(k)));  % abs kun i matlab!
                if not(u(k)==0); c(k) = c(k)/u(k); end
                if not(doOMP)  % use next line for ORMP
                    w(k) = abs(c(k));     % use w here (instead of a new variable d)
                end
            end
        end
        w(km) = 0;   % w(J) = 0;
        % select the next dictionary atom
        [temp,km] = max(abs(w)); %#ok<ASGLU>
        s = s+1;
        J(s) = km;
        T(km) = -1;
        r(s,km) = u(km);
        cm = c(km);
        n2x = n2x-cm*cm;
    end  % ******** END OF MAIN LOOP **********************************

    % ************ BACK-SUBSTITUTION *************
    w = zeros(K,1);
    for k=s:(-1):1
        Jk=J(k);
        for n=s:(-1):(k+1)
            c(Jk) = c(Jk)-c(J(n))*r(k,J(n));
        end
        if not(r(k,Jk)==0); c(Jk) = c(Jk)/r(k,Jk); end
        w(Jk) = c(Jk);
    end
    %
    W(:,columnNumber) = w;
    r2(columnNumber) = n2x;
end

if numel(dd) == K   % the dictionary was normalized here
    W = W .* repmat(dd,1,L);     % rescale W
end

return