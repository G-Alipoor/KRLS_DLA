function [W,phi_2] = KOMP_Nguyen(K,k,A,S)

%
% KOMP_Nguyen
% Kernel Orthogonal Matching Pursuit, as proposed in (Fig. 1):
%     H. Van Nguyen, V. M. Patel, N. M. Nasrabadi and R. Chellappa,
%     "Design of Non-Linear Kernel Dictionaries for Object Recognition,"
%     IEEE Transactions on Image Processing, vol. 22, no. 12, pp. 5123-5135,
%     Dec. 2013, doi: 10.1109/TIP.2013.2282078.
%
%   [W,phi_2] = KOMP_Nguyen(K,k,A,S)
%
% Inputs:
%   K: Gram matrix of the traing samples in the feature space; K(i,j) = k(x_i,x_j)
%   k: Kernel vector/matrix; K(i,j) = k(x_i,z_j), where z_j is the jth sample to be represented
%   A: Coefficients matrix representing the dictionary, i.e. D=Phi*A
%   S: Sparsity, i.e. number of the non-zero elements in the representation
% Outputs:
%   W:   Weight vector/matrix
%   phi_2: 2-norm of the representation vector/vectors in the feature space
%
% Generated by Ghasem Alipoor (alipoor@hut.ac.ir) and Karl Skretting (karl.skretting@uis.no)
% Last modification: 13 June 2023
%

N = size(K,2);          % Number of the training sapmles
if size(K,1) ~= N
    error('The first input argument (Gram matrix) must be square.')
end
L = size(k,2);            % Number of the testing sapmles
if size(k,1) ~= N
    error('Dimension mismatch!')
end
Q = size(A,2);            % Number of atoms/basis

W = zeros(Q,L);
if nargout > 1
    phi_2 = zeros(L,1);
end

if nargin < 4
    error('Not enough input arguments')
end

if S > Q
    error('Sparsity can not be greater than the number of atoms')
end

for i = 1:L
    Is   = zeros(1,S);
    vs  = zeros(N,1);
    AIs = [];
    for s = 1:S
        Tau = (k(:,i)' - vs'*K)*A;
        AbsTau = abs(Tau);
        AbsTau(Is(find(Is))) = -1;      %#ok<FNDSB> % To exclude Is indixes from searching
        [~,TempInd] = max(AbsTau);
        Is(s) = TempInd;

        AIs = [AIs A(:,TempInd)]; %#ok<AGROW>

        ws = pinv(AIs'*K*AIs)*(k(:,i)'*AIs)';         % xs in the paper
        vs = AIs*ws;
    end
    if nargout > 1
        phi_2(i) = k(:,i)'*vs;
    end

    w = zeros(Q,1);
    w(Is) = ws;
    W(:,i) = w;
end