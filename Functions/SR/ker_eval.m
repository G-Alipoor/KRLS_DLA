function y = ker_eval(X1,X2,ker_type,ker_param)

%
% ker_eval
% Kernel Evaluation
%    Modified and verified by G. Alipoor (alipoor@hut.ac.ir)
%    Last Modifications October, 12th 2018
%
%   y = ker_eval(X1,X2,ker_type,ker_param)
%   This function is used to evaluate the kernel function ker_type with the
%   ker_param parameter over the X1 and X2 vector/matrix inputs.
%
% X1:       First vector/matrix input
% X2:       Second vector/matrix input
% ker_type: Kernel type, can be either Gaussian (Gauss), Polynomial (Poly)
%
% Outputs:
% y:        K(X1,X2)
%
if nargin < 4
    ker_param = 2;
end

N1 = size(X1,2);
N2 = size(X2,2);

if strcmpi(ker_type,'Gauss')
    y = zeros(N1,N2);
    for i = 1:N1
        for j = 1:N2
            y(i,j) = exp(-sum((X1(:,i)-X2(:,j)).^2,1)/(2*ker_param^2));
        end
    end
end
if strcmpi(ker_type,'Poly')
    y = (1 + X1'*X2).^ker_param;
end
if strcmpi(ker_type,'Lin')
    y = X1'*X2;
end

% if min (N1, N2) == 1
%     y = y(:);
% end
