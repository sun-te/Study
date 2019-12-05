% Simplex (two-phase) method for the standard linear programming formalism
% Author: Hao LI
% Dependence: SimplexCanonical.m 

% Model problem: standard linear program
% min  c' * x  or  transpose(c) * x
% s.t. A * x <= b
%      x >= 0  
function [x_opt, f_opt] = SimplexStandard(c, A, b, ishow)
size(A,2)
if(nargin<4) ishow=false; end
% Preliminary checking
if (size(b,2)~=1 || size(c,2)~=1) fprintf('ERROR: b or c is not a vertical vector !\n'); return; end
if (size(A,2)~=length(c) || size(A,1)~=size(b,1)) fprintf('ERROR: Dimension inconsistency among the input !\n'); return; end

if(ishow) fprintf('START  SimplexStandard (Two-Phase Method)\n'); end

% Phase-I linear program
% A: r x n
% original vars: x1,...,xn ; slack vars: s1,...,sr ; artificial vars: a{i}
if(ishow) fprintf('START  SimplexStandard => Phase-I linear program\n'); end
n = size(A,2); rs = size(A,1);
i_slck = find(b>=0);
i_artf = find(b<0); ra = length(i_artf);
if (~isempty(i_artf))
    A_aug = [A, eye(rs), zeros(rs,ra)];
    for k=1:ra
        A_aug(i_artf(k),n+rs+k) = -1;
    end
    x_aug = zeros(n+rs+ra,1); bas_aug = zeros(n+rs+ra,1);
    x_aug(n+i_slck,1) = b(i_slck); bas_aug(n+i_slck,1) = 1;
    x_aug(n+rs+1:n+rs+ra,1) = -b(i_artf); bas_aug(n+rs+1:n+rs+ra,1) = 1;
    c_aug = zeros(n+rs+ra,1); c_aug(n+rs+1:n+rs+ra,1) = 1;
    f_aug_init = c_aug'*x_aug;

    [x_aug_opt, f_aug_opt, bas_aug_opt] = SimplexCanonical(c_aug, A_aug, b, x_aug, bas_aug, ishow);

    if (f_aug_opt>100*eps) %if (f_aug_opt>min(1,f_aug_init)*0.0000001)
        fprintf('WARN: The original linear program has no feasible solution !\n');
        return
    end    
    
    x_aug = x_aug_opt(1:n+rs,1);
    bas_aug = bas_aug_opt(1:n+rs,1);
    i_nonbas = find(bas_aug<1);
    if (length(i_nonbas)>n)
        bas_aug(i_nonbas(n+1:end)) = 1;
    end
else
    x_aug = zeros(n+rs,1); bas_aug = zeros(n+rs,1);
    x_aug(n+1:n+rs,1) = b; bas_aug(n+1:n+rs,1) = 1;
end
if(ishow) fprintf('END  SimplexStandard => Phase-I linear program\n'); end
if(ishow) fprintf('Initial feasible solution : x_init = %s ; bas_init = %s\n', mat2str(x_aug), mat2str(bas_aug)); end

% Phase-II linear program
if(ishow) fprintf('START  SimplexStandard => Phase-II linear program\n'); end
A_aug = [A, eye(rs)]; c_aug = [c; zeros(rs,1)];
[x_aug_opt, f_aug_opt, bas_aug_opt] = SimplexCanonical(c_aug, A_aug, b, x_aug, bas_aug, ishow);
x_opt = x_aug_opt(1:n,1); f_opt = c'*x_opt; 
if(ishow) fprintf('END  SimplexStandard => Phase-II linear program\n'); end
if(ishow) fprintf('Optimal solution : x_opt = %s ; f_opt = %f\n', mat2str(x_opt), f_opt); end
if(ishow) fprintf('END  SimplexStandard (Two-Phase Method)\n'); end

if(0) %% NEVER EXECUTE
% EXAMPLE 1: 
c = [-6;-5]; % - (6 x + 5 y)
A = [-5,1;2,5;3,-1;2,-5;-1,-1];
b = [-2;44;-1;27;-4];
[x_opt, f_opt] = SimplexStandard(c, A, b, true)
end
    
end

    