% Simplex method for the canonical linear programming formalism
% Author: Hao LI

% Model problem:
% min  c' * x  or  transpose(c) * x
% s.t. A * x = b
%      x >= 0
% x initial value: x_init
% basis initial indice: bas_init  e.g. [0,0,0,0,1,0,1,1,0]
function [x_opt, f_opt, bas_opt] = SimplexCanonical(c, A, b, x_init, bas_init, ishow)

if(nargin<6) ishow=false; end

x_opt = x_init; f_opt = c'*x_opt; bas_opt = bas_init;
% Preliminary checking
if (size(x_init,2)~=1 || size(b,2)~=1 || size(c,2)~=1) fprintf('ERROR: x_init, b, or c is not a vertical vector !\n'); return; end
if (size(A,2)~=length(c) || length(c)~=length(x_init) || size(A,1)~=size(b,1)) fprintf('ERROR: Dimension inconsistency among the input !\n'); return; end
if (~isempty(find(x_init<0))) fprintf('ERROR: x_init >= 0 is violated !\n'); return; end
if (~isempty(find(abs(A*x_init-b)>0.000000001))) fprintf('ERROR: A * x_init = b is violated ! max(abs(A*x_init-b)) = %f \n', max(abs(A*x_init-b))); return; end

% Step 0: Initialization
x = x_init;
fx = c'*x;
bas = bas_init;
iter_n = 0;
if(ishow) fprintf('START  SimplexCanonical\n'); end
if(ishow) fprintf('Iteration %d : x = %s ; fx = %f; basis = %s\n', iter_n, mat2str(x), fx, mat2str(bas)); end

while(1)
    iter_n = iter_n+1;
    
    % Step 1: Compute simplex multipliers and the vector of reduced costs
    %   Let A consists of B (basis columns) and N (nonbasis columns)
    %   B * dB_k = -A_k  => cost_k = c_k - cB'*inv(B)*A_k , k belongs to set_N
    %   cost_vec = cN' - cB'*inv(B)*N
    i_bas = find(bas>0);
    i_nonbas = find((ones(length(x),1)-bas)>0);
    cB = c(i_bas);
    cN = c(i_nonbas);
    B = A(:,i_bas);
    N = A(:,i_nonbas);
    
    
    invB = inv(B);
    yT = cB'*invB;
    rc_vec = zeros(length(x),1);
    rc_vec(i_nonbas) = cN' - yT*N;
    
    % Step 2: Optimality check
    %   If no improving simplex direction exists, STOP; current solution is optimal
    tol = 10*eps;
    rc_max = min(rc_vec);
    if (rc_max>=-tol)  % if (rc_max>=0)
        x_opt = x;
        f_opt = c'*x_opt;
        bas_opt = bas;
        if (abs(f_opt)<=tol) f_opt = 0; end
        break
    end
    i_imprv = find(rc_vec<-tol); % find(rc_vec<0);   
    % Choose the improving simplex direction with the smallest index
    i_enter = i_imprv(1);
    
    % Step 3: Compute the chosen improving simplex direction
    d_imprv = zeros(length(x),1);
    d_imprv(i_enter) = 1;
    d_imprv(i_bas) = -invB*A(:,i_enter);
        
    % Step 4: Compute the maximum step size
    nmd_max = 1/eps;
    for j = 1:length(x)
        if (d_imprv(j)>=0 || x(j)<=tol) % if (d_imprv(j)>=0)
            continue
        end
        nmd = x(j)/(-d_imprv(j));
        if (nmd<nmd_max)
            nmd_max = nmd;
            % Choose the leaving variable with the smallest index among the basis 
            i_leave = j;
        end
    end
    % Can no longer move
    if (nmd_max == 1/eps) 
        x_opt = x;
        f_opt = c'*x_opt;
        bas_opt = bas;        
        if (abs(f_opt)<=tol) f_opt = 0; end
        break
    end        
    
    % Step 5: Update the solution & basis
    bas(i_leave) = 0;
    bas(i_enter) = 1;
    x = x + nmd_max * d_imprv; 
    fx = c'*x;
    if(ishow) fprintf('Iteration %d : x = %s ; fx = %f; basis = %s\n', iter_n, mat2str(x), fx, mat2str(bas)); end
end

if(ishow) fprintf('Optimal solution : x_opt = %s ; f_opt = %f; basis = %s\n', mat2str(x_opt), f_opt, mat2str(bas_opt)); end
if(ishow) fprintf('END  SimplexCanonical\n'); end

if(0) %% NEVER EXECUTE
% EXAMPLE 1: [x;y;s1;s2;s4;s5;a1;a3;a5] 9-element vector
c = [0;0;0;0;0;0;1;1;1]; % corresponding to a1, a3, a5
b = [2;44;1;27;4];
A = zeros(5,9);
A(:,1) = [5;2;-3;2;1]; A(:,2) = [-1;5;1;-5;1]; 
A(1,3) = -1; A(2,4) = 1; A(4,5) = 1; A(5,6) = -1;
A(1,7) = 1; A(3,8) = 1; A(5,9) = 1;
x_init = [0;0;0;44;27;0;2;1;4];
bas_init = [0;0;0;1;1;0;1;1;1];

end

end