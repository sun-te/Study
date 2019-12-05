% Support Vector Machine (SVM)
% Author: Hao LI
% Dependence: SimplexStandard.m 

function demoSVMToBeComplete()

    % Model problem: 
    % min  nmd * ||w||^2 + (1/n) * sum(e)
    % s.t. e = (e1, e2, ..., en) >= 0
    %      yi*(xi'*w-b) + ei - 1 >= 0, i = 1,2, ..., n
    % Separating hyperplane: w'x - b = 0
    % Input:
    % X = [x1, x2, ..., xn]   size length(x1) x n
    % Y = [y1, y2, ..., yn]'  size n x 1
    function [w_opt, b_opt] = SVM(X, Y, nmd)

    if(nargin<3) nmd = 1; end
    n = length(Y);
    m = size(X,1);
    size(X)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Linear programming to get the initial value of w, b, e
    c = zeros(2*m+2+n,1); c(1:n,1) = 1; c(n+1:end,1) = 0.000001;
    % A_simplex * [e, wp, wn, bp, bn] <= b_simplex
    % w = wp - wn
    % b = bp - bn
    col1 = -eye(n);
    col2 = -diag(Y)*X';
    size(diag(Y))
    tmp = cat(2, col1, col2, -col2, Y, -Y); 
    A_simplex = tmp;%% A_simplex is what ?? TO BE COMPLETED
    b_simplex = -ones(n, 1);%% b_simplex is what ?? TO BE COMPLETED
    % c, A_simplex, b_simplex, pause %%%%%%
    [W_opt, f_opt] = SimplexStandard(c, A_simplex, b_simplex, true);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    e_init = W_opt(1:n,1);
    w_init = W_opt(n+1:n+m,1) - W_opt(n+m+1:n+2*m,1);
    b_init = W_opt(n+2*m+1,1) - W_opt(n+2*m+2,1);

    %% Quadratic programming to optimize w, b, e
    wbe = [w_init; b_init; e_init];
    fprintf('w_init = %s; b_init = %s\n', mat2str(w_init), num2str(b_init));
    A = [diag(Y)*X', -Y, eye(n), -ones(n,1)];
    function f = Fsvm(wbe)
        wtmp = wbe(1:m); btmp = wbe(m+1); etmp = wbe(m+2:m+n+1);
        f = nmd*wtmp'*wtmp + sum(etmp)/n;       
    end
    function f = FsvmPenalty(wbe, k)
        wtmp = wbe(1:m); btmp = wbe(m+1); etmp = wbe(m+2:m+n+1);
        f = nmd*wtmp'*wtmp + sum(etmp)/n;
        % Penalty part
        st = [A*[wbe;1]; etmp]; 
        idx = find(st<0); f = f + k*st(idx)'*st(idx);
    end
    deltaT = 0.000001;
    dMat = deltaT*eye(length(wbe));
    function df = dFsvmPenalty(wbe, k)
        df = zeros(size(wbe));
        for idx = 1:length(wbe)
            df(idx) = (FsvmPenalty(wbe+dMat(:,idx), k)-FsvmPenalty(wbe-dMat(:,idx), k))/(2*deltaT);
        end
    end

    % Golden section 
    function wbe_o = OneDimGoldenSearch(wbe_i, df, k, tol)
        sL = 0; sR = 1;
        fsL = FsvmPenalty(wbe_i, k);
        while (FsvmPenalty(wbe_i+df*sR, k)<fsL) sR = 2*sR; end
        while (FsvmPenalty(wbe_i+df*sR, k)>fsL) sR = 0.9*sR; end
        fprintf('Local probing range: sR = %f\n', sR);
        mL = 0.382*sR; mR = 0.618*sR;
        fmL = FsvmPenalty(wbe_i+df*mL, k); fmR = FsvmPenalty(wbe_i+df*mR, k); fsR = FsvmPenalty(wbe_i+df*sR, k);
        while(sR-sL>=tol)
            if (fmL <= fmR)
                sR = mR; fsR = fmR; 
                mR = mL; fmR = fmL;
                mL = sR - 0.618*(sR-sL); fmL = FsvmPenalty(wbe_i+df*mL, k);
            else
                sL = mL; fsL = fmL;
                mL = mR; fmL = fmR;
                mR = sL + 0.618*(sR-sL); fmR = FsvmPenalty(wbe_i+df*mR, k);
            end
        end
        fmin = min([fsL, fmL, fmR, fsR]);
        if (fsL == fmin)
            s_opt = sL;
        elseif (fsR == fmin)
            s_opt = sR;
        else
            s_opt = (mL+mR)/2;
        end
        wbe_o = wbe_i+df*s_opt;
    end


    k_penalty = 100;
    % fprintf('Penalty factor k_penalty = %d\n', k_penalty);
    fprintf('Initial: f(wbe) = %f\n', FsvmPenalty(wbe,k_penalty));
    wbe_old = wbe - 1;
    while(sum(abs(wbe-wbe_old))/sum(abs(wbe))>10^-5)
        wbe_old = wbe;
        df = -dFsvmPenalty(wbe, k_penalty);
        wbe = OneDimGoldenSearch(wbe, df, k_penalty, 10^-9);
        fprintf('Golden section: f(wbe) = %f\n', FsvmPenalty(wbe,k_penalty));
    end
    fprintf('Final f_opt = %f\n', FsvmPenalty(wbe,k_penalty));

    w_opt = wbe(1:m); b_opt = wbe(m+1);
    fprintf('w_opt = %s; b_opt = %s\n', mat2str(w_opt), num2str(b_opt));

    end

n = 100;
xp = 1-rand(1,n)*8; yp = -5+10*rand(1,n); zp = 10*rand(1,n);
xn = -1+rand(1,n)*8; yn = -5+10*rand(1,n); zn = 10*rand(1,n);
X = [[xp;yp;zp], [xn;yn;zn]];
Y = [ones(n,1); -ones(n,1)];

[w_opt, b_opt] = SVM(X, Y, 1);


figure(1),
plot3(xp, yp, zp, 'or', xn, yn, zn, 'ob'); grid on; axis equal;
% w'x - b = {0,+1,-1}
ye1 = -5; ye2 = 5; ze = 10;
hold on; 
patch([(b_opt-w_opt(2)*ye1)/w_opt(1), (b_opt-w_opt(2)*ye2)/w_opt(1), (b_opt-w_opt(2)*ye2-w_opt(3)*ze)/w_opt(1), (b_opt-w_opt(2)*ye1-w_opt(3)*ze)/w_opt(1)], ...
    [ye1, ye2, ye2, ye1], [0, 0, ze, ze], [0,0,0]);
patch([(b_opt+1-w_opt(2)*ye1)/w_opt(1), (b_opt+1-w_opt(2)*ye2)/w_opt(1), (b_opt+1-w_opt(2)*ye2-w_opt(3)*ze)/w_opt(1), (b_opt+1-w_opt(2)*ye1-w_opt(3)*ze)/w_opt(1)], ...
    [ye1, ye2, ye2, ye1], [0, 0, ze, ze], [1,0,0]);
patch([(b_opt-1-w_opt(2)*ye1)/w_opt(1), (b_opt-1-w_opt(2)*ye2)/w_opt(1), (b_opt-1-w_opt(2)*ye2-w_opt(3)*ze)/w_opt(1), (b_opt-1-w_opt(2)*ye1-w_opt(3)*ze)/w_opt(1)], ...
    [ye1, ye2, ye2, ye1], [0, 0, ze, ze], [0,0,1]);
hold off;
    
end