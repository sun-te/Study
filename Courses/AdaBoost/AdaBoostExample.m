% AdaBoot Example

y_sample = [-1, +1, +1, -1, +1, +1, +1, +1, +1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1];
samples = [1:length(y_sample); y_sample];

y_weak = [-1, +1, +1, -1, -1, +1, +1, +1, +1, +1, +1, -1, +1, +1, -1, -1, +1, +1, -1, +1;
-1, -1, +1, -1, -1, +1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, -1;
-1, +1, +1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, +1, -1, +1, +1, +1;
+1, +1, -1, -1, -1, +1, +1, +1, +1, +1, -1, +1, -1, +1, -1, -1, +1, +1, -1, -1;
+1, +1, -1, +1, +1, +1, +1, -1, +1, +1, -1, -1, +1, +1, -1, -1, -1, -1, -1, -1;
+1, +1, -1, +1, -1, -1, +1, +1, +1, -1, +1, -1, +1, -1, -1, -1, -1, -1, -1, -1;
-1, +1, -1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, +1, -1, -1, -1, +1, -1, -1;
-1, -1, -1, -1, -1, +1, +1, +1, -1, -1, -1, +1, +1, +1, -1, -1, -1, -1, +1, -1;
-1, -1, -1, +1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, -1, -1, +1, -1;
-1, -1, +1, +1, +1, +1, +1, +1, +1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1;
+1, +1, -1, -1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, -1;
+1, -1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, +1, +1, -1, -1, +1, +1, -1, -1;
-1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, -1, +1, -1, +1;
-1, +1, -1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, +1, -1, -1, +1;
-1, -1, +1, +1, -1, +1, -1, -1, +1, -1, +1, -1, -1, -1, -1, -1, -1, -1, +1, -1;
-1, +1, -1, +1, +1, +1, +1, +1, +1, +1, -1, +1, -1, +1, +1, -1, -1, -1, +1, -1];

weak_N = size(y_weak,1);
weak_L = zeros(2,size(samples,2), weak_N);
for k = 1:weak_N
    weak_L(:,:,k) = [1:size(samples,2); y_weak(k,:)];
    num = 0;
    for k2 = 1:size(samples,2)
        if weak_L(2,k2,k) == samples(2,k2)
            num = num+1;
        end
    end
    fprintf('Weak learner %d => correctness percentage: %f%%\n', k, 100*num/size(samples,2));
end
data_N = length(y_sample);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TO COMPLETE CODE BELOW
%% Final adaboost_L(x) = alf(2,1)*weak_L(2,:,alf(1,1)) + alf(2,2)*weak_L(2,:,alf(1,2)) + ... + alf(2,weak_N)*weak_L(2,:,alf(1,weak_N))
alf = zeros(2,weak_N); %% To be computed
wgt = ones(1,size(samples,2))/size(samples,2);
for k = 1:weak_N
    t = 1;
    w = 1;
    e = 1;
    best_learner = 1;
    for ll = 1:weak_N
        tmp_e = 0;
        for d = 1:data_N
            if (y_weak(ll, d) ~= y_sample(d))
                tmp_e = tmp_e + wgt(d);
            end
        end
        if tmp_e < e
            e = tmp_e;
            best_learner = ll;
        end
    end
    alf(2, k) = 0.5 * log((1-e)/e);
    alf(1, k) = best_learner;
    
    for d = 1:data_N
        wgt(d) = wgt(d) * exp(-alf(2, k) * y_sample(d) * y_weak(alf(1, k), d));
    end
    
end

    


%% ADD CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

adaboost_L = zeros(1,size(samples,2));
for k = 1:weak_N
    adaboost_L = adaboost_L + alf(2,k)*weak_L(2,:,alf(1,k));
end
for idx = 1:size(samples,2)
    if adaboost_L(idx)>0
        adaboost_L(idx) = 1;
    else
        adaboost_L(idx) = -1;
    end
end
fprintf('Adaboost Learner => correctness percentage: %f%%\n', 100*sum(adaboost_L==samples(2,:))/size(samples,2));

