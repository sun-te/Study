% Maximum Likelihood - Graph SLAM Loop Closure

function ML_GraphSLAMLoopClosure()

    % posb compound-added by posa
    function pos = comp_add(posa, posb)
        anga = posa(3);
        pos = [posb(1)*cos(anga)-posb(2)*sin(anga)+posa(1); ...
            posb(1)*sin(anga)+posb(2)*cos(anga)+posa(2); ...
            posb(3)+anga];
        while(pos(3)>pi)
            pos(3) = pos(3)-2*pi;
        end
        while(pos(3)<=-pi)
            pos(3) = pos(3)+2*pi;
        end
    end

    function pos = comp_inv(posa)
        ang = posa(3);
        pos = [-posa(1)*cos(ang)-posa(2)*sin(ang); posa(1)*sin(ang)-posa(2)*cos(ang); -ang];
        while(pos(3)>pi)
            pos(3) = pos(3)-2*pi;
        end
        while(pos(3)<=-pi)
            pos(3) = pos(3)+2*pi;
        end
    end

    % posb relative to posa
    function pos = comp_sub(posb, posa)
        pos = comp_add(comp_inv(posa), posb);
    end

L = 30;
% rpos(:,k) denotes the pose of pos(:,k+1) relative to pos(:,k) (pos(:,end+1) = pos(:,1))
rpos_set = [L;0;pi/3]*ones(1,6);  % 3 x 4 matrix
pos_set = zeros(3,size(rpos_set,2));
for k = 2:size(rpos_set,2)
    pos_set(:,k) = comp_add(pos_set(:,k-1), rpos_set(:,k-1));
end

errstd_x = 1; errstd_y = 1; errstd_a = 0.1;
% Simulate measurements with errors
rpos_msr = rpos_set + repmat([errstd_x;-errstd_y;-errstd_a],1,size(pos_set,2));

pos_msr_o = pos_set(:,1)*ones(1,size(rpos_set,2)+1);
for k = 1:size(rpos_set,2)
    pos_msr_o(:,k+1) = comp_add(pos_msr_o(:,k), rpos_msr(:,k));
end



% Loop closure optimization:
% Find the closed-loop poses (satisfying pose(N+2) = pose(1)) that fits the relative pose measurements in ML sense
% In this example, pos(:,1) is always [0;0;0], and we only need to find the optimal pos(:,2), pos(:,3), pos(:,4)
pos = pos_msr_o(:,1:(end-1));

    % Establish an objective function in terms of closed-loop poses, which characterizes the overall relative pose error
    % Compute the overall relative pose error from the closed-loop poses
    function rp_err = FErrorRPos(pos)
        rpos = zeros(3*size(pos,2),1);
        for k2 = 1:(size(pos,2)-1)
            rpos((3*k2-2):(3*k2),1) = comp_sub(pos(:,k2+1),pos(:,k2));
        end
        rpos((end-2):end,1) = comp_sub(pos(:,1),pos(:,end));
        rp_err = rpos - reshape(rpos_msr,[],1);
        % Standardize the error vector
        rp_err = rp_err./repmat([errstd_x;errstd_y;errstd_a],size(rpos_set,2),1);
    end

    % Compute the matrix of partial derivatives with respect to pos(:,2:end)
    dt = 0.0001;
    EMat = eye(3*(size(pos,2)-1))*dt;

    function difF = DifFErrorRPos(pos)
        difF = zeros(3*size(pos,2),size(EMat,1));
        for k2 = 1:size(EMat,1)
            dPos = [[0;0;0], reshape(EMat(:,k2),3,[])];

            difF(:,k2) = (FErrorRPos(pos+dPos)-FErrorRPos(pos-dPos))/(2*dt);
        end
    end

err_sum_old = sum(FErrorRPos(pos).^2);
fprintf('Pose initialization : %s \n  std_error_sum = %f \n', mat2str(pos), err_sum_old);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Practice: Complete the iteration code to find the optimal  pos  in maximum likelihood sense
% In this example, pos(:,1) is always [0;0;0], and we only need to find the optimal pos(:,2), pos(:,3), pos(:,4) ...
disp(size(pos));
for iter = 1:10
  grad = DifFErrorRPos(pos);
  step = inv(transpose(grad) * grad) * transpose(grad) * FErrorRPos(pos);
  size(reshape(step, 3, 5));
  pos(:,2:end) = pos(:,2:end) - reshape(step, 3, 5);
  sum(FErrorRPos(pos).^2)
end

figure(1); set(1, 'Position', [100,100,500,500]); hold on;
plot(pos_msr_o(1,:), pos_msr_o(2,:), '+r', 'MarkerSize', 12, 'LineWidth', 2); plot(pos_msr_o(1,:), pos_msr_o(2,:), 'r');
plot(pos(1,:), pos(2,:), '+b', 'MarkerSize', 12, 'LineWidth', 2); plot(pos(1,:), pos(2,:), 'b');
hold off; xlim([-20,50]); ylim([-10,70]); axis equal; grid on;

disp(pos);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
