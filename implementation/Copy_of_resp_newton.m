function [disp_row, disp_col, sind, max_scale_response] = resp_newton(response1, response2, iterations, ky, kx, use_sz)
responsef1 = fft2(response1);
responsef2 = fft2(response2);
if numel(response2) ~= 1 && numel(response1) ~= 1
    sz = size(kx,1);
    rg = circshift(-floor((sz-1)/2):ceil((sz-1)/2), [0 -floor((sz-1)/2)]);
    cg = circshift(-floor((sz-1)/2):ceil((sz-1)/2), [0 -floor((sz-1)/2)]);
    [rs, cs] = ndgrid(rg,cg);
    d=1-exp(-1/sz*(rs.^2 + cs.^2));%高斯型矩阵
    weight = 0.1:0.01:0.99;
    L = zeros(numel(weight),1);
    i = 4;
    r1 = response1(:,:,i);
    r2 = response2(:,:,i);
    r3=[response1,response2];
    scores_all_blocks=r3;
    filter_ids=[1,1];
    use_for_scale_estimation=[1,1];
    nScales=7;
    output_sz_score=size(r1);
                candidates = [];
                neighbour_candidates = [];
                
                % Find the peaks for each filter
                %before fusion
                %去掉小于0.5max的
                for f_id = 1:numel(filter_ids)
                    if use_for_scale_estimation(f_id)
                        %min_filter_score=0.5
                        thresh_mask = scores_all_blocks(:,:,:,f_id) > max(reshape(scores_all_blocks(:,:,:,f_id),[],1))*0.5;
                        tmp_map = scores_all_blocks(:,:,:,f_id).*single(thresh_mask);
                        local_peaks = imregionalmax(tmp_map,6);%将连通区域内的最大值置为1
                        new_candidates = find(local_peaks);  %候选目标
                        
                        if ~isempty(new_candidates)
                            candidates = [candidates; new_candidates(:)];%将所有通道级联（2类特征）
                        end
                    end
                end
                
                
                % Find peaks for the sum
                scores_joint = zeros(output_sz_score(1),output_sz_score(2), nScales);
                %fusion with 0.5 
                for f_id = 1:numel(filter_ids)
                    %if use_for_scale_estimation(f_id)  prior_alpha_t=[0.5 0.5]
                    scores_joint = scores_joint + 0.5*scores_all_blocks(:,:,:,f_id);
                    %else
                    %   scores_joint = scores_joint + params.prior_alpha_t(f_id)*scores_all_blocks(:,:,1,f_id);
                    %end
                end
                 %after fusion
                thresh_mask = scores_joint > max(reshape(scores_joint,[],1))*0.5;
                tmp_map = scores_joint.*single(thresh_mask);
                local_peaks = imregionalmax(tmp_map,6);
                new_candidates = find(local_peaks);
                
                if ~isempty(new_candidates)
                    candidates = [candidates; new_candidates(:)];%全部级联
                end
                %余下皆为高响应
                scores_size = size(scores_joint);
                
                % Remove if too many candidates
                candidates = unique(candidates);
                candidates_prior_score = scores_joint(candidates);
                
                if numel(candidates_prior_score) > params.max_num_candidates
                    % Sort
                    [~, sorted_ids] = sort(candidates_prior_score, 'descend');
                    candidates = candidates(sorted_ids(1:params.max_num_candidates));
                end
                
                % Find candidates in the neighbourhood
                [candidate_r,candidate_c, candidate_s] = ind2sub(scores_size,candidates);%找出candidate的坐标
    
    
    
    
    
    
    
    
    
    
    
    
%     for j = 1:numel(weight)
%         r = weight(j)*r1+(1-weight(j))*r2;
%         [r_max,id] = max(r(:));
%         [max_id1,max_id2] = ind2sub(size(r),id);%[r,c]
%         r_r = (r_max - r)./circshift(r_max*d,[max_id1-1,max_id2-1]);%将最小值移到最大响应值处
%         ep = min(r_r(:));
%         L(j) = 0.1*(weight(j)^2+(1-weight(j))^2)-ep;
%     end
%     [~, jidx]=min(L(:));
    response = weight(jidx)*response1+(1-weight(jidx))*response2;
    responsef = weight(jidx)*responsef1+(1-weight(jidx))*responsef2;
    aa=weight(jidx);
elseif numel(response2) ~= 1
    response = response2;
    responsef = responsef2;
else
    response = response1;
    responsef = responsef1;
end
% response = response1+response2;
% responsef = responsef1 + responsef2;
% 
% figure(3);mesh(circshift(response(:,:,4),floor(0.5*[sz,sz])));
% pause();
[max_resp_row, max_row] = max(response, [], 1);
[init_max_response, max_col] = max(max_resp_row, [], 2);
max_row_perm = permute(max_row, [2 3 1]);
col = max_col(:)';
row = max_row_perm(sub2ind(size(max_row_perm), col, 1:size(response,3)));

trans_row = mod(row - 1 + floor((use_sz(1)-1)/2), use_sz(1)) - floor((use_sz(1)-1)/2);
trans_col = mod(col - 1 + floor((use_sz(2)-1)/2), use_sz(2)) - floor((use_sz(2)-1)/2);
init_pos_y = permute(2*pi * trans_row / use_sz(1), [1 3 2]);
init_pos_x = permute(2*pi * trans_col / use_sz(2), [1 3 2]);
max_pos_y = init_pos_y;
max_pos_x = init_pos_x;

% pre-compute complex exponential
exp_iky = exp(bsxfun(@times, 1i * ky, max_pos_y));
exp_ikx = exp(bsxfun(@times, 1i * kx, max_pos_x));

% gradient_step_size = gradient_step_size / prod(use_sz);

ky2 = ky.*ky;
kx2 = kx.*kx;

iter = 1;
while iter <= iterations
    % Compute gradient
    ky_exp_ky = bsxfun(@times, ky, exp_iky);
    kx_exp_kx = bsxfun(@times, kx, exp_ikx);
    y_resp = mtimesx(exp_iky, responsef, 'speed');
    resp_x = mtimesx(responsef, exp_ikx, 'speed');
    grad_y = -imag(mtimesx(ky_exp_ky, resp_x, 'speed'));
    grad_x = -imag(mtimesx(y_resp, kx_exp_kx, 'speed'));
    ival = 1i * mtimesx(exp_iky, resp_x, 'speed');
    H_yy = real(-mtimesx(bsxfun(@times, ky2, exp_iky), resp_x, 'speed') + ival);
    H_xx = real(-mtimesx(y_resp, bsxfun(@times, kx2, exp_ikx), 'speed') + ival);
    H_xy = real(-mtimesx(ky_exp_ky, mtimesx(responsef, kx_exp_kx, 'speed'), 'speed'));
    det_H = H_yy .* H_xx - H_xy .* H_xy;
    
    % Compute new position using newtons method
    max_pos_y = max_pos_y - (H_xx .* grad_y - H_xy .* grad_x) ./ det_H;
    max_pos_x = max_pos_x - (H_yy .* grad_x - H_xy .* grad_y) ./ det_H;
    
    % Evaluate maximum
    exp_iky = exp(bsxfun(@times, 1i * ky, max_pos_y));
    exp_ikx = exp(bsxfun(@times, 1i * kx, max_pos_x));
    
    iter = iter + 1;
end
max_response = 1 / prod(use_sz) * real(mtimesx(mtimesx(exp_iky, responsef, 'speed'), exp_ikx, 'speed'));

% check for scales that have not increased in score
ind = max_response < init_max_response;
max_response(ind) = init_max_response(ind);
max_pos_y(ind) = init_pos_y(ind);
max_pos_x(ind) = init_pos_x(ind);

[max_scale_response, sind] = max(max_response(:));
disp_row = (mod(max_pos_y(1,1,sind) + pi, 2*pi) - pi) / (2*pi) * use_sz(1);
disp_col = (mod(max_pos_x(1,1,sind) + pi, 2*pi) - pi) / (2*pi) * use_sz(2);
%    figure(3);
%    plot(L(:));
%   pause(0.1);
end