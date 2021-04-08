function results = tracker(params)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get sequence info
[seq, im] = get_sequence_info(params.seq);
params = rmfield(params, 'seq');
if isempty(im)
    seq.rect_position = [];
    [~, results] = get_sequence_results(seq);
    return;
end

% Init position
pos = seq.init_pos(:)';
target_sz = seq.init_sz(:)';
params.init_sz = target_sz;

% Feature settings
features = params.t_features;

% Set default parameters
params = init_default_params(params);

% Global feature parameters
if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end

global_fparams.use_gpu = params.use_gpu;
global_fparams.gpu_id = params.gpu_id;

% Define data types
if params.use_gpu
    params.data_type = zeros(1, 'single', 'gpuArray');
else
    params.data_type = zeros(1, 'single');
end
params.data_type_complex = complex(params.data_type);

global_fparams.data_type = params.data_type;

% Load learning parameters
admm_max_iterations = params.max_iterations;
init_penalty_factor = params.init_penalty_factor;
max_penalty_factor = params.max_penalty_factor;
penalty_scale_step = params.penalty_scale_step;
temporal_regularization_factor = params.temporal_regularization_factor; 

init_target_sz = target_sz;

% Check if color image
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end

if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end

% Check if mexResize is available and show warning otherwise.
params.use_mexResize = true;
global_fparams.use_mexResize = true;
try
    [~] = mexResize(ones(5,5,3,'uint8'), [3 3], 'auto');
catch err
    params.use_mexResize = false;
    global_fparams.use_mexResize = false;
end

% Calculate search area and initial scale factor
search_area = prod(params.search_area_scale'*init_target_sz,2);
currentScaleFactor = zeros(numel(search_area),1);
for i = 1 : numel(currentScaleFactor)
    if search_area(i) > params.max_image_sample_size(i)
        currentScaleFactor(i) = sqrt(search_area(i) / params.max_image_sample_size(i));
    elseif search_area(i) < params.min_image_sample_size(i)
        currentScaleFactor(i) = sqrt(search_area(i) / params.min_image_sample_size(i));
    else
        currentScaleFactor(i) = 1.0;
    end
end

% target size at the initial scale
base_target_sz = 1 ./ currentScaleFactor*target_sz;

% window size, taking padding into account
img_sample_sz = repmat(sqrt(prod(base_target_sz,2) .* (params.search_area_scale.^2')),1,2); % square area, ignores the target aspect ratio


[features, global_fparams, feature_info] = init_features(features, global_fparams, is_color_image, img_sample_sz, 'odd_cells');


% Set feature info
img_support_sz = feature_info.img_support_sz;
feature_sz = feature_info.data_sz;
num_feature_blocks = size(feature_sz, 1);


% Size of the extracted feature maps
feature_sz_cell = permute(mat2cell(feature_sz, ones(1,num_feature_blocks), 2), [2 3 1]);
filter_sz = feature_sz + mod(feature_sz+1, 2);
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

% The size of the label function DFT. Equal to the maximum filter size
[output_sz, k1] = max(filter_sz, [], 1);
params.output_sz = output_sz;
k1 = k1(1);

% Get the remaining block indices
block_inds = 1:num_feature_blocks;
block_inds(k1) = [];

% Construct the Gaussian label function
yf = cell(numel(num_feature_blocks), 1);
for i = 1:num_feature_blocks
    sz = filter_sz_cell{i};
    output_sigma_factor = params.output_sigma_factor(feature_info.feature_is_deep(i)+1);
    output_sigma  = sqrt(prod(floor(base_target_sz(feature_info.feature_is_deep(i)+1,:))))*feature_sz_cell{i}./img_support_sz{i}* output_sigma_factor;
    rg            = circshift(-floor((sz(1)-1)/2):ceil((sz(1)-1)/2), [0 -floor((sz(1)-1)/2)]);
    cg            = circshift(-floor((sz(2)-1)/2):ceil((sz(2)-1)/2), [0 -floor((sz(2)-1)/2)]);
    [rs, cs]      = ndgrid(rg,cg);
    y             = exp(-0.5 * (((rs.^2 + cs.^2) / mean(output_sigma)^2)));
    yf{i}         = fft2(y);
end

% Compute the cosine windows
cos_window = cellfun(@(sz) hann(sz(1)+2)*hann(sz(2)+2)', feature_sz_cell, 'uniformoutput', false);
cos_window = cellfun(@(cos_window) cast(cos_window(2:end-1,2:end-1), 'like', params.data_type), cos_window, 'uniformoutput', false);

% Define spatial regularization windows
reg_window = cell(num_feature_blocks, 1);
for i = 1:num_feature_blocks
    reg_scale = floor(base_target_sz(feature_info.feature_is_deep(i)+1,:) /params.feature_downsample_ratio(feature_info.feature_is_deep(i)+1));
    use_sz = filter_sz_cell{i};    
    reg_window{i} = ones(use_sz) * params.reg_window_max;
    range = zeros(numel(reg_scale), 2);
    
    % determine the target center and range in the regularization windows
    for j = 1:numel(reg_scale)
        range(j,:) = [0, reg_scale(j) - 1] - floor(reg_scale(j) / 2);
    end
    center = floor((use_sz + 1)/ 2) + mod(use_sz + 1,2);
    range_h = (center(1)+ range(1,1)) : (center(1) + range(1,2));
    range_w = (center(2)+ range(2,1)) : (center(2) + range(2,2));
    
    reg_window{i}(range_h, range_w) = params.reg_window_min;
end

% Pre-computes the grid that is used for socre optimization
ky = circshift(-floor((filter_sz_cell{1}(1) - 1)/2) : ceil((filter_sz_cell{1}(1) - 1)/2), [1, -floor((filter_sz_cell{1}(1) - 1)/2)]);
kx = circshift(-floor((filter_sz_cell{1}(2) - 1)/2) : ceil((filter_sz_cell{1}(2) - 1)/2), [1, -floor((filter_sz_cell{1}(2) - 1)/2)])';
newton_iterations = params.newton_iterations;

% Use the translation filter to estimate the scale
nScales = params.number_of_scales;
scale_step = params.scale_step;
scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
scaleFactors = scale_step .^ scale_exp;


seq.time = 0;

% Define the learning variables
f_pre_f = cell(num_feature_blocks, 1);
cf_f = cell(num_feature_blocks, 1);

% Allocate
det_sample_pos = pos;
scores_fs_feat = cell(1,1,num_feature_blocks);
response_feat = cell(1,1,num_feature_blocks);
a=zeros(100,1);
while true
    % Read image
    if seq.frame > 0
        [seq, im] = get_sequence_frame(seq);
        if isempty(im)
            break;
        end
        if size(im,3) > 1 && is_color_image == false
            im = im(:,:,1);
        end
    else
        seq.frame = 1;
    end

    tic();
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Target localization step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Do not estimate translation and scaling on the first frame, since we 
    % just want to initialize the tracker there
    if seq.frame > 1
        global_fparams.augment = 0;
        old_pos = inf(size(pos));
        iter = 1;
        seq.frame
        %translation search
        while iter <= params.refinement_iterations && any(old_pos ~= pos)
            % Extract features at multiple resolutions
            sample_pos = round(pos);
            det_sample_pos = pos;
            sample_scale = currentScaleFactor*scaleFactors;
            xt = extract_features(im, sample_pos, sample_scale, features, global_fparams, feature_info);
                                    
            % Do windowing of features
            xtw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt, cos_window, 'uniformoutput', false);
            
            % Compute the fourier series
            xtf = cellfun(@fft2, xtw, 'uniformoutput', false);
                        
            % Compute convolution for each feature block in the Fourier domain
            % and the sum over all blocks.
            response_handcrafted = 0;
            response_deep = 0;
            for k = [k1 block_inds]
                if feature_info.feature_is_deep(k) == 0
                    scores_fs_feat{k} = gather(sum(bsxfun(@times, conj(cf_f{k}), xtf{k}), 3));
                    scores_fs_feat{k} = resizeDFT2(scores_fs_feat{k}, output_sz);
                    response_feat{k} = ifft2(scores_fs_feat{k}, 'symmetric');  
                    response_handcrafted = response_handcrafted + response_feat{k};
                else
                    output_sz_deep = round(output_sz/img_support_sz{k1}*img_support_sz{k});
                    output_sz_deep = output_sz_deep + 1 + mod(output_sz_deep,2);
                    scores_fs_feat{k} = gather(sum(bsxfun(@times, conj(cf_f{k}), xtf{k}), 3));
                    scores_fs_feat{k} = resizeDFT2(scores_fs_feat{k}, output_sz_deep);
                    response_feat{k} = ifft2(scores_fs_feat{k}, 'symmetric');
                    response_feat{k}(ceil(output_sz(1)/2)+1:output_sz_deep(1)-floor(output_sz(1)/2),:,:,:)=[];
                    response_feat{k}(:,ceil(output_sz(2)/2)+1:output_sz_deep(2)-floor(output_sz(2)/2),:,:)=[];
                    response_deep = response_deep + response_feat{k};
                end
            end
             
            [disp_row, disp_col, sind, ~,aa] = resp_newton(squeeze(response_handcrafted)/feature_info.feature_hc_num, squeeze(response_deep)/feature_info.feature_deep_num,...
                newton_iterations, ky, kx, output_sz);
            a(seq.frame)=aa;
%             a(seq.frame)=aa;
%             if seq.frame==584
%                 display(a);
%             end
            % Compute the translation vector in pixel-coordinates and round
            % to the closest integer pixel.
            translation_vec = [disp_row, disp_col] .* (img_support_sz{k1}./output_sz) * currentScaleFactor(1) * scaleFactors(sind); 
            if seq.frame < 10
                scale_change_factor = scaleFactors(ceil(params.number_of_scales/2));
            else
                scale_change_factor = scaleFactors(sind);
            end

            
            % update position
            old_pos = pos;
            pos = sample_pos + translation_vec;
            
            if params.clamp_position
                pos = max([1 1], min([size(im,1) size(im,2)], pos));
            end
                        
            % Update the scale
            currentScaleFactor = currentScaleFactor * scale_change_factor;
            
            % Adjust to make sure we are not to large or to small
            
            iter = iter + 1;
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Model update step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % extract image region for training sample
    sample_pos = round(pos);
    global_fparams.augment = 1;
    xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_info);

    % do windowing of features
    xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);

    % compute the fourier series
    xlf = cellfun(@fft2, xlw, 'uniformoutput', false);
    
    % train the CF model for each feature
    for k = 1: numel(xlf)
        model_xf = xlf{k};

        if (seq.frame == 1)
            f_pre_f{k} = zeros(size(model_xf));
            mu = 0;
        else
            mu = temporal_regularization_factor(feature_info.feature_is_deep(k)+1);
        end
        
        % intialize the variables
        f_f = single(zeros(size(model_xf)));
        g_f = f_f;
        h_f = f_f;
        gamma  = init_penalty_factor(feature_info.feature_is_deep(k)+1);
        gamma_max = max_penalty_factor(feature_info.feature_is_deep(k)+1);
        gamma_scale_step = penalty_scale_step(feature_info.feature_is_deep(k)+1);
        
        % use the GPU mode
        if params.use_gpu
            model_xf = gpuArray(model_xf);
            f_f = gpuArray(f_f);
            f_pre_f{k} = gpuArray(f_pre_f{k});
            g_f = gpuArray(g_f);
            h_f = gpuArray(h_f);
            reg_window{k} = gpuArray(reg_window{k});
            yf{k} = gpuArray(yf{k});
        end

        % pre-compute the variables
        T = prod(output_sz);
        S_xx = sum(conj(model_xf) .* model_xf, 3);
        Sf_pre_f = sum(conj(model_xf) .* f_pre_f{k}, 3);
        Sfx_pre_f = bsxfun(@times, model_xf, Sf_pre_f);

        % solve via ADMM algorithm
        iter = 1;
        while (iter <= admm_max_iterations)

            % subproblem f
            B = S_xx + T * (gamma + mu);
            Sgx_f = sum(conj(model_xf) .* g_f, 3);
            Shx_f = sum(conj(model_xf) .* h_f, 3);
 
            f_f = ((1/(T*(gamma + mu)) * bsxfun(@times,  yf{k}, model_xf)) - ((1/(gamma + mu)) * h_f) +(gamma/(gamma + mu)) * g_f) + (mu/(gamma + mu)) * f_pre_f{k} - ...
                bsxfun(@rdivide,(1/(T*(gamma + mu)) * bsxfun(@times, model_xf, (S_xx .*  yf{k})) + (mu/(gamma + mu)) * Sfx_pre_f - ...
                (1/(gamma + mu))* (bsxfun(@times, model_xf, Shx_f)) +(gamma/(gamma + mu))* (bsxfun(@times, model_xf, Sgx_f))), B);

            %   subproblem g
            g_f = fft2(argmin_g(reg_window{k}, gamma, real(ifft2(gamma * f_f+ h_f)), g_f));

            %   update h
            h_f = h_f + (gamma * (f_f - g_f));

            %   update gamma
            gamma = min(gamma_scale_step * gamma, gamma_max);
            
            iter = iter+1;
        end
        f_pre_f{k} = f_f;
        if seq.frame == 1,
            % save the trained filters
            cf_f{k} = f_f;
        else 
            cf_f{k} = feature_info.learning_rate(k) * f_f + (1 - feature_info.learning_rate(k)) * cf_f{k};
        end
    end  
            
    % Update the target size (only used for computing output box)
    target_sz = base_target_sz(1,:) * currentScaleFactor(1);
    
    %save position and calculate FPS
    tracking_result.center_pos = double(pos);
    tracking_result.target_size = double(target_sz);
    seq = report_tracking_result(seq, tracking_result);
    
    seq.time = seq.time + toc();
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Visualization
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % visualization
    if params.visualization
        rect_position_vis = [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        if seq.frame == 1,  %first frame, create GUI
            fig_handle = figure('Name', 'Tracking');
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(seq.frame), 'color', [0 1 1]);
            hold off;
            axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
            

        else
            % Do visualization of the sampled confidence scores overlayed
            resp_sz = round(img_support_sz{1}*currentScaleFactor(2)*scaleFactors(sind));
            xs = floor(det_sample_pos(2)) + (1:resp_sz(2)) - floor(resp_sz(2)/2);
            ys = floor(det_sample_pos(1)) + (1:resp_sz(1)) - floor(resp_sz(1)/2);

            sampled_scores_display = circshift(imresize(response_deep(:,:,sind),...
                [numel(xs),numel(ys)]),round(0.5*([size(xs,2),size(ys,2)])+translation_vec));

            figure(fig_handle);
            imagesc(im_to_show);
            hold on;
            resp_handle = imagesc(xs, ys, sampled_scores_display); colormap hsv;
            alpha(resp_handle, 0.4);
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(seq.frame), 'color', [0 1 1]);
            hold off;
%             figure(fig_handle1);

        end

                    
        drawnow
    end
end

[~, results] = get_sequence_results(seq);

disp(['fps: ' num2str(results.fps)])

