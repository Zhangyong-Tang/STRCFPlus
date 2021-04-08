
% This demo script runs the STRCF tracker with deep features on the
% included "Human3" video.

% Add paths
setup_paths();

%  Load video information
base_path  =  'E:/dataset/GTOT';
video  = choose_video(base_path);
% video = 'Human3';

video_path = [base_path '/' video];
[seq] = load_video_info(video_path,video);

% Run STRCF
results = run_STRCFplus(seq);
% results = run_DeepSTRCF(seq);
% results=ECCV2018settings(seq)
pd_boxes = results.res;
thresholdSetOverlap = 0: 0.05 : 1;
success_num_overlap = zeros(1, numel(thresholdSetOverlap));
res = calcRectInt(gt_boxes, pd_boxes);
for t = 1: length(thresholdSetOverlap)
    success_num_overlap(1, t) = sum(res > thresholdSetOverlap(t));
end
cur_AUC = mean(success_num_overlap) / size(gt_boxes, 1);
FPS_vid = results.fps;
display([video  '---->' '   FPS:   ' num2str(FPS_vid)   '    op:   '   num2str(cur_AUC)]);