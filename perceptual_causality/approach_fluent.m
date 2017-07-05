
addpath(genpath('/home/mark/Dropbox/Developer/matlab'));

approach_cell_mat = csv2cell('/home/mark/Desktop/actions/approach.csv', 'fromfile');

[data, tf_ordering] = extract_data_from_cell(approach_cell_mat);

poses = data(:, 1:size(data,2) - 26);
forces = data(:, size(data,2) - 25:size(data,2));

wrist_bottle_tf_offset = tf_offset_lookup_new('vicon/wrist/wrist', 'vicon/bottle69/bottle69', tf_ordering);

wrist_bottle_prewindows = poses(1:2:size(poses,1),(wrist_bottle_tf_offset-1)*7+1:wrist_bottle_tf_offset*7)
wrist_bottle_postwindows = poses(2:2:size(poses,1),(wrist_bottle_tf_offset-1)*7+1:wrist_bottle_tf_offset*7)

x_pre = wrist_bottle_prewindows(:,1);
y_pre = wrist_bottle_prewindows(:,2);
z_pre = wrist_bottle_prewindows(:,3);
x_post = wrist_bottle_postwindows(:,1);
y_post = wrist_bottle_postwindows(:,2);
z_post = wrist_bottle_postwindows(:,3);

% compute the relative distance between the wrist and the bottle
rel_dist_pre = dist_fluent(x_pre, y_pre, z_pre);
rel_dist_post = dist_fluent(x_post, y_post, z_post);

% compute and plot histogram
edges = linspace(min(min(rel_dist_pre), min(rel_dist_post)), max(max(rel_dist_pre), max(rel_dist_post)), 101);
figure();
hold on;
h_pre = histogram(rel_dist_pre, edges);
h_post = histogram(rel_dist_post, edges);
pre_counts = h_pre.Values;
post_counts = h_post.Values;
hold off;

% compute distributions
eps_ = 0.0001
pre_pd = (pre_counts - min(pre_counts)) / (max(pre_counts) - min(pre_counts));
post_pd = (post_counts - min(post_counts)) / (max(post_counts) - min(post_counts));
pre_pd(find(pre_pd == 0)) = eps_;
post_pd(find(post_pd == 0)) = eps_;

% kl_div = KLDiv(pre_dist, post_dist);
kl_div = sum(pre_pd .* (log2(pre_pd) - log2(post_pd)));