close all; clear all; clc;

data_dir = '~/Dropbox/Documents/VCLA/SIMPLEX/datacollection/11_29_data_local/proc/3_bottle69_open_bottle_palm_2_tf_convert_merged_successes_proc/';
data_file = 'hand_only_with_tf_labels_data';

data = load(strcat(strcat(data_dir, data_file), '.mat'));
data = data.data;

num_samples = size(data, 1);

% setup column-wise normalization
column_min = min(data, [], 1);
column_max = max(data, [], 1);
% if max == min, set max to 1 and min = 0 to prevent dividing by zero
for i = 1:size(column_min, 2)
  if column_min(i) == column_max(i)
    column_min(i) = 0;
    column_max(i) = 1;
  end
end

% normalize data between [0,1]
norm_data = zeros(size(data));
for i =1:num_samples
  norm_data(i,:) = (data(i,:) - column_min) ./ (column_max - column_min);
end

% shift data to mean using mean pose
norm_mean_pose = mean(norm_data, 1);
shifted_data = norm_data - norm_mean_pose(ones(num_samples, 1), :);

% run pca
[evectors, score, evalues] = pca(shifted_data);

% display the eigenvalues
normalized_evalues = evalues / sum(evalues);
figure, plot(cumsum(normalized_evalues));
xlabel('No. of eigenvectors'), ylabel('Variance accounted for');
ylim([0 1]), grid on;

% plot the top two eigenvectors
% proj = norm_data * evectors(:, 1:2);
% plot(proj(1,:), proj(2,:), 'r.');

% reconstruct
num_eigenvectors = 25;
% project into eigen subspace
projection = shifted_data * evectors(:, 1:num_eigenvectors); 
% reconstruction sample using the projection; adding mean unshifts data
reconstruction = projection * evectors(:, 1:num_eigenvectors)' + norm_mean_pose(ones(num_samples, 1), :);

% unnormalize data
reconstructed_data = zeros(size(data));
for i =1:num_samples
  reconstructed_data(i,:) = reconstruction(i,:) .* (column_max - column_min) + column_min;
end

% compute reconstruction error
reconstructed_err = data - reconstructed_data;


% save reconstructed data to mat file
reconstruction_file = strcat(strcat(data_dir, data_file), '_reconstructed.mat')
save(reconstruction_file, 'reconstructed_data');