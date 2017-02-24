%
% PCA reduction for glove data
%
% Parameters
%   data - contains original data to run dimension reduction on
%   num_eigenvectors - number of eigenvectors to use
%   
% Results:
%   evectors - matrix of eigenvectors 
%   scores - score of each eigenvector of original data in eigenspace
%   evalues - eigenvalues
%   columnmin - each column's min (used for normalization)
%   columnmax - each column's max (used for normalization)
%   norm_mean - the normalized mean of the data
%

function [evectors, scores, evalues, column_min, column_max, norm_mean] = pca_redcution(data, num_eigenvectors)

  num_samples = size(data, 1);
  
  % normalize data. need column_min and column_max to unnormalize
  [norm_data, column_min, column_max] = normalize(data);

  % shift data to mean using mean pose
  norm_mean = mean(norm_data, 1);
  shifted_data = norm_data - norm_mean(ones(num_samples, 1), :);

  % run pca
  [evectors, scores, evalues] = pca(shifted_data);

  % only keep top num_eigenvectors
  scores = scores(:, 1:num_eigenvectors);

  % display the eigenvalues
  normalized_evalues = evalues / sum(evalues);
  figure, plot(cumsum(normalized_evalues));
  xlabel('No. of eigenvectors'), ylabel('Variance accounted for');
  ylim([0 1]), grid on;

  % plot the top two eigenvectors
  % proj = norm_data * evectors(:, 1:2);
  % plot(proj(1,:), proj(2,:), 'r.');

  % reconstruct
  % num_eigenvectors = 25;
  % project into eigen subspace
  % projection = shifted_data * evectors(:, 1:num_eigenvectors);   % same as top num_eigenvectors from scores from pca()
  % reconstruction sample using the projection; adding mean unshifts data
  % reconstruction = projection * evectors(:, 1:num_eigenvectors)' + norm_mean(ones(num_samples, 1), :);

  % unnormalize data
  % reconstructed_data = zeros(size(data));
  % for i =1:num_samples
    % reconstructed_data(i,:) = reconstruction(i,:) .* (column_max - column_min) + column_min;
  % end

  % compute reconstruction error
  % reconstructed_err = data - reconstructed_data;

end
