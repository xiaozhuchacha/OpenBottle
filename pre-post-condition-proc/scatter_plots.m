
cluster_num = 3;

window_width = pose_window_width + force_window_width;
precondition = windows(:, 4:(4+window_width-1));
postcondition = windows(:, (4+window_width):size(windows,2));

pre_model = fitgmdist(precondition, cluster_num, 'CovarianceType', 'full', 'Regularize', 1e-5);
post_model = fitgmdist(postcondition, cluster_num, 'CovarianceType', 'full', 'Regularize', 1e-5);

window_width = double(window_width);
subplot_idx = 1;
for dim1=1:window_width
  figure;
  hold on;
  for dim2=1:window_width
    x1 = precondition(:,dim1);
    x2 = precondition(:,dim2);
    subplot(4, 5, subplot_idx);
    hold on;
    scatter(x1, x2, 10, '.');
    for model_idx = 1:cluster_num
      model_sig = pre_model.Sigma(:,:,1);
      model_mu1 = pre_model.mu(model_idx, dim1);
      model_mu2 = pre_model.mu(model_idx, dim2);
      plot(model_mu1, model_mu2, 'r*');
      mu = [model_mu1, model_mu2];
      sig = [model_sig(dim1, dim1), model_sig(dim1, dim2); model_sig(dim2, dim1), model_sig(dim2, dim2)];
      ax1 = sig(1,1)^2;
      ax2 = sig(2,2)^2;
      theta = 1.57 - atan(sig(2,1)/sig(1,2));
      for i =1:3
        ellipse(ax1, ax2, theta, mu(1), mu(2));
        ax1 = i * ax1;
        ax2 = i * ax2;
      end
    end
    title(sprintf('d1: %i d2: %i', dim1, dim2), 'FontSize', 8);
    subplot_idx = subplot_idx + 1;
  end
  subplot_idx = 1;
  hold off;
end