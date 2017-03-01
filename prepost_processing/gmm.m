
cluster_num = 10;

window_width = pose_window_width + force_window_width;
precondition = windows(:, 4:(4+window_width-1));
postcondition = windows(:, (4+window_width):size(windows,2));

pre_model = fitgmdist(precondition, cluster_num, 'CovarianceType', 'diagonal', 'Regularize', 1e-5);
post_model = fitgmdist(postcondition, cluster_num, 'CovarianceType', 'diagonal', 'Regularize', 1e-5);

% plot pre and post conditions
subplot_idx = 1;
figure;
hold on;
window_width = double(window_width);
for dim=1:window_width
  x = precondition(:,dim);
  x_plot = linspace(min(x),max(x));
  for model_idx=1:cluster_num
    sig = pre_model.Sigma(1, dim, model_idx);
    mu = pre_model.mu(model_idx, dim);
    subplot(5, 8, subplot_idx);
    hold on;
    model_pdf = normpdf(x_plot, mu, sig);
    model_pdf = model_pdf / sum(model_pdf);
    plot(x_plot, model_pdf);
    title(sprintf('pre %i', dim), 'FontSize', 8);
  end
  subplot_idx = subplot_idx + 1;

  x = postcondition(:,dim);
  x_plot = linspace(min(x),max(x));
  for model_idx=1:cluster_num
    sig = post_model.Sigma(1, dim, model_idx);
    mu = post_model.mu(model_idx, dim);
    subplot(5, 8, subplot_idx);
    hold on;
    model_pdf = normpdf(x_plot, mu, sig);
    model_pdf = model_pdf / sum(model_pdf);
    plot(x_plot, model_pdf);
    title(sprintf('post %i', dim), 'FontSize', 8);
  end
  subplot_idx = subplot_idx + 1;
hold off;
end

% [Priors, Mu, Sigma] = EM_init_kmeans(precondition, cluster_num);
% [Priors, Mu, Sigma] = EM(precondition, Priors, Mu, Sigma);
    
% post_gmm = fitgmdist(postcondition,5);