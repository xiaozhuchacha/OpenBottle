clear
close all;
clc;

% index = {[8, 12, 13], [3, 4], [9]};
index = {4, 9, 12};
mode = {'vec'};

% loop over index group
for i_bag_group = 1:length(index)
    % loop over force and point mode
    for i_mode = 1:length(mode)
        fprintf('loading %s ', mode{i_mode});
        for i_bag = 1:length(index{i_bag_group})
            fprintf('%d ', index{i_bag_group}(i_bag));
        end
        fprintf('\n');
        
        % feature vector
        feature_all = [];

        % merge feature vectors
        for i_bag = 1:length(index{i_bag_group})
            feature_file = dir(sprintf('%d_*_%s.mat', index{i_bag_group}(i_bag), mode{i_mode}));
            features_mat = load(sprintf('%s', feature_file.name));

            feature_all = [feature_all; features_mat.force_wrist_vec];
        end
        
        % linkage and cluster
        % thumb index middle ring little palm fingers
        feature_group_id = {1:6, 7:12, 13:18, 19:24, 25:30, 31:78, 1:30, 1:78};
        fprintf('force and point mode\n');
        for i_feature_group = 1:length(feature_group_id)
            feature = feature_all(:, feature_group_id{i_feature_group});

            % linkage
            linkage_filename = sprintf('Z_%s_%d_%d.mat', mode{i_mode}, i_bag_group, i_feature_group);
            if exist(linkage_filename, 'file')
                load(linkage_filename);
                fprintf('force: linkage.%s computed before. loading...\n', linkage_filename);
            else
                fprintf('force: linkage. this will take a while. take a cup of coffee...\n');
                try
                    Z = linkage(single(feature), 'ward', 'euclidean');
                catch
                    fprintf('matrix is too large. savememory is on. will take longer.\n');
                    Z = linkage(single(feature), 'ward', 'euclidean', 'savememory', 'on');
                end
                save(linkage_filename, 'Z');
                fprintf('results saved in %s\n', linkage_filename);
            end

            % cluster
            fprintf('force: cluster.\n');
            for i_cluster = 2:10
                cluster_filename = sprintf('C_%s_%d_%d_%d.mat', mode{i_mode}, i_bag_group, i_feature_group, i_cluster);
                if exist(cluster_filename, 'file')
                    continue;
                else
                    cluster_result = cluster(Z, 'maxclust',i_cluster);
                    save(cluster_filename, 'cluster_result');
                end
            end
        end
    end
end