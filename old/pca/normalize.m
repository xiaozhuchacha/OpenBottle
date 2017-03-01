function [ norm_data, column_min, column_max] = normalize(data)

  num_samples = size(data, 1);

  column_min = min(data, [], 1);
  column_max = max(data, [], 1);
  % if max == min, set max to 1 and min = 0 to prevent dividing by zero
  for i = 1:size(column_min, 2)
    if column_min(i) == column_max(i)
      column_min(i) = 0;
      % if both are zero (i.e. column is zero), set max = 1 to prevent dividing by zero
      if column_max(i) == 0
        column_max(i) = 1;
      end
    end
  end

  % normalize data between [0,1]
  norm_data = zeros(size(data));
  for i =1:num_samples
    norm_data(i,:) = (data(i,:) - column_min) ./ (column_max - column_min);
  end

end

