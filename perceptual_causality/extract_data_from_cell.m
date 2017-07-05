function [data, tf_ordering] = extract_data_from_cell(cell_mat)
  base_offset = 4; % skip sec, nsec, and image id cols
  num_tfs = 0;
  while true
    if(all(ismember(cell_mat{1, base_offset}, '0123456789+-.eEdD')))
      break % hit the beginning of force values
    else
      num_tfs = num_tfs + 1;
    end
    base_offset = base_offset + 9;
  end

  % each tf is 7 numbers, 26 = number of force readings
  data = zeros(size(cell_mat, 1), (num_tfs * 7) + 26);
  tf_ordering = cell(size(cell_mat, 1), num_tfs*2); % first row = parent frame, second row = child frame
  base_offset = 4;
  
  for i=1:num_tfs
    tf_ordering(:,(i-1)*2 + 1) = cell_mat(:, base_offset);
    tf_ordering(:,(i-1)*2 + 2) = cell_mat(:, base_offset+1);
    base_offset = base_offset + 2;
    data_col_group = (((i-1) * 7) + 1:((i-1) * 7) + 7);
    cell_col_group = (base_offset:base_offset + 6);
    data(:, data_col_group) = cellfun(@str2double, cell_mat(:, cell_col_group));
    base_offset = base_offset + 7;
  end

  data(:, size(data,2)-26:size(data,2)) = cellfun(@str2double, cell_mat(:, (size(cell_mat, 2)-27):size(cell_mat, 2)-1));
end