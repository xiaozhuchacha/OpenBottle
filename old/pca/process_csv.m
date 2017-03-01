function [data, times] = process_csv(input_cell)
  vicon_wrist = 'vicon/wrist/wrist';
  glove_link = 'palm_link';
  num_rows = size(input_cell, 1);
  times = str2double(input_cell(:, 1:2));   % extract times from the CSV
  data = zeros(num_rows, 1); % need initial column for horzcat
  i = 4; % skip the seconds, nanoseconds, and image id (column 1-3)
  cont = true;
  while cont
    if ~isstrprop(input_cell{1,i}(1), 'digit')  % if first character is not a digit, we are at a tf
      % process a tf CSV output
      % skip any world -> anything frame except for the one going to vicon_wrist
      % (this removes all objects; we are left with a world->glove link and all glove related links)
      if (strcmp(input_cell{1,i}, vicon_wrist) || strcmp(input_cell{1,i}, strcat('/', vicon_wrist))) && (~strcmp(input_cell{1,i+1}, glove_link) || ~strcmp(input_cell{1,i+1}, strcat('/', glove_link)))
        i = i + 9; % move to the next tf (or force readings)
        continue
      else
        % concat this tf's contents
        data = horzcat(data, str2double(input_cell(:, i+2:i+8)));
        disp(sprintf('Added tf between %s and %s', input_cell{1,i}, input_cell{1,i+1}));
        i = i + 9; % move to the next tf (or force readings)
      end
    else
      % force output
      data = horzcat(data, str2double(input_cell(:, i:i+25)));
      cont = false;
      disp('Added force data');
    end
  end
  num_cols = size(data, 2);
  data = data(:, 2:num_cols); % remove leading column of 0's
end
