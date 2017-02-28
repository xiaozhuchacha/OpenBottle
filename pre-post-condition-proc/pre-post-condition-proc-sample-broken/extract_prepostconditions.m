% skip timestamps columns
cur_col = 4;
% extract preconditions
pose_precondition = windows(:, cur_col:(cur_col+pose_window_width-1));
cur_col = cur_col + pose_window_width;
force_precondition = windows(:, cur_col:(cur_col+force_window_width-1));
cur_col = cur_col + force_window_width;
% extract postconditions
pose_postcondition = windows(:, cur_col:(cur_col+pose_window_width-1));
cur_col = cur_col + pose_window_width;
force_postcondition = windows(:, cur_col:(cur_col+force_window_width-1));
cur_col = cur_col + force_window_width;
% total pre and post conditions
precondition = horzcat(pose_precondition, force_precondition);
postcondition = horzcat(pose_postcondition, force_postcondition);