
% the following is unnecessary, but is an example of computing the offset of 
% pose data from the tf_ordering of the poses, avoiding the use of magic numbers
function [col_idx] = tf_offset_lookup_new(frame_id, child_frame_id, tf_ordering)
  col_idx = -1; % not found
  for idx = 1:2:size(tf_ordering, 2)
    if strcmp(tf_ordering{1,idx}, frame_id) && strcmp(tf_ordering{1,idx+1}, child_frame_id)
      col_idx = idivide(idx, int32(2), 'floor') + 1;
      return;
    end
  end
end
