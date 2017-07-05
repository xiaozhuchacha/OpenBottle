function [dist] = dist_fluent(x, y, z)
  dist = sqrt(x.^2 + y.^2 + z.^2);
end