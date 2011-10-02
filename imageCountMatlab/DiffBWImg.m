function [res] = DiffBWImg(X0, X1)
%% returns 0 if images are equal

diff = X0 - X1;
res = max(diff(:));

end
