%Find cars in picture
function [L, num]=FindCars(dF)

disp(['Morph']);
tic
%SE = [ 0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0];
SE = [ 0 1 0; 1 1 1; 0 1 0];
BW1 = imerode(dF, SE);

%Dilate to find cars
%SE = strel('disk',6,6);
%SE = [ 0 1 1 1 1 0; 0 1 1 1 1 0; 1 1 1 1 1 1; 1 1 1 1 1 1; 0 1 1 1 1 0; 0 1 1 1 1 0];
SE = [ 0 1 1 1 0; 0 1 1 1 0; 1 1 1 1 1; 0 1 1 1 0; 0 1 1 1 0];
BW2 = imdilate(BW1, SE);
toc
%figure, imshow(BW2)

% Label the regions - morfologisk genkendenelse - segmentering
disp(['Label']);
tic
[L,num] = bwlabel(BW2, 8); 
toc

