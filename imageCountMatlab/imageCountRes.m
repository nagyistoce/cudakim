%% KBE, 2/10-2011
clear, close all;

% Import image
Im1 = imread('nordResult11.bmp');
Im2 = imread('nordResult12.bmp');
Im3 = imread('nordResult13.bmp');
Im4 = imread('nordResult14.bmp');
Im5 = imread('nordResult15.bmp');
Im6 = imread('nordResult16.bmp');
Im7 = imread('nordResult17.bmp');
Im8 = imread('nordResult18.bmp');
Im9 = imread('nordResult19.bmp');

% Show the images
figure, imshow([Im1, Im2, Im3;Im4, Im5, Im6;Im7, Im8, Im9]); title('9 frames labeled with colored connected components');

