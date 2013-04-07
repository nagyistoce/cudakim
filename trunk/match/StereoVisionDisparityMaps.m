%% KBE, 14/2-2013
clear, close all;

%I1 = rgb2gray(imread('RocksL.png'));
%I2 = rgb2gray(imread('RocksR.png'));
I1 = rgb2gray(imread('BowlingL.png'));
I2 = rgb2gray(imread('BowlingR.png'));
imshowpair(I1, I2,'montage');
title('I1 (left); I2 (right)');

imwrite(I1, 'Bowling-l.pgm', 'pgm');
imwrite(I2, 'Bowling-r.pgm', 'pgm');

I1d = im2double(I1);
I2d = im2double(I2);

disp_CENSUS = im2double(imread('Disp_CENSUS.pgm'));
disp_SAD = im2double(imread('Disp_SAD.pgm'));

figure;
mesh(disp_SAD);
title('SAD Disparity map'); 
figure;
mesh(disp_CENSUS);
title('CENSUS Disparity map'); 

% Census transform 
%[disp scores] = CENSUS(I1d, I2d, 'l', 5, 11, [0 100]); 

% Sum of Absolute Differences
%[disp scores] = SAD(I1, I2, 'l', 11, [0 100]);
% Zero Mean Sum of Absolute Differences
%[disp scores] = ZSAD(I1, I2, 'l', 11, [0 100]);

% Sum of Squared Differences
%[disp scores] = SSD(I1, I2, 'l', 11, [0 100]);
% Zero Mean Sum of Squared Differences
%[disp scores] = ZSSD(I1, I2, 'l', 11, [0 100]);

% Normalised Cross Correlation
%[disp scores] = NCC(I1, I2, 'l', 11, [0 100]);
% Zero Mean Normalised Cross Correlation
%[disp scores] = ZNCC(I1, I2, 'l', 11, [0 100]);

[status, result] = system('match -m SSD Bowling Disparity');
status

disparity = im2double(imread('Disparity.pgm'));

figure;
mesh(disparity);
title('Disparity map'); 
