%% KBE, 2/10-2011
clear, close all;

%% Image Count, Processing, Segmentation and classification
%Original from Lab 12, DIP1
%Opgave 1
%Motorvejen uden for Århus overvåges af et webcamera. Kameraets adresse er:
%http://www.trafikken.dk/wimpdoc.asp?page=document&objno=76653
%Den 24. nov. 2003 grappede jeg 9 billeder fra dette kamera. Disse ligger på nettet under kursets
%hjemmeside. Lav et MatLab-program som tæller antallet af biler på vejen (sådan ca.).
%(Hint: Du kan f.eks. bruge følgende procedure:
%1. Indlæs billederne og omdan dem til gråtoner.
%2. Find baggrunden ved at tage medianen af billderne (samme pixel-position men til
%forskellige tidspunkter).
%3. Bestem nu forgrunde.
%4. Lav billederne om til sort/hvid og, bestem antallet af sammenhængskomponenter, lav evt. en
%dilation for ikke at få for mange komponenter.
%5. Tæl nu antallet af biler (ca.)

% Import image
Im1 = imread('E45nord1.jpg');
Im2 = imread('E45nord2.jpg');
Im3 = imread('E45nord3.jpg');
Im4 = imread('E45nord4.jpg');
Im5 = imread('E45nord5.jpg');
Im6 = imread('E45nord6.jpg');
Im7 = imread('E45nord7.jpg');
Im8 = imread('E45nord8.jpg');
Im9 = imread('E45nord9.jpg');

%tic
N = 9; % Number of pictures
Header = 20; % Hight of header of picture in pixels

%1. Indlæs billederne og omdan dem til gråtoner.
I = double(rgb2gray(Im1))/255; %Convert image to gray scale
I(1:Header, :) = 0; % Zero header area
dI1 = I; % Store pictures in array

I = double(rgb2gray(Im1))/255; %Convert image to gray scale
I(1:Header, :) = 0; % Zero header area
dI1 = I; % Store pictures in array

I = double(rgb2gray(Im2))/255; %Convert image to gray scale
I(1:Header, :) = 0; % Zero header area
dI2 = I; % Store pictures in array

I = double(rgb2gray(Im3))/255; %Convert image to gray scale
I(1:Header, :) = 0; % Zero header area
dI3 = I; % Store pictures in array

I = double(rgb2gray(Im4))/255; %Convert image to gray scale
I(1:Header, :) = 0; % Zero header area
dI4 = I; % Store pictures in array

I = double(rgb2gray(Im5))/255; %Convert image to gray scale
I(1:Header, :) = 0; % Zero header area
dI5 = I; % Store pictures in array

I = double(rgb2gray(Im6))/255; %Convert image to gray scale
I(1:Header, :) = 0; % Zero header area
dI6 = I; % Store pictures in array

I = double(rgb2gray(Im7))/255; %Convert image to gray scale
I(1:Header, :) = 0; % Zero header area
dI7 = I; % Store pictures in array

I = double(rgb2gray(Im8))/255; %Convert image to gray scale
I(1:Header, :) = 0; % Zero header area
dI8 = I; % Store pictures in array

I = double(rgb2gray(Im9))/255; %Convert image to gray scale
I(1:Header, :) = 0; % Zero header area
dI9 = I; % Store pictures in array

% Show the images
figure, imshow([dI1, dI2, dI3;dI4, dI5, dI6;dI7, dI8, dI9]); title('9 frames in grayscale');

%2. Find baggrunden ved at tage medianen af billderne (samme pixel-position men til
%forskellige tidspunkter).

disp(['Background']);
tic
for s=1:size(dI1,1)
    for r=1:size(dI1,2)
        v= [dI1(s,r), dI2(s,r), dI3(s,r), dI4(s,r), dI5(s,r), dI6(s,r), dI7(s,r), dI8(s,r), dI9(s,r)];
        dB(s,r) = median(v);
    end
end
toc
figure, imshow(dB) % Background

M = 3;
h = ones(M,M)./(M*M);
imgBlur = imfilter(abs(dI1-dB), h);
figure, imshow(abs(dI1-dB));
figure, imshow(imgBlur);

%3. Bestem nu forgrunde using adaptive threshold filtering
S = size(dI1);
% Using blur filter
% dF1 = blkproc(imfilter(abs(dI1-dB), h), S, @AdapThreshold);
% dF2 = blkproc(imfilter(abs(dI2-dB), h), S, @AdapThreshold);
% dF3 = blkproc(imfilter(abs(dI3-dB), h), S, @AdapThreshold);
% dF4 = blkproc(imfilter(abs(dI4-dB), h), S, @AdapThreshold);
% dF5 = blkproc(imfilter(abs(dI5-dB), h), S, @AdapThreshold);
% dF6 = blkproc(imfilter(abs(dI6-dB), h), S, @AdapThreshold);
% dF7 = blkproc(imfilter(abs(dI7-dB), h), S, @AdapThreshold);
% dF8 = blkproc(imfilter(abs(dI8-dB), h), S, @AdapThreshold);
% dF9 = blkproc(imfilter(abs(dI9-dB), h), S, @AdapThreshold);

dF1 = blkproc(abs(dI1-dB), S, @AdapThreshold);
dF2 = blkproc(abs(dI2-dB), S, @AdapThreshold);
dF3 = blkproc(abs(dI3-dB), S, @AdapThreshold);
dF4 = blkproc(abs(dI4-dB), S, @AdapThreshold);
dF5 = blkproc(abs(dI5-dB), S, @AdapThreshold);
dF6 = blkproc(abs(dI6-dB), S, @AdapThreshold);
dF7 = blkproc(abs(dI7-dB), S, @AdapThreshold);
dF8 = blkproc(abs(dI8-dB), S, @AdapThreshold);
dF9 = blkproc(abs(dI9-dB), S, @AdapThreshold);

figure, imshow([dF1, dF2, dF3;dF4, dF5, dF6;dF7, dF8, dF9]); title('BW "forgrund - cars" adaptive threshold');

%dF1 = abs(dI1-dB) > thresh;
%dF2 = abs(dI2-dB) > thresh;
%dF3 = abs(dI3-dB) > thresh;
%dF4 = abs(dI4-dB) > thresh;
%dF5 = abs(dI5-dB) > thresh;
%dF6 = abs(dI6-dB) > thresh;
%dF7 = abs(dI7-dB) > thresh;
%dF8 = abs(dI8-dB) > thresh;
%dF9 = abs(dI9-dB) > thresh;
%figure, imshow([dF1, dF2, dF3;dF4, dF5, dF6;dF7, dF8, dF9]); title('9 frames in grayscale');

%dF = dF1;
%Removes noise
%SE = [ 0 1 0; 1 1 1; 0 1 0];
%BW1 = imerode(dF, SE);

%Erode to find cars
%SE = strel('disk',6,6);
%BW2 = imdilate(BW1, SE);
%figure, imshow(BW2)

% Label the regions - morfologisk genkendenelse - segmentering
%[L,num] = bwlabel(BW3, 8);

num = zeros(N,1);
[L1,num(1)] = FindCars(dF1);
figure, imshow(L1+1,colormap(hsv(8))), title('Image 1');
disp(['Number of cars in image1:  ' num2str(num(1))]);
[L2,num(2)] = FindCars(dF2);
figure, imshow(L2+1,colormap(hsv(8))), title('Image 2');
[L3,num(3)] = FindCars(dF3);
figure, imshow(L3+1,colormap(hsv(8))), title('Image 3');
[L4,num(4)] = FindCars(dF4);
figure, imshow(L4+1,colormap(hsv(8))), title('Image 4');
[L5,num(5)] = FindCars(dF5);
figure, imshow(L5+1,colormap(hsv(8))), title('Image 5');
[L6,num(6)] = FindCars(dF6);
figure, imshow(L6+1,colormap(hsv(8))), title('Image 6');
[L7,num(7)] = FindCars(dF7);
figure, imshow(L7+1,colormap(hsv(8))), title('Image 7');
[L8,num(8)] = FindCars(dF8);
figure, imshow(L8+1,colormap(hsv(8))), title('Image 8');
[L9,num(9)] = FindCars(dF9);
figure, imshow(L9+1,colormap(hsv(8))), title('Image 9');

total = sum(num);
disp(['Number of cars in all images:  ' num2str(total)]);
%toc

