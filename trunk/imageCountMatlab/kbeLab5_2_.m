%% KBE, 7/3-2011
clear, close all;


%% Exercise 5.2, counts number of particles in the image "dots.bmp" thats:
%
% 1. on the boundary on the image
% 2. overlapping
% 3. not overlapping
clear, close all;

% Read image and make a bw copy
I=imread('dots.bmp');
dI=double(I)/255;
m = mean(dI(:));
bw=zeros(size(I));
r=find(dI > m);
bw(r)=1;
figure, imshow(bw), title('Original')

% Select structuring element se
nhood = [0 1 0; 1 1 1; 0 1 0];
B = strel('arbitrary',nhood);
bw1 = imerode(bw, B); % Erode image
figure, imshow(bw1), title('Erode')


% bwconncomp(bw,4); - does the same

% Perform extraction of connected components
k = 1;
n = 2;
%A = bw1;
A = bw;
Xres = A;

while 1 % Finds all components
    
h = find(Xres == 1); % Vector for white pixels not yet found
if (isempty(h)) 
    break; % No more components found
end;

% Select picture with one white pixel not yet found
Xk_1 = zeros(size(bw)); 
Xk_1(h(1)) = 1;

while 1 % Finds one component
    Xk = imdilate(Xk_1, B) & A;
    if DiffBWImg(Xk, Xk_1) == 1 % not equal
        k = k + 1;
        Xk_1 = Xk;
    else % equal
        h = find(Xk == 1);
        Xres(h) = n;
        n = n + 1;
        break;
    end;
end;

end;

figure, imshow(Xres), title('Component');
n - 1

colormap=rand(n,3);
figure, imshow(Xres,colormap);

Area_n = zeros(n,1);
for i=1:n
    Area_n(i) = size(find(Xres == i), 1);
end;

ha = hist(Area_n, max(Area_n));
figure;
plot(ha);
title('Histogram for area of particles');

%m = find(ha == max(ha));

Area_p = 69; % Particle size found by analyzing histogram of Area_n manually
count_p_boundary = 0;
count_p_overlapping = 0;
count_p_not_overlapping = 0;

X_boundary = Xres;
X_overlapping = Xres;
X_not_overlapping = Xres;

for i=2:n
    p = find(Xres == i);
    if Area_n(i) < Area_p % Boundary
        count_p_boundary = count_p_boundary + 1;
        X_overlapping(p) = 0; 
        X_not_overlapping(p) = 0; 
    end;
    if Area_n(i) > Area_p % Overlapping
        count_p_overlapping = count_p_overlapping + 1;
        X_not_overlapping(p) = 0; 
        X_boundary(p) = 0; 
    end;
    if Area_n(i) == Area_p % Not overlapping
        count_p_not_overlapping = count_p_not_overlapping + 1;
        X_overlapping(p) = 0; 
        X_boundary(p) = 0; 
    end;
end;

count_p_boundary
count_p_overlapping
count_p_not_overlapping

figure, imshow(X_boundary, colormap), title('Particles on boundary');
figure, imshow(X_overlapping, colormap), title('Particles overlapping');
figure, imshow(X_not_overlapping, colormap), title('Particles not overlapping');

