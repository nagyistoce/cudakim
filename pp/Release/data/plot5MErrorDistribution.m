load 'fingerprintingErrorMNN_0.txt'
load 'fingerprintingErrorMNN_1.txt'
load 'fingerprintingErrorMNN_2.txt'
load 'fingerprintingErrorMNN_3.txt'
load 'fingerprintingErrorMNN_4.txt'
figure(1)
hold on 
plot(fingerprintingErrorMNN_0(:,1), fingerprintingErrorMNN_0(:,2), ':xr')
plot(fingerprintingErrorMNN_1(:,1), fingerprintingErrorMNN_1(:,2), ':xg')
plot(fingerprintingErrorMNN_2(:,1), fingerprintingErrorMNN_2(:,2), ':xb')
plot(fingerprintingErrorMNN_3(:,1), fingerprintingErrorMNN_3(:,2), ':xc')
plot(fingerprintingErrorMNN_4(:,1), fingerprintingErrorMNN_4(:,2), ':xm')
hold off
title('Model-based NNSS Error Distribution (5 iterations)')
xlabel('Error distance (meters)')
ylabel('Probability')

load 'fingerprintingErrorMKNN_0.txt'
load 'fingerprintingErrorMKNN_1.txt'
load 'fingerprintingErrorMKNN_2.txt'
load 'fingerprintingErrorMKNN_3.txt'
load 'fingerprintingErrorMKNN_4.txt'
figure(2)
hold on 
plot(fingerprintingErrorMKNN_0(:,1), fingerprintingErrorMKNN_0(:,2), ':xr')
plot(fingerprintingErrorMKNN_1(:,1), fingerprintingErrorMKNN_1(:,2), ':xg')
plot(fingerprintingErrorMKNN_2(:,1), fingerprintingErrorMKNN_2(:,2), ':xb')
plot(fingerprintingErrorMKNN_3(:,1), fingerprintingErrorMKNN_3(:,2), ':xc')
plot(fingerprintingErrorMKNN_4(:,1), fingerprintingErrorMKNN_4(:,2), ':xm')
hold off
title('Model-based KNNSS Error Distribution (5 iterations)')
xlabel('Error distance (meters)')
ylabel('Probability')
