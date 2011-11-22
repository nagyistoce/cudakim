load 'fingerprintingErrorNN_0.txt'
load 'fingerprintingErrorNN_1.txt'
load 'fingerprintingErrorNN_2.txt'
load 'fingerprintingErrorNN_3.txt'
load 'fingerprintingErrorNN_4.txt'
figure(1)
hold on 
plot(fingerprintingErrorNN_0(:,1), fingerprintingErrorNN_0(:,2), ':xr')
plot(fingerprintingErrorNN_1(:,1), fingerprintingErrorNN_1(:,2), ':xg')
plot(fingerprintingErrorNN_2(:,1), fingerprintingErrorNN_2(:,2), ':xb')
plot(fingerprintingErrorNN_3(:,1), fingerprintingErrorNN_3(:,2), ':xc')
plot(fingerprintingErrorNN_4(:,1), fingerprintingErrorNN_4(:,2), ':xm')
hold off
title('Empirical NNSS Error Distribution (5 iterations)')
xlabel('Error distance (meters)')
ylabel('Probability')

load 'fingerprintingErrorKNN_0.txt'
load 'fingerprintingErrorKNN_1.txt'
load 'fingerprintingErrorKNN_2.txt'
load 'fingerprintingErrorKNN_3.txt'
load 'fingerprintingErrorKNN_4.txt'
figure(2)
hold on 
plot(fingerprintingErrorKNN_0(:,1), fingerprintingErrorKNN_0(:,2), ':xr')
plot(fingerprintingErrorKNN_1(:,1), fingerprintingErrorKNN_1(:,2), ':xg')
plot(fingerprintingErrorKNN_2(:,1), fingerprintingErrorKNN_2(:,2), ':xb')
plot(fingerprintingErrorKNN_3(:,1), fingerprintingErrorKNN_3(:,2), ':xc')
plot(fingerprintingErrorKNN_4(:,1), fingerprintingErrorKNN_4(:,2), ':xm')
hold off
title('Empirical KNNSS Error Distribution (5 iterations)')
xlabel('Error distance (meters)')
ylabel('Probability')
