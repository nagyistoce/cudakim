load 'fingerprintingErrorNN_0.txt'
figure(1)
plot(fingerprintingErrorNN_0(:,1), fingerprintingErrorNN_0(:,2), ':ob')
title('Empirical NNSS Error Distribution')
xlabel('Error distance (meters)')
ylabel('Probability')

load 'fingerprintingErrorKNN_0.txt'
figure(2)
plot(fingerprintingErrorKNN_0(:,1), fingerprintingErrorKNN_0(:,2), ':ob')
title('Empirical KNNSS Error Distribution')
xlabel('Error distance (meters)')
ylabel('Probability')

load 'fingerprintingErrorMNN_0.txt'
figure(3)
plot(fingerprintingErrorMNN_0(:,1), fingerprintingErrorMNN_0(:,2), ':ob')
title('Model-based NNSS Error Distribution')
xlabel('Error distance (meters)')
ylabel('Probability')

load 'fingerprintingErrorMKNN_0.txt'
figure(4)
plot(fingerprintingErrorMKNN_0(:,1), fingerprintingErrorMKNN_0(:,2), ':ob')
title('Model-based KNNSS Error Distribution')
xlabel('Error distance (meters)')
ylabel('Probability')

figure(5)
hold on 
plot(fingerprintingErrorNN_0(:,1), fingerprintingErrorNN_0(:,2), ':+b')
plot(fingerprintingErrorMNN_0(:,1), fingerprintingErrorMNN_0(:,2), ':*r')
hold off
title('NNSS Error Distribution - Empirical(+) vs. Model-based(*)')
xlabel('Error distance (meters)')
ylabel('Probability')

figure(6)
hold on 
plot(fingerprintingErrorKNN_0(:,1), fingerprintingErrorKNN_0(:,2), ':+b')
plot(fingerprintingErrorMKNN_0(:,1), fingerprintingErrorMKNN_0(:,2), ':*r')
hold off
title('KNNSS Error Distribution - Empirical(+) vs. Model-based(*)')
xlabel('Error distance (meters)')
ylabel('Probability')

figure(7)
hold on 
plot(fingerprintingErrorNN_0(:,1), fingerprintingErrorNN_0(:,2), ':+b')
plot(fingerprintingErrorKNN_0(:,1), fingerprintingErrorKNN_0(:,2), ':*r')
hold off
title('Empirical Error Distribution - NNSS(+) vs. KNNSS(*)')
xlabel('Error distance (meters)')
ylabel('Probability')

