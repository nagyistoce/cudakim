REM Command file to produce text files for plotting error distributions
REM Uses the fingerprinting method based on the empirical and model-based algorithms
REM Creates files with actual and estimated positions
REM Converts and creates CDF plot files for each of the four algorithms (NN, KNN)
REM *************************************************************************************
java -jar empirical.jar
java -jar modelBased.jar
java -jar score.jar data\fingerprintingResultNN_0.txt data\fingerprintingErrorNN_0.txt
java -jar score.jar data\fingerprintingResultNN_1.txt data\fingerprintingErrorNN_1.txt
java -jar score.jar data\fingerprintingResultNN_2.txt data\fingerprintingErrorNN_2.txt
java -jar score.jar data\fingerprintingResultNN_3.txt data\fingerprintingErrorNN_3.txt
java -jar score.jar data\fingerprintingResultNN_4.txt data\fingerprintingErrorNN_4.txt
java -jar score.jar data\fingerprintingResultKNN_0.txt data\fingerprintingErrorKNN_0.txt
java -jar score.jar data\fingerprintingResultKNN_1.txt data\fingerprintingErrorKNN_1.txt
java -jar score.jar data\fingerprintingResultKNN_2.txt data\fingerprintingErrorKNN_2.txt
java -jar score.jar data\fingerprintingResultKNN_3.txt data\fingerprintingErrorKNN_3.txt
java -jar score.jar data\fingerprintingResultKNN_4.txt data\fingerprintingErrorKNN_4.txt
java -jar score.jar data\fingerprintingResultMNN_0.txt data\fingerprintingErrorMNN_0.txt
java -jar score.jar data\fingerprintingResultMNN_1.txt data\fingerprintingErrorMNN_1.txt
java -jar score.jar data\fingerprintingResultMNN_2.txt data\fingerprintingErrorMNN_2.txt
java -jar score.jar data\fingerprintingResultMNN_3.txt data\fingerprintingErrorMNN_3.txt
java -jar score.jar data\fingerprintingResultMNN_4.txt data\fingerprintingErrorMNN_4.txt
java -jar score.jar data\fingerprintingResultMKNN_0.txt data\fingerprintingErrorMKNN_0.txt
java -jar score.jar data\fingerprintingResultMKNN_1.txt data\fingerprintingErrorMKNN_1.txt
java -jar score.jar data\fingerprintingResultMKNN_2.txt data\fingerprintingErrorMKNN_2.txt
java -jar score.jar data\fingerprintingResultMKNN_3.txt data\fingerprintingErrorMKNN_3.txt
java -jar score.jar data\fingerprintingResultMKNN_4.txt data\fingerprintingErrorMKNN_4.txt
