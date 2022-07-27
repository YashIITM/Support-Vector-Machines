clear;
dir;
%EXAMPLE DATASET 1
%Let's start by an example 2D dataset which can be separated by a linear
%boundary
% Load from ex6data1: 
% You will have X, y in your environment
load('ex6data1.mat');

% Plot training data
figure;
plotData(X, y);
xlabel('x data points');
ylabel('y data points');
title('Example Training Data 1');

%Let's train our data using svmTrain.m, with the C parameter as 1
C=1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
figure;
visualizeBoundaryLinear(X, y, model);
xlabel('x data points');
ylabel('y data points');
title('SVM Model with C = 1');


%SVMs with Gaussian Kernels : A test run/example
x1 = [1 2 1]; 
x2 = [0 4 -1];
sigma = 2;
sim = gaussianKernel(x1,x2,sigma);
fprintf('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f : \n\t%g\n', sigma, sim);

%EXAMPLE DATASET 2
% Load from ex6data2: 
% You will have X, y in your environment
load('ex6data2.mat');

% Plot training data
figure;
plotData(X, y);
xlabel('x data points');
ylabel('y data points');
title('Example Training Data 2');

% SVM Parameters
C = 1; sigma = 0.1;

% We set the tolerance and max_passes lower here so that the code will run faster. However, in practice, 
% you will want to run the training to convergence.
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
figure;
visualizeBoundary(X, y, model);
xlabel('x data points');
ylabel('y data points');
title('SVM training with C = 1');

%EXAMPLE DATASET 3
% Load from ex6data3: 
% You will have X, y in your environment
load('ex6data3.mat');

% Plot training data
figure;
plotData(X, y);
xlabel('x data points');
ylabel('y data points');
title('Example Training Data 3');


% Try different SVM Parameters here
[C, sigma] = dataset3Params(X, y, Xval, yval);

% Train the SVM
model = svmTrain(X, y, C, @(x1, x2)gaussianKernel(x1, x2, sigma));
figure;
visualizeBoundary(X, y, model);
xlabel('x data points');
ylabel('y data points');
title('SVM with C in "dataset3Params"');




