clear all;
close all;
clc;

addpath('..');
addpath('../../data');

data_file = 'SSL,set=3,data.mat';
split_file = 'SSL,set=3,splits,labeled=100.mat';
load(data_file);
load(split_file);

num_splits = size(idxLabs,1);
K = create_kernel(X, 'rbf', 'sig', 10);

for i = 1:num_splits
   labeled_ind = idxLabs(i,:);
   unlabeled_ind = idxUnls(i,:);   
   K_train = K(labeled_ind, labeled_ind);
   K_test = K(unlabeled_ind, labeled_ind);
   n = size(K_train, 1);
   C = ones(n,1);
   y_train = y(labeled_ind);
   M = K_train.*(y_train*y_train');
   alpha = two_svm( M, C );
   predict = sign(K_test*diag(y_train)*alpha);
   disp('------------------------------------------');
   fprintf('(TWO-SVM) Split #%d: error = %f\n', i, sum(predict ~= y(unlabeled_ind))/length(unlabeled_ind));
   
   % Built in Matlab SVM
   svmstruct = svmtrain(X(labeled_ind, :), y_train, 'kernel_function', 'rbf', 'rbf_sigma', 10);
   matlab_predict = svmclassify(svmstruct, X(unlabeled_ind, :));
   fprintf('(Matlab-SVM) Split #%d: error = %f\n', i, sum(matlab_predict ~= y(unlabeled_ind))/length(unlabeled_ind));
   disp('------------------------------------------');
end