clear;
close all;
clc;
%% input the data

img_name='Indian_pines';
addpath(genpath('.\'));
R=importdata(['.\data\' img_name '_corrected.mat']);
% X=importdata(['.\code_SS-RLRA\data\' img_name '_corrected_ss.mat']);
gt=importdata(['.\data\' img_name '_gt.mat']);

if strcmp(img_name, 'Salinas')
   CTrain = [68 67 67 67 67 68 68 70 69 67 67 67 68 67 68 68];
end
if strcmp(img_name, 'Indian_pines')
    CTrain = [6 144 84 24 50 75 3 49 2 97 247 62 22 130 38 10];
end
[loc_train, loc_test, CTest] = Generating_training_testing(gt,CTrain);

[m,n,d]=size(R);
% X=R./max(R(:));
%% parameter settings
[idxx,idxy,idxz]=split_tensor(m,n,d,6,6,200);

opts.mu = 1e-4;
opts.tol = 1e-5;
opts.rho = 1.1;
opts.max_iter = 500;
opts.DEBUG = 1;
opts.idxx=idxx;
opts.idxy=idxy;
opts.idxz=idxz;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opts.alpha=1; % the negative low rnak term  
        opts.beta=1; % the band-wise 
        opts.lambda=0.0001; % the error term
        [Y1,E1,Y2,E2,obj] = GLALRR(R,opts);
        [OA_SVM1, OA_SVM2] =Classification_V2(Y1,Y1./(max(Y1(:))),gt,50,CTrain);
        disp(OA_SVM1)