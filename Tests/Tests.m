% All test results reported in the paper, can ne reproduced simply by running this code.
%
% Generated by Ghasem Alipoor (alipoor@hut.ac.ir) and Karl Skretting (karl.skretting@uis.no)
% Last modification: 13 June 2023
%


clearvars
close all
clc

rng('default')

addpath(genpath('..\Functions'))

%% Test1: Comparing With Other KDL Algorithms
USPS_Fig = figure('color', 'w'); clf;
grid on; hold on; %title('USPS Dataset')
ISOLET_Fig = figure('color', 'w'); clf;
grid on; hold on; %title('ISOLET Dataset')
EEG_Fig = figure('color', 'w'); clf;
grid on; hold on; %title('EEG Dataset')
DistNet_Fig = figure('color', 'w'); clf;
grid on; hold on; %title('DistNet Dataset')

USPS_testAfterBatch = 0;
ISOLET_testAfterBatch = 0;
EEG_testAfterBatch = 0;
DistNet_testAfterBatch = 0;

USPS_TrainTime = zeros(6, 1);
ISOLET_TrainTime = zeros(6, 1);
EEG_TrainTime = zeros(6, 1);
DistNet_TrainTime = zeros(6, 1);
USPS_TestTime = zeros(6, 1);
ISOLET_TestTime = zeros(6, 1);
EEG_TestTime = zeros(6, 1);
DistNet_TestTime = zeros(6, 1);

% KRLS Algorithm
Options = {'KernelType','Poly', 'KernelParam',2.0, 'Q',30, 's', 5, 'LamFunc','Lin', 'Lambda0',.98, 'a_lambda',.8, ...
    'Gamma', 0.1,'GrowCriterion','Representation', 'GrowThresh',0.93, ...
    'BatchSize',10, 'maxL',200, 'newL',190, 'NMissing',10, 'NFolds',5, 'nofTests',20, 'verbose',0, ...
    'Comment',sprintf('Same batch size value is used in both growing and pruning steps.')};
res = KRLSTest('DataSet','USPS', 'nofBatches',60, Options{:});
USPS_TrainTime(1) = 1000*mean(res.ElapsedTime.Grow)/(60*10);   % 60 batches, 10 dictionaries (classes)
USPS_TrainTime(2) = 1000*mean(res.ElapsedTime.Prune)/(60*10);   % 60 batches, 10 dictionaries (classes)
USPS_TestTime(1) = 1000*mean(mean(res.ElapsedTime.Test))/(220*10);  % 10 classes, 220 samples in each class
USPS_TestTime(2) = 1000*mean(mean(res.ElapsedTime.Test))/(220*10);  % 10 classes, 220 samples in each class
USPS_testAfterBatch = max(USPS_testAfterBatch, res.Options.testAfterBatch);
figure(USPS_Fig);
plot([0 res.Options.testAfterBatch], squeeze(mean(res.Acc(:, 1, :), 3)), ...
    '-o', 'MarkerSize',10, 'LineWidth', 3, 'DisplayName', 'KRLS')
res = KRLSTest('DataSet','ISOLET', 'nofBatches',60, Options{:});
ISOLET_TrainTime(1) = 1000*mean(res.ElapsedTime.Grow)/(60*26);   % 60 batches, 26 dictionaries (classes)
ISOLET_TrainTime(2) = 1000*mean(res.ElapsedTime.Prune)/(60*26);   % 60 batches, 26 dictionaries (classes)
ISOLET_TestTime(1) = 1000*mean(mean(res.ElapsedTime.Test))/(60*26);  % 26 classes, 60 samples in each class
ISOLET_TestTime(2) = 1000*mean(mean(res.ElapsedTime.Test))/(60*26);  % 26 classes, 60 samples in each class
ISOLET_testAfterBatch = max(ISOLET_testAfterBatch, res.Options.testAfterBatch);
figure(ISOLET_Fig);
plot([0 res.Options.testAfterBatch], squeeze(mean(res.Acc(:, 1, :), 3)), ...
    '-o', 'MarkerSize',10, 'LineWidth', 3, 'DisplayName', 'KRLS')
res = KRLSTest('DataSet','EEG', 'nofBatches',60, Options{:});
EEG_TrainTime(1) = 1000*mean(res.ElapsedTime.Grow)/(60*2);   % 60 batches, 2 dictionaries (classes)
EEG_TrainTime(2) = 1000*mean(res.ElapsedTime.Prune)/(60*2);   % 60 batches, 2 dictionaries (classes)
EEG_TestTime(1) = 1000*mean(mean(res.ElapsedTime.Test))/(473*2);  % 2 classes, 473 samples in each class
EEG_TestTime(2) = 1000*mean(mean(res.ElapsedTime.Test))/(473*2);  % 2 classes, 473 samples in each class
EEG_testAfterBatch = max(EEG_testAfterBatch, res.Options.testAfterBatch);
figure(EEG_Fig);
plot([0 res.Options.testAfterBatch], squeeze(mean(res.Acc(:, 1, :), 3)), ...
    '-o', 'MarkerSize',10, 'LineWidth', 3, 'DisplayName', 'KRLS')
res = KRLSTest('DataSet','DistNet', 'nofBatches',60, Options{:});
DistNet_TrainTime(1) = 1000*mean(res.ElapsedTime.Grow)/(60*5);   % 60 batches, 5 dictionaries (classes)
DistNet_TrainTime(2) = 1000*mean(res.ElapsedTime.Prune)/(60*5);   % 60 batches, 5 dictionaries (classes)
DistNet_TestTime(1) = 1000*mean(mean(res.ElapsedTime.Test))/(80*5);  % 5 classes, 80 samples in each class
DistNet_TestTime(2) = 1000*mean(mean(res.ElapsedTime.Test))/(80*5);  % 5 classes, 80 samples in each class
DistNet_testAfterBatch = max(DistNet_testAfterBatch, res.Options.testAfterBatch);
figure(DistNet_Fig);
plot([0 res.Options.testAfterBatch], squeeze(mean(res.Acc(:, 1, :), 3)), ...
    '-o', 'MarkerSize',10, 'LineWidth', 3, 'DisplayName', 'KRLS')

% OKDLRS Algorithm
Options = {'KernelType','Poly', 'KernelParam',2.0, 'Q',30, 's', 5, 'Rho',1.1, 'mu_m',0.2, ...
    'BatchSize',10, 'maxL',200, 'newL',190, 'NMissing',1, 'NFolds',5, 'nofTests',20, 'verbose',0, ...
    'Comment',sprintf('Same batch size value is used in both growing and pruning steps.')};
res = OKDLRSTest('DataSet','USPS', 'nofBatches',60, Options{:});
USPS_TrainTime(3) = 1000*mean(res.ElapsedTime.Train)/(60*10);   % 60 batches, 10 dictionaries (classes)
USPS_TestTime(3) = 1000*mean(mean(res.ElapsedTime.Test))/(220*10);  % 10 classes, 220 samples in each class
USPS_testAfterBatch = max(USPS_testAfterBatch, res.Options.testAfterBatch);
figure(USPS_Fig);
plot([0 res.Options.testAfterBatch], squeeze(mean(res.Acc(:, 1, :), 3)), ...
    '-*', 'MarkerSize',8, 'LineWidth', 2, 'DisplayName', 'OKDLRS')
res = OKDLRSTest('DataSet','ISOLET', 'nofBatches',60, Options{:});
ISOLET_TrainTime(3) = 1000*mean(res.ElapsedTime.Train)/(60*26);   % 60 batches, 26 dictionaries (classes)
ISOLET_TestTime(3) = 1000*mean(mean(res.ElapsedTime.Test))/(60*26);  % 26 classes, 60 samples in each class
ISOLET_testAfterBatch = max(ISOLET_testAfterBatch, res.Options.testAfterBatch);
figure(ISOLET_Fig);
plot([0 res.Options.testAfterBatch], squeeze(mean(res.Acc(:, 1, :), 3)), ...
    '-*', 'MarkerSize',8, 'LineWidth', 2, 'DisplayName', 'OKDLRS')
res = OKDLRSTest('DataSet','EEG', 'nofBatches',60, Options{:});
EEG_TrainTime(3) = 1000*mean(res.ElapsedTime.Train)/(60*2);   % 60 batches, 2 dictionaries (classes)
EEG_TestTime(3) = 1000*mean(mean(res.ElapsedTime.Test))/(473*2);  % 2 classes, 473 samples in each class
EEG_testAfterBatch = max(EEG_testAfterBatch, res.Options.testAfterBatch);
figure(EEG_Fig);
plot([0 res.Options.testAfterBatch], squeeze(mean(res.Acc(:, 1, :), 3)), ...
    '-*', 'MarkerSize',8, 'LineWidth', 2, 'DisplayName', 'OKDLRS')
res = OKDLRSTest('DataSet','DistNet', 'nofBatches',60, Options{:});
DistNet_TrainTime(3) = 1000*mean(res.ElapsedTime.Train)/(60*5);   % 60 batches, 5 dictionaries (classes)
DistNet_TestTime(3) = 1000*mean(mean(res.ElapsedTime.Test))/(80*5);  % 5 classes, 80 samples in each class
DistNet_testAfterBatch = max(DistNet_testAfterBatch, res.Options.testAfterBatch);
figure(DistNet_Fig);
plot([0 res.Options.testAfterBatch], squeeze(mean(res.Acc(:, 1, :), 3)), ...
    '-*', 'MarkerSize',8, 'LineWidth', 2, 'DisplayName', 'OKDLRS')

% OKDLFB Algorithm
Options = {'KernelType','Poly', 'KernelParam',2.0, 'Q',30, 's', 5, 'GrowingTrsh',0.9, 'lambda',0.05, 'mu',0.05, ...
    'BatchSize',10, 'maxL',200, 'newL',190, 'NMissing',1, 'NFolds',5, 'nofTests',20, 'verbose',0, ...
    'Comment',sprintf('Same batch size value is used in both growing and pruning steps.')};
res = OKDLFBTest('DataSet','USPS', 'nofBatches',60, Options{:});
USPS_TrainTime(4) = 1000*mean(res.ElapsedTime.Train)/(60*10);   % 60 batches, 10 dictionaries (classes)
USPS_TestTime(4) = 1000*mean(mean(res.ElapsedTime.Test))/(220*10);  % 10 classes, 220 samples in each class
USPS_testAfterBatch = max(USPS_testAfterBatch, res.Options.testAfterBatch);
figure(USPS_Fig);
plot([0 res.Options.testAfterBatch], squeeze(mean(res.Acc(:, 1, :), 3)), ...
    '-x', 'MarkerSize',8, 'LineWidth', 2, 'DisplayName', 'OKDLFB')
res = OKDLFBTest('DataSet','ISOLET', 'nofBatches',60, Options{:});
ISOLET_TrainTime(4) = 1000*mean(res.ElapsedTime.Train)/(60*26);   % 60 batches, 26 dictionaries (classes)
ISOLET_TestTime(4) = 1000*mean(mean(res.ElapsedTime.Test))/(60*26);  % 26 classes, 60 samples in each class
ISOLET_testAfterBatch = max(ISOLET_testAfterBatch, res.Options.testAfterBatch);
figure(ISOLET_Fig);
plot([0 res.Options.testAfterBatch], squeeze(mean(res.Acc(:, 1, :), 3)), ...
    '-x', 'MarkerSize',8, 'LineWidth', 2, 'DisplayName', 'OKDLFB')
res = OKDLFBTest('DataSet','EEG', 'nofBatches',60, Options{:});
EEG_testAfterBatch = max(EEG_testAfterBatch, res.Options.testAfterBatch);
EEG_TrainTime(4) = 1000*mean(res.ElapsedTime.Train)/(60*2);   % 60 batches, 2 dictionaries (classes)
EEG_TestTime(4) = 1000*mean(mean(res.ElapsedTime.Test))/(473*2);  % 2 classes, 473 samples in each class
figure(EEG_Fig);
plot([0 res.Options.testAfterBatch], squeeze(mean(res.Acc(:, 1, :), 3)), ...
    '-x', 'MarkerSize',8, 'LineWidth', 2, 'DisplayName', 'OKDLFB')
res = OKDLFBTest('DataSet','DistNet', 'nofBatches',60, Options{:});
DistNet_TrainTime(4) = 1000*mean(res.ElapsedTime.Train)/(60*5);   % 60 batches, 5 dictionaries (classes)
DistNet_TestTime(4) = 1000*mean(mean(res.ElapsedTime.Test))/(80*5);  % 5 classes, 80 samples in each class
DistNet_testAfterBatch = max(DistNet_testAfterBatch, res.Options.testAfterBatch);
figure(DistNet_Fig);
plot([0 res.Options.testAfterBatch], squeeze(mean(res.Acc(:, 1, :), 3)), ...
    '-x', 'MarkerSize',8, 'LineWidth', 2, 'DisplayName', 'OKDLFB')

% sKDL Algorithm
Options = {'KernelType','Poly', 'KernelParam',2.0, 'Q',30, 's', 5, 'GrowingTrsh',0.9, ...
    'LearningRate_Rep',1e-13 ,'LearningRate_Dic',1e-5 ,'ISTA_Thrsh',0.01 ,'lam_Rep',0.005 ,'lam_Dic',0.001 ...
    'BatchSize',10, 'maxL',200, 'newL',190, 'NMissing',1, 'NFolds',5, 'nofTests',20, 'verbose',0, ...
    'Comment',sprintf('Same batch size value is used in both growing and pruning steps.')};
res = sKDLTest('DataSet','USPS', 'nofBatches',60, Options{:});
USPS_TrainTime(5) = 1000*mean(res.ElapsedTime.Train)/(60*10);   % 60 batches, 10 dictionaries (classes)
USPS_TestTime(5) = 1000*mean(mean(res.ElapsedTime.Test))/(220*10);  % 10 classes, 220 samples in each class
USPS_testAfterBatch = max(USPS_testAfterBatch, res.Options.testAfterBatch);
figure(USPS_Fig);
plot([0 res.Options.testAfterBatch], squeeze(mean(res.Acc(:, 1, :), 3)), ...
    '-s', 'MarkerSize',10, 'LineWidth', 3, 'DisplayName', 'sKDL')
res = sKDLTest('DataSet','ISOLET', 'nofBatches',60, Options{:});
ISOLET_TrainTime(5) = 1000*mean(res.ElapsedTime.Train)/(60*26);   % 60 batches, 26 dictionaries (classes)
ISOLET_TestTime(5) = 1000*mean(mean(res.ElapsedTime.Test))/(60*26);  % 26 classes, 60 samples in each class
ISOLET_testAfterBatch = max(ISOLET_testAfterBatch, res.Options.testAfterBatch);
figure(ISOLET_Fig);
plot([0 res.Options.testAfterBatch], squeeze(mean(res.Acc(:, 1, :), 3)), ...
    '-s', 'MarkerSize',10, 'LineWidth', 3, 'DisplayName', 'sKDL')
res = sKDLTest('DataSet','EEG', 'nofBatches',60, Options{:});
EEG_TrainTime(5) = 1000*mean(res.ElapsedTime.Train)/(60*2);   % 60 batches, 2 dictionaries (classes)
EEG_TestTime(5) = 1000*mean(mean(res.ElapsedTime.Test))/(473*2);  % 2 classes, 473 samples in each class
EEG_testAfterBatch = max(EEG_testAfterBatch, res.Options.testAfterBatch);
figure(EEG_Fig);
plot([0 res.Options.testAfterBatch], squeeze(mean(res.Acc(:, 1, :), 3)), ...
    '-s', 'MarkerSize',10, 'LineWidth', 3, 'DisplayName', 'sKDL')
res = sKDLTest('DataSet','DistNet', 'nofBatches',60, Options{:});
DistNet_TrainTime(5) = 1000*mean(res.ElapsedTime.Train)/(60*5);   % 60 batches, 5 dictionaries (classes)
DistNet_TestTime(5) = 1000*mean(mean(res.ElapsedTime.Test))/(80*5);  % 5 classes, 80 samples in each class
DistNet_testAfterBatch = max(DistNet_testAfterBatch, res.Options.testAfterBatch);
figure(DistNet_Fig);
plot([0 res.Options.testAfterBatch], squeeze(mean(res.Acc(:, 1, :), 3)), ...
    '-s', 'MarkerSize',10, 'LineWidth', 3, 'DisplayName', 'sKDL')

% KMOD Batch Algorithm
Options = {'KernelType','Poly', 'KernelParam',2.0, 'Q',30, 's', 5, ...
    'NIterations',100, 'NMissing',1, 'NFolds',5, 'verbose',0, 'Comment',''};
res = KMODTest('DataSet','USPS', Options{:});
USPS_TrainTime(6) = 1000*mean(res.ElapsedTime.Train)/(100*10);     % 100 iterations, 10 dictionaries (classes)
USPS_TestTime(6) = 1000*mean(res.ElapsedTime.Test)/(220*10);  % 10 classes, 220 samples in each class
figure(USPS_Fig);
plot([0 USPS_testAfterBatch], repmat(squeeze(mean(res.Acc(1, :), 2)), length(USPS_testAfterBatch) + 1, 1), ...
    '-d', 'MarkerSize',10, 'LineWidth', 3, 'DisplayName', 'KMOD')
res = KMODTest('DataSet','ISOLET', Options{:});
ISOLET_TrainTime(6) = 1000*mean(res.ElapsedTime.Train)/(100*26);   % 100 iterations, 26 dictionaries (classes)
ISOLET_TestTime(6) = 1000*mean(res.ElapsedTime.Test)/(60*26);  % 26 classes, 60 samples in each class
figure(ISOLET_Fig);
plot([0 ISOLET_testAfterBatch], repmat(squeeze(mean(res.Acc(1, :), 2)), length(ISOLET_testAfterBatch) + 1, 1), ...
    '-d', 'MarkerSize',10, 'LineWidth', 3, 'DisplayName', 'KMOD')
res = KMODTest('DataSet','EEG', Options{:});
EEG_TrainTime(6) = 1000*mean(res.ElapsedTime.Train)/(100*2);   % 100 iterations, 2 dictionaries (classes)
EEG_TestTime(6) = 1000*mean(res.ElapsedTime.Test)/(473*2);  % 2 classes, 473 samples in each class
figure(EEG_Fig);
plot([0 EEG_testAfterBatch], repmat(squeeze(mean(res.Acc(1, :), 2)), length(EEG_testAfterBatch) + 1, 1), ...
    '-d', 'MarkerSize',10, 'LineWidth', 3, 'DisplayName', 'KMOD')
res = KMODTest('DataSet','DistNet', Options{:});
DistNet_TrainTime(6) = 1000*mean(res.ElapsedTime.Train)/(100*5);   % 100 iterations, 5 dictionaries (classes)
DistNet_TestTime(6) = 1000*mean(res.ElapsedTime.Test)/(80*5);  % 5 classes, 80 samples in each class
figure(DistNet_Fig);
plot([0 DistNet_testAfterBatch], repmat(squeeze(mean(res.Acc(1, :), 2)), length(DistNet_testAfterBatch) + 1, 1), ...
    '-d', 'MarkerSize',10, 'LineWidth', 3, 'DisplayName', 'KMOD')

% Summary Training and Test Times
AlgNames = {'KRLS-Growing'; 'KRLS-Pruning'; 'OKDLRS'; 'OKDLFB'; 'sKDL'; 'KMOD'};
T = table(AlgNames, USPS_TrainTime, ISOLET_TrainTime, EEG_TrainTime, DistNet_TrainTime, USPS_TestTime, ISOLET_TestTime, EEG_TestTime, DistNet_TestTime) %#ok<NOPTS>

figure(USPS_Fig);
ax = gca;
ax.FontSize = 16;
ylabel('Classification Accuracy', 'FontSize', 18)
xlabel('Number of Mini-Batches', 'FontSize', 18)
legend('FontSize', 22, 'Location','southeast')
xticks(0:10:60);
figure(ISOLET_Fig);
ax = gca;
ax.FontSize = 16;
ylabel('Classification Accuracy', 'FontSize', 18)
xlabel('Number of Mini-Batches', 'FontSize', 18)
legend('FontSize', 22, 'Location','southeast')
xticks(0:10:60);
figure(EEG_Fig);
ax = gca;
ax.FontSize = 16;
ylabel('Classification Accuracy', 'FontSize', 18)
xlabel('Number of Mini-Batches', 'FontSize', 18)
legend('FontSize', 22, 'Location','southeast')
xticks(0:10:60);
figure(DistNet_Fig);
ax = gca;
ax.FontSize = 16;
ylabel('Classification Accuracy', 'FontSize', 18)
xlabel('Number of Mini-Batches', 'FontSize', 18)
legend('FontSize', 22, 'Location','southeast')
xticks(0:10:60);

%% Test2: Noise Effect on the KRLS Algorithm
Options = {'KernelType','Poly', 'KernelParam',2.0, 'Q',30, 's', 5, 'LamFunc','Lin', 'Lambda0',.98, 'a_lambda',.8, ...
    'Gamma', 0.1,'GrowCriterion','Representation', 'GrowThresh',0.93, ...
    'BatchSize',10, 'maxL',200, 'newL',190, 'NMissing',10, 'NFolds',5, 'nofTests',20, 'verbose',1, ...
    'Comment',sprintf('Same batch size value is used in both growing and pruning steps.')};
KRLSTest('DataSet','USPS', 'nofBatches',60, Options{:});
KRLSTest('DataSet','ISOLET', 'nofBatches',60, Options{:});
KRLSTest('DataSet','EEG', 'nofBatches',60, Options{:});
KRLSTest('DataSet','DistNet', 'nofBatches',60, Options{:});