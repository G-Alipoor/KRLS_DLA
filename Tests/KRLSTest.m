function res = KRLSTest(varargin)
%
% KRLSTest    Function used to test the Kernel RLS-DLA over different datasets
%
% varargin enables the user to specify optional parameter/value pairs to control the algorithm.
%     Valid parameter strings are as follow:
%         DataSet:                 Dataset; a string. Options are 'USPS' (Default), 'ISOLET', 'EEG', and 'DistNet"
%         KernelType:            Kernel type; a string. Options are 'Lin', 'Gauss' (Default) or 'Poly'
%         KernelParam:          Kernel parameter; a scalar number (Default: 4.0)
%         Q:                           Number of dictionary atoms; a scalar integer number (Default: 30)
%         s:                            Sparsity level, , in training and test; a scalar integer number (Default: 5)
%         Lambda0:                Initial value for forgetting factor; a scalar number smaller than 1 (Default: 0.98)
%         a_lambda:               When lambda reach 1, in proportion to number of batches;
%                                           a scalar number smaller than 1 (Default: 0.8)
%         LamFunc:                The function based on which lambda changes over the time;a string
%                                           Options are 'Constant1', '199', 'qud', 'Lin' (Default) 'cub', 'exp', 'exp'
%         Gamma:                  l2-regularization parameter in the dictionary learning cost function;
%                                           a scalar number (Default: 0.2)
%         GrowCriterion:    The growing criterion used to check the informativeness of the new sample;
%                                           a string. Options are 'Coherency', 'Representation'(Default) and 'Projection'
%         GrowThresh:      The growing treshold; a scalar number smaller than or equal to 1 (Default: 0.95)
%         BatchSize:               Number of vectors for batch when growing; a scalar number (Default: 10)
%         nofBatches:             Total number of batches (may use training data more than once);
%                                           a scalar number (Default: 150)
%         maxL:                      Profile size beyond which profile is pruned; a scalar number (Default: 200)
%         newL:                      Profile size after pruning; a scalar number (Default: 180)
%         NMissing:                Number of states for mission indice ratios; a scalar number (Default: 5)
%         NFolds:                    Number of folds in kfold cross-validation; a scalar number (Default: 5)
%         nofTests:                  Number of tests to do as batches are processed; a scalar number (Default: 20)
%         verbose:                   0: show only errors, 1: (Default), and warnings, 2: and more
%         extraText:                 A short text used in file name; a string (Default: '')
%         Comment:                Any comment to be saved with the test result; a string (Default: '')
%
% The result is stored in a file which name includes:
%   DataSet, KernelType, extraText, Q, s, floor(Gamma*100), BatchSize, NMissing
% Examples:
%   res = KRLSTest('BatchSize',10, 'NMissing',4, 'nofTests',10);
%   res = KRLSTest('NMissing',5, 'nofTests',10);
%   res = KRLSTest('NMissing',2, 'nofTests',25, 'NFolds',2, 'KernelType','Poly', ...
%               'KernelParam',1.0, 'Gamma',0.8, 'verbose',0);
%   res = KRLSTest();
%   res = KRLSTest('KernelType','Poly', 'KernelParam',1.0);
%
% Generated by Ghasem Alipoor (alipoor@hut.ac.ir) and Karl Skretting (karl.skretting@uis.no)
% Last modification: 13 June 2023
%

%% Parameter setting
% Setting input (or default) parameter values
pnames = {'DataSet', ...     % Either 'USPS', 'ISOLET', 'EEG', or 'DistNet'
    'KernelType', ...  % 'Gauss', 'Poly' or 'Lin'
    'KernelParam', ... % ex. 2.0
    'Q', ...        % Number of dictionary atoms
    's', ...        % Sparsity level, in training and test
    'Lambda0', ...   % initial value for forgetting factor, ex. 0.9
    'a_lambda', ... % when lambda reach 1, a_lambda <= 1.0, ex. 0.9
    'LamFunc', ...  % The function based on which lambda changes over the time
    'Gamma', ...    % l2-regularization parameter in the dictionary learning cost function
    'GrowCriterion', ... % Growimg Criterion
    'GrowThresh', ... % Growing threshold
    'BatchSize', ... % nof vectors for batch when growing, ex. 10
    'nofBatches', ... % may use training data more than once, ex. 100
    'maxL', ...     % max value for profile L, ex. 400
    'newL', ...     % Profile L after pruning, ex. 300
    'NMissing', ... % Number of states for mission indice ratios, ex. 10
    'NFolds', ...   % Number of folds in kfold cross-validation, ex. 10
    'nofTests', ... % number of tests to do as batches are processed, ex. 30
    'verbose', ... % 1 (default) to show warnings, ex. 2
    'extraText', ...% a short text used in file name, ex. 'Ver1'
    'Comment', ...  % Any comment to be saved with the test result, ex. 'This is a test.'
    };
dflts = {'USPS', 'Poly', 2.0, 30, 5, ...  % DataSet, KernelType, KernelParam, Q, s
    0.98, 0.8, 'Lin', ... % Lambda0, a_lambda, LamFunc (most 'reasonable' values are good)
    0.2, 'Representation', .95, 10, 150, ...  % Gamma, GrowCriterion, GrowThresh, BatchSize, nofBatches,
    200, 190, 5, 5, 20, ...     % maxL, newL, NMissing, NFolds, nofTests
    1, '', '', ... % verbose, extraText, Comment
    };
[DataSet, KernelType, KernelParam, Q, s, Lambda0, a_lambda, LamFunc, Gamma, ...
    GrowCriterion, GrowThresh, BatchSize, nofBatches, maxL, newL, NMissing, NFolds, ...
    nofTests, verbose, extraText, Comment] = getarg(varargin,pnames,dflts); %#ok<ASGLU>

% Test version; a string attached to the name of the saved file
TestVersion = sprintf('%s%iQ%is%ig%03ib%im%i%s', ...
    KernelType,KernelParam,Q,s,floor(Gamma*100),BatchSize,NMissing,extraText);
resFileName = fullfile('..\Results\', sprintf('KRLS_%s_%s', DataSet, TestVersion));

Options = [];
if verbose
    fprintf('\n%s has been started using parameters: \n',mfilename())
    for i = 1:length(pnames)
        fprintf('  %20s : %s\n', pnames{i}, string(eval(pnames{i})));
        Options = setfield(Options,string(pnames{i}),eval(pnames{i})); %#ok<SFLD>
    end

    fprintf('This gives:\n');
    fprintf('  %20s : %s\n', 'TestVersion', TestVersion);
    fprintf('  %20s : %s.mat\n\n', 'resFileName', resFileName);
    % pause  % can be included to check parameters
else
    fprintf('\n%s has been started.\n',mfilename())
end

% set some more parameters, and derive variables from parameters
if nofTests < nofBatches
    testAfterBatch = floor(linspace(nofBatches/nofTests,nofBatches,nofTests));
else
    testAfterBatch = 1:nofBatches;
    nofTests = nofBatches;
end

%% Do Experiment or load results
if (exist([resFileName,'.mat'],'file') == 2)
    fprintf('\nLoad results from file instead of doing experiment once more.\n');
    if verbose
        fprintf('Change TestVersion (in the start of this m-file) to make a new test\n');
    end
    load([resFileName,'.mat']);  %#ok<LOAD> % load res
    field_names_list = fieldnames(res); %#ok<NODEF>
    for k = 1:length(field_names_list)
        field_name = field_names_list{k};
        % disp( [field_name, ' = res.', field_name, ';'] )
        eval([field_name, ' = res.',field_name,';']);
    end
else  % do the experiments
    Options.testAfterBatch = testAfterBatch;
    ElapsedTime = struct('Grow',zeros(1, NFolds), 'Prune',zeros(1, NFolds), 'Test',zeros(nofTests + 1, NFolds));

    % Defining the forgetting factor (Lambda) values over all batches
    i1 = round(min(a_lambda,1.0) * nofBatches);
    i2 = nofBatches - i1;
    switch LamFunc
        case 'Constant1'
            LambdaBatch = ones(1, nofBatches);
        case '199'
            % changes between Lambda0 1 and 1.0
            LambdaBatch = [Lambda0*ones(1,i1), ones(1,i2)];
        case 'Lin'
            % grows linearly up to a_lambda of total
            LambdaBatch = [linspace(Lambda0, 1.0, i1), ones(1,i2)];
        case 'qud'
            % grows quadraticly up to a_lambda of total
            LambdaBatch = [1 - (1 - Lambda0)*((1 - (0:(i1-1))/(i1-1)).^2), ones(1,i2)];
        case 'cub'
            % grows cubicly up to a_lambda of total
            LambdaBatch = [1 - (1 - Lambda0)*((1 - (0:(i1-1))/(i1-1)).^3), ones(1,i2)];
        case 'exp'
            % grows exponentially up to a_lambda of total
            LambdaBatch = [1 - (1 - Lambda0)*((1/2).^linspace(0,8,i1)), ones(1,i2)];
    end
    % may uncomment to plot LambdaBatch
    % figure(1); clf; plot(1:nofBatches, LambdaBatch);
    % title('Value of \lambda for each batch'); xlabel('Batch number');

    %% Loading data
    switch DataSet
        case 'USPS'
            load('..\Data\USPS'); %#ok<LOAD>
            Data = double(data)/255;
        case 'ISOLET'
            A = readtable('..\Data\ISOLET.csv');
            Data = zeros(617, 300, 26);
            for c = 1:26
                Temp = table2array(A(string(A.class) == sprintf("'%d'", c), 1:end-1))';
                if size(Temp, 2) < 300 % To account for missing entries
                    Idx = randi(size(Temp, 2), 300 - size(Temp, 2), 1);
                    Temp = cat(2, Temp, Temp(:, Idx));
                end
                Data(:, :, c) = Temp;
            end
        case 'EEG'
            Trials = dir('..\Data\EEG\h*.mat');
            Frame_Length = 250;
            % Filters' Bands
            DeltaRange = [0 4];
            ThetaRange = [4 8];
            AlphaRange = [8 13];
            BetaRange  = [13 31];
            Fs = 250;
            % Filtering
            DeltaFilt = designfilt('lowpassiir','FilterOrder',10, 'PassbandFrequency',DeltaRange(2) + .5,'SampleRate',Fs);
            ThetaFilt = designfilt('bandpassiir','FilterOrder',10, 'HalfPowerFrequency1',ThetaRange(1) - .5, ...
                'HalfPowerFrequency2',ThetaRange(2) + .5, 'SampleRate',Fs);
            AlphaFilt = designfilt('bandpassiir','FilterOrder',10, 'HalfPowerFrequency1',AlphaRange(1) - .5, ...
                'HalfPowerFrequency2',AlphaRange(2) + .5, 'SampleRate',Fs);
            BetaFilt  = designfilt('bandpassiir','FilterOrder',10, 'HalfPowerFrequency1',BetaRange(1) - .5, ...
                'HalfPowerFrequency2',BetaRange(2) + .5, 'SampleRate',Fs);
            % Feature Extraction
            Data1 = [];
            for trial = 1:numel(Trials)
                x = importdata([Trials(1).folder '\' Trials(1).name], Trials(1).name(1:3));
                x = ExtractEEGFeatures(x', Frame_Length, ...
                    DeltaFilt, DeltaRange, ThetaFilt, ThetaRange, AlphaFilt, AlphaRange, BetaFilt, BetaRange);
                Data1 = cat(2, Data1, x);
            end
            Trials = dir('..\Data\EEG\s*.mat');
            Data2 = [];
            for trial = 1:numel(Trials)
                x = importdata([Trials(1).folder '\' Trials(1).name], Trials(1).name(1:3));
                x = ExtractEEGFeatures(x', Frame_Length, ...
                    DeltaFilt, DeltaRange, ThetaFilt, ThetaRange, AlphaFilt, AlphaRange, BetaFilt, BetaRange);
                Data2 = cat(2, Data2, x);
            end
            NSamples = min(size(Data1, 2), size(Data2, 2));                       % Number of data samples in each class
            Data = cat(3, Data1(:, 1:NSamples), Data2(:, 1:NSamples));
            clear Data1 Data2 x;
        case 'DistNet'
            %             % Considering data a two-class (Healthy and Faulty) dataset
            %             HTrials = dir('..\Data\DistNet\H*.mat');
            %             FTrials = dir('..\Data\DistNet\F*.mat');
            %             N_Trials = min([length(HTrials), length(FTrials)]);
            %             HTrials = HTrials(randperm(length(HTrials),N_Trials));
            %             FTrials = FTrials(randperm(length(FTrials),N_Trials));
            %
            %             Data1 = [];
            %             Data2 = [];
            %             for trial = 1:N_Trials
            %                 load(['..\Data\DistNet\' HTrials(trial).name], 'Signals');
            %                 Data1 = cat(2, Data1, [Signals.va; Signals.vb; Signals.vc; Signals.ia; Signals.ib; Signals.ic]);
            %                 clear Signals
            %                 load(['..\Data\DistNet\' FTrials(trial).name], 'Signals');
            %                 Data2 = cat(2, Data2, [Signals.va; Signals.vb; Signals.vc; Signals.ia; Signals.ib; Signals.ic]);
            %                 clear Signals
            %             end
            %             Data = cat(3, Data1, Data2);
            %             clear Data1 Data2;

            % Considering data as a five-class (Healthy, SLG, LLG, LL, and LLL) dataset
            Trials = dir('..\Data\DistNet\*.mat');

            Data1 = [];
            Data2 = [];
            Data3 = [];
            Data4 = [];
            Data5 = [];
            for trial = 1:numel(Trials)
                load([Trials(trial).folder, '\',Trials(trial).name], 'Signals');
                if ~isempty(strfind(Trials(trial).name,'HLT'))
                    Data1 = cat(2, Data1, [Signals.va; Signals.vb; Signals.vc; Signals.ia; Signals.ib; Signals.ic]);
                elseif ~isempty(strfind(Trials(trial).name,'SLG'))
                    Data2 = cat(2, Data2, [Signals.va; Signals.vb; Signals.vc; Signals.ia; Signals.ib; Signals.ic]);
                elseif ~isempty(strfind(Trials(trial).name,'_LLG'))
                    Data3 = cat(2, Data3, [Signals.va; Signals.vb; Signals.vc; Signals.ia; Signals.ib; Signals.ic]);
                elseif ~isempty(strfind(Trials(trial).name,'LL('))
                    Data4 = cat(2, Data4, [Signals.va; Signals.vb; Signals.vc; Signals.ia; Signals.ib; Signals.ic]);
                elseif ~isempty(strfind(Trials(trial).name,'LLL'))
                    Data5 = cat(2, Data5, [Signals.va; Signals.vb; Signals.vc; Signals.ia; Signals.ib; Signals.ic]);
                end
            end
            N_Samples = min([size(Data1, 2) size(Data2, 2) size(Data3, 2) size(Data4, 2) size(Data5, 2)]);
            Data = cat(3, Data1(:, 1:N_Samples), Data2(:, 1:N_Samples), Data3(:, 1:N_Samples), Data4(:, 1:N_Samples), Data5(:, 1:N_Samples));
            clear Data1 Data2 Data3 Data4 Data5;
    end

    % Data parsing for train and test
    NClass = size(Data, 3);            % Number of classes
    NSamples = size(Data, 2);          % Number of data samples in each class
    Ntest = floor(NSamples/NFolds);    % Number of test samples in each class
    Ntrain = Ntest*(NFolds - 1);       % Number of train samples in each class
    NSamples = NFolds*Ntest;           % Total number of sample, after rounding

    Data = Data(:, 1:NSamples, :);     % Rounding the number of data samples in all classes
    CVO = cvpartition(NSamples, 'KFold', NFolds);     % Cross-Validation Object

    Target = repmat((1:NClass)', 1, Ntest);
    Acc = zeros(nofTests + 1, NMissing, NFolds);

    for f = 1:NFolds
        fprintf('Fold %i (of %i), %i batches of %i vectors, do batch %3i: ', ...
            f, NFolds, nofBatches, BatchSize, 0)
        if verbose   % else \b is used to reprint batch number 'b'
            fprintf('\n')
        end

        % Training and test data samples for each fold
        XTrain = zeros(size(Data, 1), Ntrain, NClass);
        XTest = zeros(size(Data, 1), Ntest, NClass);
        for c = 1:NClass
            XTrain(:, :, c) = Data(:, CVO.training(f), c);
            XTest(:, :, c) = Data(:, CVO.test(f), c);
        end

        % Initialization
        P_i = cell(NClass, 1);    % the profiles, one for each class
        % Idx = 1:Q;
        Idx = randperm(Ntrain,Q);    % random start may be as good as anything else
        for c = 1:NClass
            P_i{c} = ProfileInit( XTrain(:, Idx, c), ...
                's',s, 'KernelType',KernelType, 'KernelParam',KernelParam);
        end
        % Test the initialized dictionaries
        tic
        for m = 1:NMissing
            if ~verbose
                fprintf('m =%2i',m);
            end
            Prd = zeros(NClass, Ntest);

            for i = 1:NClass
                CurrXTest = XTest(:, :, i);
                Sig = zeros(1, Ntest);

                for j = 1:Ntest
                    MissingIdx = randperm(size(CurrXTest, 1), round((m - 1)*.1*size(CurrXTest, 1)));
                    CurrXTest(MissingIdx, j) = 0;
                    % Energy of the current sample in the feature space
                    Sig(j) = ker_eval(CurrXTest(:, j), CurrXTest(:, j), KernelType, KernelParam);
                end

                Err = zeros(NClass, Ntest);
                for c = 1:NClass
                    % Representing the current data using the cth dictionary
                    k = ker_eval(P_i{c}.X, CurrXTest, KernelType, KernelParam);
                    h = P_i{c}.U * k;
                    [~, r2] = myKOMP(P_i{c}.Psi, h, Sig, s, 1e-6, false, false);
                    Err(c, :) = r2(:)';
                end
                [~, Prd(i, :)] = min(Err);
            end

            Acc(1, m, f) = nnz(Target == Prd)/numel(Target);
            if verbose
                fprintf('\t\t\t\t Missing Pixels = %2.0f%%,   Accuracy = %4.2f%%\n', ...
                    100*(m - 1)*.1, 100*Acc(1, m, f))
            else
                fprintf('\b\b\b\b\b');
            end
        end
        ElapsedTime.Test(1, f) = ElapsedTime.Test(1, f) + toc;

        % Next batches
        % Training
        for b = 1:nofBatches
            if verbose
                fprintf('Fold %i (of %i), %i batches of %i vectors, do batch %3i: \n', ...
                    f, NFolds, nofBatches, BatchSize, b)
            else
                fprintf('\b\b\b\b\b%3i: ', b);   % print on top of previous, i.e. reprint
            end
            Idx = Idx(end) + (1:BatchSize);
            if Idx(end) > Ntrain
                Idx = rem(Idx-1,Ntrain)+1;
            end

            for c = 1:NClass
                if (P_i{c}.L + BatchSize) > maxL
                    % prune profile
                    tic
                    P_i{c} = ProfilePrune(P_i{c}, newL, verbose);
                    ElapsedTime.Prune(f) = ElapsedTime.Prune(f) + toc;
                end
                % grow profile
                tic
                P_i{c} = ProfileGrow(P_i{c}, XTrain(:, Idx, c), LambdaBatch(b), GrowThresh, GrowCriterion, verbose);
                ElapsedTime.Grow(f) = ElapsedTime.Grow(f) + toc;
            end

            % Test after some batches
            if ismember(b,testAfterBatch)  % or when wanted
                tic
                for m = 1:NMissing
                    if ~verbose
                        fprintf('m =%2i',m);
                    end
                    Prd = zeros(NClass, Ntest);

                    for i = 1:NClass
                        CurrXTest = XTest(:, :, i);
                        Sig = zeros(1, Ntest);

                        for j = 1:Ntest
                            MissingIdx = randperm(size(CurrXTest, 1), round((m - 1)*.1*size(CurrXTest, 1)));
                            CurrXTest(MissingIdx, j) = 0;
                            % Energy of the current sample in the feature space
                            Sig(j) = ker_eval(CurrXTest(:, j), CurrXTest(:, j), KernelType, KernelParam);
                        end

                        Err = zeros(NClass, Ntest);
                        for c = 1:NClass
                            % Representing the current data using the cth dictionary
                            k = ker_eval(P_i{c}.X, CurrXTest, KernelType, KernelParam);
                            h = P_i{c}.U * k;
                            [~, r2] = myKOMP(P_i{c}.Psi, h, Sig, s, 1e-6, false, false);
                            Err(c, :) = r2(:)';
                        end
                        [~, Prd(i, :)] = min(Err);
                    end

                    bb = find(testAfterBatch==b);
                    Acc(bb + 1, m, f) = nnz(Target == Prd)/numel(Target);
                    if verbose
                        fprintf('\t\t\t\t Missing Pixels = %2.0f%%,   Accuracy = %4.2f%%\n', ...
                            100*(m - 1)*.1, 100*Acc(bb, m, f))
                    else
                        fprintf('\b\b\b\b\b');
                    end
                end
                ElapsedTime.Test(bb + 1, f) = ElapsedTime.Test(bb + 1, f) + toc;
            end % test
        end  % batch
        if ~verbose
            fprintf('\n');
        end
    end % fold
    %
    res = struct('Acc', Acc, 'ElapsedTime', ElapsedTime, 'Options', Options);
    save(resFileName, 'res')
end

% Averaging and plotting the results
Acc = mean(Acc, 3);
if verbose
    nofTests = size(Acc,1) - 1;
    if (length(testAfterBatch) ~= nofTests)
        testAfterBatch = 1:nofTests;
    end
    figure('color','w');   clf;
    grid on; hold on
    for m = 1:NMissing
        plot([0 testAfterBatch], Acc(:, m), ...
            'LineWidth', 2, 'DisplayName', sprintf('Missing Pixels: %2.0f%%', 100*(m - 1)*.1))
    end
    ax = gca;
    ax.FontSize = 14;
    ax.PositionConstraint = 'innerposition';
    ylabel('Classification Accuracy', 'FontSize', 16)
    xlabel(sprintf('Number of Mini-Batches each %i samples',BatchSize), 'FontSize', 16)
    legend('FontSize', 14, 'Location','southeast')
    %     title(sprintf('KRLS test on %s dataset',DataSet), 'FontSize',16)
    %     V = axis();
    %     V(3) = max( 0.8, V(3) );
    % v(4) = 0.98;
    %     axis(V);

    fprintf('KRLS-DLA test of %s data, Q=%i, s=%i, Gamma=%.4f.\n', DataSet, Q, s, Gamma)
    fprintf('Process %i training vectors in %i batches of %i training vectors.\n', ...
        nofBatches*BatchSize, nofBatches, BatchSize )
    for m = 1:NMissing
        fprintf('Accuracy (first,last,best) for Missing Pixels: %2.0f%% : (%4.2f%%, %4.2f%%, %4.2f%%)\n', ...
            100*(m - 1)*.1, 100*Acc(1, m), 100*Acc(end, m), 100*max(Acc(:,m)) );
    end
    t = sum(sum(ElapsedTime.Test));
    fprintf('Total time testing:  %3i minutes %4.1f seconds.\n', floor(t/60), rem(t,60));
    t = sum(sum(ElapsedTime.Grow)) + sum(sum(ElapsedTime.Prune));
    fprintf('Total time training: %3i minutes %4.1f seconds.\n', floor(t/60), rem(t,60));
    t = sum(sum(ElapsedTime.Grow));
    fprintf('            growing: %3i minutes %4.1f seconds.\n', floor(t/60), rem(t,60));
    t = sum(sum(ElapsedTime.Prune));
    fprintf('            pruning: %3i minutes %4.1f seconds.\n', floor(t/60), rem(t,60));
end