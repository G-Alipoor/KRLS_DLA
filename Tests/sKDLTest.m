function res = sKDLTest(varargin)

% This routine tests the online kernel dictionary learning algorithm,
%     proposed in the following paper, for kernel sparse representation classification.
%     C. O�Brien and M. D. Plumbley, "Sparse kernel dictionary learning," 2016
%     Proceedings of the 11th IMA International Conference on Mathematics in Signal Processing.
%
% This algorithm is used in our paper to compare with the proposed KRLS-DLA algorithm.
% 
% Examples:
%   res = sKDLTest('BatchSize',10, 'NMissing',4, 'nofTests',10);
%   res = sKDLTest('NMissing',5, 'nofTests',10);
%   res = sKDLTest('NMissing',2, 'nofTests',25, 'NFolds',2, 'KernelType','Poly', 'KernelParam',1.0, 'verbose',0);
%   res = sKDLTest();
%   res = sKDLTest('KernelType','Poly', 'KernelParam',1.0);
%
% Generated by Ghasem Alipoor (alipoor@hut.ac.ir) and Karl Skretting (karl.skretting@uis.no)
% Last modification: 3 August 2022
%

%% Parameter setting
pnames = {'DataSet', ...  % Either USPS or ISOLET
    'KernelType', ...          % 'Gauss', 'Poly' or 'Lin'
    'KernelParam', ...        % 1.0
    'Q', ...                         % Number of dictionary atoms
    's', ...                          % Sparsity level, in training and test
    'GrowingTrsh', ...         % Growing threshold
    'LearningRate_Rep', ...	% Learning Rate used in the Iterative Shrinkage-Thresholding Algorithm (ISTA)
    'LearningRate_Dic', ...  % Learning Rate used in the Iterative Shrinkage-Thresholding Algorithm (ISTA)
    'ISTA_Thrsh', ...            % Convergence threshold used in the STA
    'lam_Rep', ...                % Sparsity parameter for the representation coefficients term, lamda_1 in the paper
    'lam_Dic', ...                 % Sparsity parameter for the dictionary sparsity term, lamda_2 in the paper
    'BatchSize', ...              % Number of  vectors for batch when growing, ex 10;
    'nofBatches', ...            % may use training data more than once, ex 1200.
    'maxL', ...                    % max value for profile L, ex 400
    'newL', ...                    % Profile L after pruning, ex 300
    'NMissing', ...               % Number of states for mission indice ratios, was 10
    'NFolds', ...                   % Number of folds in kfold cross-validation
    'extraText', ...                % A short text used in file name
    'nofTests', ...                 % Number of tests to do as batches are processed: Takes TIME.
    'Comment', ...               % Any comment to be saved with the test result
    'verbose', ...                  % 1 (default) to show warnings
    };
dflts = {'USPS', 'Poly', 2.0, 30, 5 , ...% DataSet, KernelType, KernelParam, Q, s
    0.9, 1e-5, 1e-5, ... % GrowingTrsh, LearningRate_Rep, LearningRate_Dic
    0.01, 0.01, 0.01, ... % ISTA_Thrsh, lam_Rep, lam_Dic
    10, 150, ... % BatchSize, nofBatches
    200, 190, ... % maxL, newL
    5, 5, '', ... % NMissing, NFolds, extraText
    20, '', 1, ... % nofTests, Comment, verbose
    };
[DataSet, KernelType, KernelParam, Q, s, GrowingTrsh, LearningRate_Rep, LearningRate_Dic, ...
    ISTA_Thrsh, lam_Rep, lam_Dic, BatchSize, nofBatches, maxL, newL, NMissing, NFolds, ...
    extraText, nofTests, Comment, verbose] = getarg(varargin,pnames,dflts); %#ok<ASGLU>

% Test version; a string attached to the name of the saved file
TestVersion = sprintf('%s%iQ%is%ib%im%i%s',KernelType,KernelParam,Q,s,BatchSize,NMissing,extraText);
resFileName = fullfile('..\Results\', sprintf('sKDL_%s_%s', DataSet, TestVersion));

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
Options.testAfterBatch = testAfterBatch;

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
end
% [N,tL] = size(XTrain);

%% Data parsing for train and test
NClass = size(Data, 3);            % Number of classes
NSamples = size(Data, 2);          % Number of data samples in each class
Ntest = floor(NSamples/NFolds);    % Number of test samples in each class
Ntrain = Ntest*(NFolds - 1);       % Number of train samples in each class
NSamples = NFolds*Ntest;           % Total number of sample, after rounding

Data = Data(:, 1:NSamples, :);     % Rounding the number of data samples in all classes
CVO = cvpartition(NSamples, 'KFold', NFolds);     % Cross-Validation Object

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
    Target = repmat((1:NClass)', 1, Ntest);
    Acc = zeros(nofTests + 1, NMissing, NFolds);
    ElapsedTime = struct('Train',zeros(1, NFolds), 'Test',zeros(nofTests + 1, NFolds));
    
    for f = 1:NFolds
        fprintf('Fold %i (of %i), %i batches of %i vectors, do batch %3i: ', ...
            f, NFolds, nofBatches, BatchSize, 0)
        if verbose   % else \b is used to reprint batch number 'b'
            fprintf('\n')
        end
        
        % Data parsing between Train and Test sets
        XTrain = zeros(size(Data, 1), Ntrain, NClass);
        XTest = zeros(size(Data, 1), Ntest, NClass);
        for c = 1:NClass
            XTrain(:, :, c) = Data(:, CVO.training(f), c);
            XTest(:, :, c) = Data(:, CVO.test(f), c);
        end
        
        % Initialization
        P_i = cell(NClass, 1);    % the profiles, one for each class
        for c = 1:NClass
            P_i{c}.KernelType = KernelType;
            P_i{c}.KernelParam = KernelParam;
            P_i{c}.Q = Q;
            %             Idx = randperm(Ntrain, Q);    % random start may be as good as anything else
            Idx = 1:Q;
            P_i{c}.X = XTrain(:, Idx, c);
            P_i{c}.K = ker_eval(P_i{c}.X, P_i{c}.X, KernelType, KernelParam);
            P_i{c}.W = eye(Q);
            %             P_i{c}.A = zeros(Q, Q);
            %             P_i{c}.B = 0;
            P_i{c}.C = eye(Q);
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
                    h = P_i{c}.C' * k;
                    [~, r2] = myKOMP(P_i{c}.C'*P_i{c}.K*P_i{c}.C, h, Sig, s, 1e-6, false, false);
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
                tic
                P_i{c} = sKDLUpdate(XTrain(:, Idx, c), P_i{c}, ...
                    'GrowingTrsh',GrowingTrsh, 'maxL',maxL, 'newL', newL, 'LearningRate_Rep',LearningRate_Rep, ...
                    'LearningRate_Dic', LearningRate_Dic, 'ISTA_Thrsh',ISTA_Thrsh, 'lam_Rep',lam_Rep, ...
                    'lam_Dic',lam_Dic);
                ElapsedTime.Train(f) = ElapsedTime.Train(f) + toc;
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
                            h = P_i{c}.C' * k;
                            [~, r2] = myKOMP(P_i{c}.C'*P_i{c}.K*P_i{c}.C, h, Sig, s, 1e-6, false, false);
                            Err(c, :) = r2(:)';
                        end
                        [~, Prd(i, :)] = min(Err);
                    end
                    
                    bb = find(testAfterBatch==b);
                    Acc(bb + 1, m, f) = nnz(Target == Prd)/numel(Target);
                    if verbose
                        fprintf('\t\t\t\t Missing Pixels = %2.0f%%,   Accuracy = %4.2f%%\n', ...
                            100*(m - 1)*.1, 100*Acc(bb + 1, m, f))
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
    figure('color', 'w');   clf;
    grid on; hold on
    for m = 1:NMissing
        plot([0 testAfterBatch], Acc(:, m), ...
            'LineWidth', 2, 'DisplayName', sprintf('Missing Pixels: %2.0f%%', 100*(m - 1)*.1))
    end
    ylabel('Classification Accuracy', 'FontSize', 14)
    xlabel(sprintf('Number of Mini-Batches each %i samples',BatchSize), 'FontSize', 14)
    legend('FontSize', 14, 'Location','southeast')
    title(sprintf('sKDL test on %s dataset',DataSet), 'FontSize',16)
    V = axis();
    V(3) = max( 0.8, V(3) );
    % v(4) = 0.98;
    axis(V);
    
    fprintf('sKDL-DLA test of %s data, Q=%i, s=%i.\n', DataSet, Q, s)
    fprintf('Process %i (of %i) training vectors in %i batches of %i training vectors.\n', ...
        nofBatches*BatchSize, Ntrain, nofBatches, BatchSize )
    for m = 1:NMissing
        fprintf('Accuracy (first,last,best) for Missing Pixels: %2.0f%% : (%4.2f%%, %4.2f%%, %4.2f%%)\n', ...
            100*(m - 1)*.1, 100*Acc(1, m), 100*Acc(end, m), 100*max(Acc(:,m)) );
    end
    t = sum(sum(ElapsedTime.Test));
    fprintf('Total time testing:  %3i minutes %4.1f seconds.\n', floor(t/60), rem(t,60));
    t = sum(sum(ElapsedTime.Train));
    fprintf('Total time training: %3i minutes %4.1f seconds.\n', floor(t/60), rem(t,60));
end
