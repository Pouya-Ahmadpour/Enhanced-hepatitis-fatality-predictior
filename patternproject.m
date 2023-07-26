clear all
close all
clc

    
        
%%
%data loader
dataset = importdata('hepatitis.data');
data1 = dataset.textdata;
data2 = str2double(data1(:,:));
data2 = data2(randperm(size(data2, 1)), :);
labels = data2(:,1);
features = data2(:,2:end);

data =[features,labels];


%% Mean tester

%for iter=1:100

%%
% data split


sorteddata = sortrows(data,19);
liverow = find(sorteddata(:,19)==1);
fatalrow = find(sorteddata(:,19)==2);

liveset = sorteddata(liverow,:)
fatalset = sorteddata(fatalrow,:);

liveset = liveset(randperm(size(liveset, 1)), :);
fatalset = fatalset(randperm(size(fatalset, 1)), :);


training_live = liveset(floor(1:3/4*size(liveset,1)),:);
test_live = liveset(floor(3/4*size(liveset,1))+1:end,:);

training_fatal = fatalset(floor(1:3/4*size(fatalset,1)),:);
test_fatal = fatalset(floor(3/4*size(fatalset,1))+1:end,:);

train_data = [training_fatal;training_live];
test_data = [test_fatal;test_live];

train_data = train_data(randperm(size(train_data, 1)), :);
test_data = test_data(randperm(size(test_data, 1)), :);



xtrain = train_data(:,1:end-1);
ytrain = train_data(:,end);

xtest = test_data(:,1:end-1);
ytest = test_data(:,end);

fotApp = [xtrain,ytrain];



%%
% replace Nan with median
nansetTrain = isnan(xtrain);
featurespp = xtrain;
[b,c] = find(nansetTrain==1);
m = nanmedian(xtrain);
for i = 1:length(b)
    featurespp(b(i),c(i)) = m(c(i));
end

xtrain = featurespp;

% doing the same on test data
nansetTest = isnan(xtest);
featurespt = xtest;
[e,r] = find(nansetTest==1);
for it = 1:length(e)
    featurespt(e(it),r(it)) = m(c(it));
end

xtest = featurespt;
%%
% feature selection based on correlation (no correlation is found between the dataset)

 %correlationMatrix = corrcoef(xtrain);
 %corre = find(correlationMatrix > 0.85 & correlationMatrix ~= 1)
 %xtrain(:,corre) = []

%%
% Z-score normalization
%

xtrainN = zscore(xtrain);
mu = mean(xtrain);
STD = std(xtrain);


xtrain = xtrainN;

% doing the same on test set

for mi=1:size(xtest,2)
    xtestN(:,mi)=(xtest(:,mi)-mu(mi))/STD(mi);
end
xtest = xtestN;
%}



%%
%{
  % outlier replacement in training set


  outtersTrain = isoutlier( xtrain , 'mean' );
  [bb cc] = find(outtersTrain==1);
   mm = nanmean(xtrain);
for ii2 = 1:length(bb)
    xtrain(bb(ii2),cc(ii2)) = mm(cc(ii2));
end

% 

%}

%%
% NaN deletion (not used due to data limition)

%rowsnan = isnan(features);
%ln = find(rowsnan==1);
%features(ln,:) = [];


%%
% data normalization (giving the same results as Z-score)
%features = mapminmax(features',0,1);
%featuresN = features';

    
    

%%


%feature selection based on ranking

%{
% 'ttest' , 'entropy','bhattacharyya','wilcoxon','roc'
[idx,z] = rankfeatures(xtrain',ytrain,'criterion','bhattacharyya');
myfig1 = figure(1)
set(myfig1 , 'name' , 'feature ranking' , 'numbertitle' , 'off');
bar(z)
title('rank features   (Bhattacharyya)');
%rf1 = idx(18);
%rf2 = idx(17);
 %xtrain(:,[rf1]) = [];
 %xtest(:,[rf1]) = [];
  %}



%% PCA feature selection
%
[coeff,scoreTrain,~,~,explained,mu] = pca(xtrain);
%This code returns four outputs: coeff, scoreTrain, explained, and mu. Use explained (percentage of total variance explained) to find the number of components required to explain at least 95% variability. Use coeff (principal component coefficients) and mu (estimated means of XTrain) to apply the PCA to a test data set. Use scoreTrain (principal component scores) instead of XTrain when you train a model.

%Display the percent variability explained by the principal components.

%explained

idx = find(cumsum(explained)>99,1);

% Train a classification tree using the first two components.
scoretrain = scoreTrain(:,1:idx);


% To use the trained model for the test set, you need to transform the test data set by using the PCA obtained from the training data set. Obtain the principal component scores of the test data set by subtracting mu from XTest and multiplying by coeff. Only the scores for the first two components are necessary, so use the first two coefficients coeff(:,1:idx).

scoretest = (xtest-mu)*coeff(:,1:idx);
% Pass the trained model mdl and the transformed test data set scoreTest to the predict function to predict ratings for the test set.


xtrain = scoretrain;
xtest = scoretest;


%}



%%
% bayes classification (not working because of variant data distribution)
%
%{
bayseModel = fitcnb(xtrain,ytrain);
Lpredicted = predict(bayseModel,xtest);
n = sum(Lpredicted==ytest);
accuracyNB = n/length(xtest)*100
%}

  %%
  %{
  % knn classification with no validation

  %uncomment these two down bellow if using z-score
  
%knnModel = fitcknn(xtrain,ytrain,'NumNeighbors',3,'distance','euclidean');  % 'numneighbor',3,'distance','minkowski');
%knnModel = fitcknn(xtrain,ytrain,'NumNeighbors',3,'distance','cityblock');
%knnModel = fitcknn(xtrain,ytrain,'NumNeighbors',3,'distance','chebychev');
%knnModel = fitcknn(xtrain,ytrain,'NumNeighbors',15,'distance','cosine');
%knnModel = fitcknn(xtrain,ytrain,'NumNeighbors',5,'distance','minkowski');
knnModel = fitcknn(xtrain,ytrain,'NumNeighbors',23,'distance','correlation');
class = predict(knnModel,xtest);
n = sum(class==ytest);
accuracyKNN = n/length(xtest)*100;
accu(iter,1) = accuracyKNN;
%}
  
  %%
  %% training model & validation
% K-fold
%
K = 20;
n_run = 3;
accuracy = zeros(K,n_run);
% 10_fold
for i_run=1:n_run
    indices = crossvalind('Kfold',ytrain,K);
    
    for i_fold = 1:K
        Val = indices==i_fold;
        train = ~Val;
        featureTrain = xtrain(train,:);
        featureVal = xtrain(Val,:);
        
        %% Classification with KNN
        Model = fitcknn(featureTrain,ytrain(train),'NumNeighbors',23,'distance','correlation','ClassNames', [2,1]);
        class = predict(Model, featureVal);
        %accuracy(i_fold,i_run) = 100*length(find(class == ytrain(Val)))/length(ytrain(Val));
        
        %tree decision on validation
        
       % modeltree = fitctree(featureTrain,ytrain(train),'Classnames' , [2,1]);
        %preclass = predict(modeltree, featureVal);
        
        % Linear on Validation
       % Modlinear = fitclinear(featureTrain,ytrain(train), 'Classnames' , [2,1]);
        %classl = predict(Model, featureVal);
        
        
    end    
        %disp(['n_run = ',num2str(i_run),', Accuracy = ',num2str(mean(accuracy(:,i_run))),' ± ',num2str(std(accuracy(:,i_run)))])
  
end
%disp(['Total Accuracy of KNN = ',num2str(mean(accuracy(:))),' ',native2unicode(177),' ',num2str(std(accuracy(:)))]);



%}



%%
%
%  KNN test after validation
y_predict = predict(Model, xtest);
n = sum(y_predict==ytest);
accuracyofKNN = n/length(ytest)*100
%accu(iter,1) = accuracyofKNN;



%% Confusion matrix
%
ytestc = ytest;
y_predictc = y_predict;
yt(:) = find(ytestc==2);
pt(:) = find(y_predictc==2);
ytestc(yt,:) = 0;
y_predictc(pt,:) = 0;

%
figure(3)
plotconfusion(ytestc',y_predictc')
title('confusion matrix')
%}
%% ROC curve and AUC

%
ytestc = ytest;
y_predictc = y_predict;
yt(:) = find(ytestc==2);
pt(:) = find(y_predictc==2);
ytestc(yt,:) = 0;
y_predictc(pt,:) = 0;
[x4,y4, ~,aucKNN] = perfcurve(ytestc',y_predictc,1);
xf=[0,1];
yf=xf;
AUCKNN=aucKNN*100
%%
figure
plot(x4,y4)
hold on
plot(xf,yf,'k--.')
xlabel('False Positive Rate');
ylabel('True Positive Rate');
legend(['KNN (AUC=',num2str(AUCKNN),')'])
title(['ROC KNN'])

%}

%% linear classifier after Validation test
%{
prediction = predict(Modlinear, xtest);
nn = sum(prediction==ytest);
accuracyOfLINEAR = nn/length(ytest)*100;
accu5(iter,1) = accuracyOfLINEAR;
%}

%% Tree test after validation
%{
tree_pre = predict(modeltree, xtest);
n2 = sum(tree_pre==ytest);
accuracyofTree = n2/length(ytest)*100;


%accu2(iter,1) = accuracyofTree;
%}


%%
% SVM classifier
%
%train                      'linear' , 'rbf'
%classtrain = fitcsvm(xtrain,ytrain,'KernelFunction','polynomial','ClassNames',[2,1]);
classtrain = fitcsvm(xtrain,ytrain,'Kernelscale',4.25,'BoxConstraint',0.42,'ClassNames',[2,1]);
%classtrain = fitcsvm(xtrain,ytrain,'OptimizeHyperparameters','auto','ClassNames',[2,1]);
%test
accuracyOfSVM = sum(predict(classtrain,xtest) == ytest)/length(ytest)*100

%accu1(iter,1) = accuracyOfSVM;



%%
% SVM k-fold cross validation
%{
K2 = 10;
n_run2 = 3;
accuracy2 = zeros(K2,n_run2);
% 10_fold
for i_run2=1:n_run2
    myindices = crossvalind('Kfold',ytrain,K2);
    
    for i_fold2 = 1:K2
        myVal = myindices==i_fold2;
        train = ~myVal;
        featureTrain2 = xtrain(train,:);
        featureVal2 = xtrain(myVal,:);
        
        %% Classification using SVM
        classtrain = fitcsvm(featureTrain2,ytrain(train),'KernelFunction','rbf',...
            'BoxConstraint',Inf,'ClassNames',[2,1]); 
        classS = predict(classtrain, featureVal2);
       % accuracy2(i_fold2,i_run2) = sum(predict(classtrain,xtest) == ytest)/length(ytest)*100;
        
    end    
     
  
end
%disp(['Total Accuracy of SVM = ',num2str(mean(accuracy2(:))),' ',native2unicode(177),' ',num2str(std(accuracy2(:)))]);

%% Test of SVM
y_predict2 = predict(classtrain, xtest);
n = sum(y_predict2==ytest);
accuracySV = n/length(ytest)*100;
%accu1(iter,1) = accuracySV;

%%
%accu1(iter,1) = accuracyOfSVM;
%accu1(iter,1) = mean(accuracy2(:));
%}

%%  Tree classifier
%
classtree = fitctree(xtrain,ytrain,'MinLeafSize',13,'Classnames' , [2,1]);


accuracyofTree = sum(predict(classtree,xtest) == ytest)/length(ytest)*100

%accu2(iter,1) = accuracyofTree;
%}


%% lnear Classification
trainedlog = fitclinear(xtrain,ytrain,'Learner','logistic','Lambda',0.070583, 'Classnames' , [2,1]);
accuracyOfLogistocRegression = sum(predict(trainedlog,xtest) == ytest)/length(ytest)*100
%accu3(iter,1) = accuracyOfLinear;


%%
%{
end
MeanAccuracyKNN = mean(accu)
%
meanAccuracyOfSVM = mean(accu1)
MeanAccuracyKNN = mean(accu)
meanAccuracyTree = mean(accu2)
meanAccuracyLogistic = mean(accu3)

%}



