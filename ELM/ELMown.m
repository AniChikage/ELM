function [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = ELM(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)

% Usage: elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
% OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
%
% Input:
% TrainingData_File     - Filename of training data set
% TestingData_File      - Filename of testing data set
% Elm_Type              - 0 for regression; 1 for (both binary and multi-classes) classification
% NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
% ActivationFunction    - Type of activation function:
%                           'sig' for Sigmoidal function
%                           'sin' for Sine function
%                           'hardlim' for Hardlim function
%                           'tribas' for Triangular basis function
%                           'radbas' for Radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
%
% Output: 
% TrainingTime          - Time (seconds) spent on training ELM
% TestingTime           - Time (seconds) spent on predicting ALL testing data
% TrainingAccuracy      - Training accuracy: 
%                           RMSE for regression or correct classification rate for classification
% TestingAccuracy       - Testing accuracy: 
%                           RMSE for regression or correct classification rate for classification
%
% MULTI-CLASSE CLASSIFICATION: NUMBER OF OUTPUT NEURONS WILL BE AUTOMATICALLY SET EQUAL TO NUMBER OF CLASSES
% FOR EXAMPLE, if there are 7 classes in all, there will have 7 output
% neurons; neuron 5 has the highest output means input belongs to 5-th class
%
% Sample1 regression: [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm('sinc_train', 'sinc_test', 0, 20, 'sig')
% Sample2 classification: elm('diabetes_train', 'diabetes_test', 1, 20, 'sig')
%
    %%%%    Authors:    MR QIN-YU ZHU AND DR GUANG-BIN HUANG
    %%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
    %%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
    %%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
    %%%%    DATE:       APRIL 2004

%%%%%%%%%%% Macro definition
REGRESSION=0;
CLASSIFIER=1;

% 加载训练数据
train_data=load(TrainingData_File);
T=train_data(:,1)';
P=train_data(:,2:size(train_data,2))';
clear train_data;                                 

% 加载测试数据
test_data=load(TestingData_File);
TV.T=test_data(:,1)';
TV.P=test_data(:,2:size(test_data,2))';
clear test_data;                                  

NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);

if Elm_Type~=REGRESSION
    % 初始化分类数据
    sorted_target=sort(cat(2,T,TV.T),2);
    label=zeros(1,1);     %  查找和保存在label类中的训练数据和分类数据
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTrainingData+NumberofTestingData)
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    NumberofOutputNeurons=number_class;
       
    % 预处理训练目标
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;
    end
    T=temp_T*2-1;

    %预处理测试目标 
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == TV.T(1,i)
                break; 
            end
        end
        temp_TV_T(j,i)=1;
    end
    TV.T=temp_TV_T*2-1;

end                                  

%计算权值和偏差
start_time_train=cputime;

%随机生成输入权值和基本隐层节点 
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
tempH=InputWeight*P;
clear P;                               %   释放输入训练数据
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);        
tempH=tempH+BiasMatrix;

%计算隐层网络输出节点矩阵H 
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %激励函数Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %激励函数Sine
        H = sin(tempH);    
    case {'hardlim'}
        %激励函数Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %激励函数Triangular basis function
        H = tribas(tempH);
    case {'radbas'}
        %激励函数Radial basis function
        H = radbas(tempH);               
end
clear tempH;                                    

%计算输出权值 
OutputWeight=pinv(H') * T';        
end_time_train=cputime;
TrainingTime=end_time_train-start_time_train    

%计算训练精度
Y=(H' * OutputWeight)';                 % Y：实际输出训练数据
if Elm_Type == REGRESSION
    TrainingAccuracy=sqrt(mse(T - Y))   % 计算回归案例的训练精度
end
clear H;

%计算输入测试数据的输出 
start_time_test=cputime;
tempH_test=InputWeight*TV.P;
clear TV.P;                      
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);            
tempH_test=tempH_test + BiasMatrix;
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        H_test = sin(tempH_test);        
    case {'hardlim'}
        H_test = hardlim(tempH_test);        
    case {'tribas'}
        H_test = tribas(tempH_test);        
    case {'radbas'}
        H_test = radbas(tempH_test);              
end
TY=(H_test' * OutputWeight)';       %   TY: 实际的测试输出
end_time_test=cputime;
TestingTime=end_time_test-start_time_test       

if Elm_Type == REGRESSION
    TestingAccuracy=sqrt(mse(TV.T - TY))         
end

if Elm_Type == CLASSIFIER
%计算训练和测试的分类精度 
    MissClassificationRate_Training=0;
    MissClassificationRate_Testing=0;

    for i = 1 : size(T, 2)
        [x, label_index_expected]=max(T(:,i));
        [x, label_index_actual]=max(Y(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Training=MissClassificationRate_Training+1;
        end
    end
    TrainingAccuracy=1-MissClassificationRate_Training/size(T,2)
    for i = 1 : size(TV.T, 2)
        [x, label_index_expected]=max(TV.T(:,i));
        [x, label_index_actual]=max(TY(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
    end
    TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2)  
end