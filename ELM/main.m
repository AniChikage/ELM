clear all
clc
load train_data.txt
load test_data.txt

[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy,TY,TV] = ......
ELM('train_data.txt', 'test_data.txt', 1, 80, 'sig');

%figure;
%plot(TV.T);
%hold on;
%plot(TY(1,:),'r');
%hold on;
%plot(TY,'r');
%hold off;