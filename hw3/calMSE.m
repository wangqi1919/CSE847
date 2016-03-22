function [MSE]=calMSE(prediction, truevalue)
MSE=(prediction-truevalue)'*(prediction-truevalue)/length(prediction);
end
