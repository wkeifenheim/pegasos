%%
% simple binary classification
function [yhat] = PegasosPredict(weights, Xtest)

%% vectorizing this calculation improves run-time by 94%
predictions = Xtest*weights;
predictions(predictions < 0) = -1;
predictions(predictions >= 0) = 1;

yhat = predictions;






