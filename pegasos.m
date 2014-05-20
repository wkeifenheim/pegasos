%%
% X: feature vector
% Y: true classification vector
% lambda: regularization parameter
% T: number of iterations to perform
% k: size of subset of the full training set

function [trainedWeights] = pegasos(X, Y, lambda, T, k)

    weights = zeros(size(X,2),1);
    weights(:) = (1 / sqrt(lambda * size(weights,1)));

    wait_h = waitbar(0,'progress of training SVM weights');
    for t = 1:T

        [subsetX, subsetIdx] = datasample(X,k,'Replace',false);
        subsetY = Y(subsetIdx,:);

        %determine the prediction for each sample, given the current set of weights
        predictions = PegasosPredict(weights, subsetX);

        eta = 1 / (lambda * t); %learning rate changes based on the iteration
        
        % move forward only with instances for which the initial weights
        % gave incorrect classifications, called set A+ in the paper
        incorrect = predictions ~= subsetY;
        subsetX = subsetX(incorrect,:);
        subsetY = subsetY(incorrect,:);
        for i = 1 : size(subsetX,2)
            subsetX(:,i) = subsetX(:,i) .* subsetY;
        end
        sumOverIncorrectPredicitons = sum(subsetX)';

        updatedWeights = (1 - eta*lambda) * weights + (eta / k) * sumOverIncorrectPredicitons;
        weights = min(1, (1/sqrt(lambda)) / norm(updatedWeights, 2)) * updatedWeights;

        waitbar(t/T);
    end
    delete(wait_h);
    trainedWeights = weights;
    toc
end
