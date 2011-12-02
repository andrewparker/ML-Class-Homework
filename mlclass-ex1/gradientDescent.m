function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	temptheta = zeros(length(theta), 1);
    
	for k = 1:length(theta)
	    sum = 0;
		for h = 1:m
		    sum = sum + (theta(1) + theta(2) * X(h,2) - y(h))*X(h,k);
	    end
        temptheta(k) = theta(k) - alpha*(1/m)*sum;
    end
    for k = 1:length(theta)
	    theta(k) = temptheta(k);
	end

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
