clc; clear; close all;

prompt = 'Enter the suggested character (h/g/w): ';
userInput = input(prompt, 's'); 

switch lower(userInput)
    case 'h'
        targetStr = 'hello';
    case 'g'
        targetStr = 'green';
    case 'w'
        targetStr = 'white';
    otherwise
        error('Just enter h, g or w!');
end

fprintf('Target String: "%s"\n', targetStr);

nVars = length(targetStr); 
LB = ones(1, nVars) * 32;  
UB = ones(1, nVars) * 126; 

FitnessFcn = @(x) find_text_fitness(x, targetStr);

options = optimoptions('ga', ...
    'PopulationSize', 200, ...     
    'MaxGenerations', 1000, ...     
    'Display', 'iter', ...          
    'PlotFcn', @gaplotbestf, ...   
    'CrossoverFraction', 0.8, ...  
    'MutationFcn', @mutationadaptfeasible); 

IntCon = 1:nVars; 

[x_best, fval] = ga(FitnessFcn, nVars, [], [], [], [], LB, UB, [], IntCon, options);

finalString = char(x_best); 
fprintf('\n---------------------------------\n');
fprintf('Final Result: %s\n', finalString);
fprintf('Final Fitness Value: %d\n', fval);
fprintf('---------------------------------\n');