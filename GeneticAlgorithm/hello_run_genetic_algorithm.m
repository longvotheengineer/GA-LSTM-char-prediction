clc; clear; close all;

finalGoal = 'hello';       
currentString = 'h';       
fprintf('================================================\n');
fprintf('Roadmap Target: "%s"\n', finalGoal);
fprintf('Current State : "%s"\n', currentString);
fprintf('------------------------------------------------\n');

while length(currentString) < length(finalGoal)
    
    nextIndex = length(currentString) + 1;
    expectedChar = finalGoal(nextIndex);
    
    isValidInput = false;
    while ~isValidInput
        prompt = sprintf('\n[Next Step] Please enter "%s": ', expectedChar);
        userInput = input(prompt, 's');
        
        if strcmpi(userInput, 'exit')
            fprintf('Program terminated by user.\n');
            return;
        end
        
        if isempty(userInput)
            fprintf('   -> [ERROR] Input cannot be empty.\n');
            continue;
        end
        
        targetChar = userInput(1);
        
        if targetChar ~= expectedChar
            fprintf('   -> [WARNING] Deviation detected! You entered "%s", but the roadmap expects "%s".\n', ...
                targetChar, expectedChar);
            
            choice = input('   -> Do you want to proceed anyway? (y/n): ', 's');
            if strcmpi(choice, 'y')
                isValidInput = true; 
                fprintf('   -> [SYSTEM] Proceeding with deviation...\n');
            else
                fprintf('   -> [SYSTEM] Please retry.\n');      
            end
        else
            isValidInput = true; 
        end
    end

    fprintf('   -> Initializing GA to find "%s"...\n', targetChar);
    
    nVars = 1;
    LB = 32; UB = 126;
    FitnessFcn = @(x) abs(x - double(targetChar));
    
    options = optimoptions('ga', ...
        'PopulationSize', 50, ...
        'MaxGenerations', 50, ...
        'Display', 'off');
    
    [x_best, ~] = ga(FitnessFcn, nVars, [], [], [], [], LB, UB, [], 1, options);
    
    foundChar = char(x_best);
    currentString = [currentString, foundChar];
    
    fprintf('   -> String Updated: "%s"\n', currentString);
end

fprintf('================================================\n');
fprintf('Final String: %s\n', currentString);
fprintf('================================================\n');