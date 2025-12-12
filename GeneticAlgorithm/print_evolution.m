function [state, options, optchanged] = print_evolution(options, state, flag)
    optchanged = false;
    switch flag
        case 'iter' 
            [bestScore, idx] = min(state.Score);
            bestIndividual = state.Population(idx, :);
            
            currentString = char(bestIndividual);
            
            fprintf('Gen %3d: "%s" (Best Fitness: %d)\n', ...
                state.Generation, currentString, bestScore);
    end
end