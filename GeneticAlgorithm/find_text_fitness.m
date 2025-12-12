function error = find_text_fitness(x, targetStr)
    targetCode = double(targetStr);    
    error = sum(abs(x - targetCode));
end