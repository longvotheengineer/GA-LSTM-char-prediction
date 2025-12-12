function [h_next, c_next, cache] = lstm_cell(xt, h_prev, c_prev, parameters)
    Wf = parameters.Wf; bf = parameters.bf;
    Wi = parameters.Wi; bi = parameters.bi;
    Wc = parameters.Wc; bc = parameters.bc;
    Wo = parameters.Wo; bo = parameters.bo;
    
    concat_input = [h_prev; xt]; 
    
    ft = sigmoid(Wf * concat_input + bf);        % Forget Gate
    it = sigmoid(Wi * concat_input + bi);        % Input Gate
    c_tilde = tanh(Wc * concat_input + bc);      % Candidate Gate
    ot = sigmoid(Wo * concat_input + bo);        % Output Gate
    
    c_next = (ft .* c_prev) + (it .* c_tilde);   % New Cell State
    h_next = ot .* tanh(c_next);                 % New Hidden State
    
    cache.concat_input = concat_input;
    cache.ft = ft;
    cache.it = it;
    cache.c_tilde = c_tilde;
    cache.ot = ot;
    cache.c_prev = c_prev;
    cache.c_next = c_next;
end

function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end