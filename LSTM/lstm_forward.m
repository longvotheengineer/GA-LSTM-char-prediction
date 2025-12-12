function [h_states, c_states] = lstm_forward(x_sequence, h0, c0, parameters)
    [~, time_steps] = size(x_sequence);
    hidden_size = length(h0);
    
    h_states = zeros(hidden_size, time_steps);
    c_states = zeros(hidden_size, time_steps);
    
    h_curr = h0;
    c_curr = c0;
    
    fprintf('=== STARTING LSTM TIME SEQUENCE ===\n');
    
    for t = 1:time_steps
        xt = x_sequence(:, t);
        
        Wf = parameters.Wf; bf = parameters.bf;
        Wi = parameters.Wi; bi = parameters.bi;
        Wc = parameters.Wc; bc = parameters.bc;
        Wo = parameters.Wo; bo = parameters.bo;
        
        concat_input = [h_curr; xt];
        
        ft = 1 ./ (1 + exp(-(Wf * concat_input + bf)));
        it = 1 ./ (1 + exp(-(Wi * concat_input + bi)));
        c_tilde = tanh(Wc * concat_input + bc);
        ot = 1 ./ (1 + exp(-(Wo * concat_input + bo)));
        
        c_next = (ft .* c_curr) + (it .* c_tilde);
        h_next = ot .* tanh(c_next);
        
        fprintf('\n--- Time Step t = %d ---\n', t);
        fprintf('Input (xt):         %s\n', mat2str(xt, 2));
        fprintf('Forget Gate (ft):   %s  (decides what to keep from old memory)\n', mat2str(ft, 2));
        fprintf('Input Gate (it):    %s  (decides what new info to add)\n', mat2str(it, 2));
        fprintf('Old Cell (C_prev):  %s\n', mat2str(c_curr, 2));
        fprintf('New Cell (C_new):   %s  <-- Updated Long-Term Memory\n', mat2str(c_next, 2));
        fprintf('Output (h_new):     %s  <-- This goes to the Graph\n', mat2str(h_next, 2));
        
        h_curr = h_next;
        c_curr = c_next;
        
        h_states(:, t) = h_curr;
        c_states(:, t) = c_curr;
    end
    fprintf('\n=== SEQUENCE COMPLETE ===\n');
end