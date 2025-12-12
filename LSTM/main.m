clc; clear; close all;
fprintf('>> Status: Initializing Simulation...\n');

config.input_size  = 2;     
config.hidden_size = 2;     
config.time_steps  = 50;    

concat_len = config.input_size + config.hidden_size;

params.Wf = randn(config.hidden_size, concat_len); params.bf = zeros(config.hidden_size, 1); % Forget
params.Wi = randn(config.hidden_size, concat_len); params.bi = zeros(config.hidden_size, 1); % Input
params.Wc = randn(config.hidden_size, concat_len); params.bc = zeros(config.hidden_size, 1); % Candidate
params.Wo = randn(config.hidden_size, concat_len); params.bo = zeros(config.hidden_size, 1); % Output

t = linspace(0, 4*pi, config.time_steps);
wave = sin(t);
X = repmat(wave, config.input_size, 1); 

h0 = zeros(config.hidden_size, 1);
c0 = zeros(config.hidden_size, 1);

fprintf('>> Status: Running LSTM Forward Pass...\n');
[h_final, c_final] = lstm_forward(X, h0, c0, params);

plot_results(h_final, config.time_steps);
fprintf('>> Status: Complete.\n');

function plot_results(h_states, time_steps)
    figure('Name', 'LSTM Analysis', 'Color', 'w');
    hold on;
    
    [num_neurons, ~] = size(h_states);
    colors = lines(num_neurons);
    
    for i = 1:num_neurons
        plot(1:time_steps, h_states(i,:), 'LineWidth', 2, ...
             'Color', colors(i,:), 'DisplayName', sprintf('Neuron %d', i));
    end
    
    title('LSTM Hidden State Activations over Time');
    xlabel('Time Step (t)');
    ylabel('Activation Level (-1 to 1)');
    legend('show');
    grid on;
    ylim([-1.1 1.1]);
end

function [h_states, c_states] = lstm_forward(x_sequence, h0, c0, parameters)
    [~, time_steps] = size(x_sequence);
    hidden_size = length(h0);
    
    h_states = zeros(hidden_size, time_steps);
    c_states = zeros(hidden_size, time_steps);
    
    h_curr = h0;
    c_curr = c0;
    
    for t = 1:time_steps
        xt = x_sequence(:, t);
        [h_curr, c_curr] = lstm_cell(xt, h_curr, c_curr, parameters);
        h_states(:, t) = h_curr;
        c_states(:, t) = c_curr;
    end
end

function [h_next, c_next] = lstm_cell(xt, h_prev, c_prev, p)
    concat_input = [h_prev; xt];
    
    ft = 1 ./ (1 + exp(-(p.Wf * concat_input + p.bf)));
    it = 1 ./ (1 + exp(-(p.Wi * concat_input + p.bi)));
    ot = 1 ./ (1 + exp(-(p.Wo * concat_input + p.bo)));
    
    c_tilde = tanh(p.Wc * concat_input + p.bc);
    
    c_next = (ft .* c_prev) + (it .* c_tilde);
    h_next = ot .* tanh(c_next);
end