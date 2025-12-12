clc; clear; close all;

h_prev = [0.5; -0.2; 0.1];  
C_prev = [0.8; -0.5; 0.0];   
x_t    = [1.0; 0.5];         

hidden_size = 3;
input_size = 2;

rng(42);
Wf = randn(hidden_size, input_size + hidden_size); bf = randn(hidden_size,1);
Wi = randn(hidden_size, input_size + hidden_size); bi = randn(hidden_size,1);
Wc = randn(hidden_size, input_size + hidden_size); bc = randn(hidden_size,1);
Wo = randn(hidden_size, input_size + hidden_size); bo = randn(hidden_size,1);

f = figure('Name', 'LSTM Single Step Analysis', 'Color', [0.1 0.1 0.1], ...
    'Position', [100, 100, 1200, 700], 'MenuBar', 'none');
ax = axes('Position', [0 0 1 1], 'Color', [0.1 0.1 0.1]);
axis off; hold on;
xlim([0 100]); ylim([0 100]);

draw_text = @(x, y, txt, sz, col) text(x, y, txt, 'Color', col, ...
    'FontSize', sz, 'FontWeight', 'bold', 'FontName', 'Consolas', 'HorizontalAlignment', 'left');

draw_text(5, 95, 'LSTM SINGLE STEP DIAGNOSTIC [t=1]', 16, 'w');
draw_text(5, 91, 'Analyzing Internal Matrix Operations...', 10, [0.7 0.7 0.7]);

pause(0.5);
draw_text(5, 80, '1. GATHER INPUTS', 12, 'y');
pause(0.5);
draw_text(10, 76, sprintf('Previous Hidden (h_{t-1}) size 3x1'), 10, 'c');
draw_text(10, 72, mat2str(h_prev, 2), 10, 'c');
pause(0.5);
draw_text(40, 76, sprintf('Current Input (x_t) size 2x1'), 10, 'g');
draw_text(40, 72, mat2str(x_t, 2), 10, 'g');

pause(1.0);
concat_vec = [h_prev; x_t];
draw_text(70, 76, '>> CONCATENATE [h; x]', 10, 'w');
draw_text(70, 72, mat2str(concat_vec, 2), 10, 'w');
plot([30 68], [74 74], 'w:', 'LineWidth', 1);

pause(1.5);
draw_text(5, 60, '2. CALCULATE GATES (Sigmoid Activations)', 12, 'y');

% Forget Gate
pause(0.5);
ft_raw = Wf * concat_vec + bf;
ft = 1 ./ (1 + exp(-ft_raw));
draw_text(10, 55, 'Forget Gate (f_t):', 11, [1 0.5 0.5]);
t_f = draw_text(35, 55, 'Thinking...', 10, [0.5 0.5 0.5]);
pause(0.5);
delete(t_f);
draw_text(35, 55, ['Sigmoid(' mat2str(ft_raw,2) ') -> ' mat2str(ft,2)], 11, [1 0.5 0.5]);

% Input Gate
pause(0.5);
it_raw = Wi * concat_vec + bi;
it = 1 ./ (1 + exp(-it_raw));
draw_text(10, 50, 'Input Gate (i_t):', 11, [0.5 1 0.5]);
t_i = draw_text(35, 50, 'Thinking...', 10, [0.5 0.5 0.5]);
pause(0.5);
delete(t_i);
draw_text(35, 50, ['Sigmoid(' mat2str(it_raw,2) ') -> ' mat2str(it,2)], 11, [0.5 1 0.5]);

% Candidate
pause(0.5);
C_tilde = tanh(Wc * concat_vec + bc);
draw_text(10, 45, 'Candidate (~C_t):', 11, [0.5 0.5 1]);
t_c = draw_text(35, 45, 'Thinking...', 10, [0.5 0.5 0.5]);
pause(0.5);
delete(t_c);
draw_text(35, 45, ['Tanh(' mat2str(Wc*concat_vec,2) ')    -> ' mat2str(C_tilde,2)], 11, [0.5 0.5 1]);

pause(1.5);
draw_text(5, 35, '3. UPDATE MEMORY (Element-wise Math)', 12, 'y');
text(10, 31, 'C_t = (f_t .* C_{prev}) + (i_t .* ~C_t)', 'Color', 'w', 'FontSize', 12, 'FontName', 'Consolas');

term1 = ft .* C_prev;
term2 = it .* C_tilde;
C_new = term1 + term2;

pause(1.0);
draw_text(10, 26, 'Keep Old?', 10, [0.7 0.7 0.7]);
draw_text(10, 23, mat2str(term1, 2), 10, 'c');

pause(0.5);
text(25, 24, '+', 'Color', 'w', 'FontSize', 14, 'FontWeight', 'bold');

draw_text(30, 26, 'Add New?', 10, [0.7 0.7 0.7]);
draw_text(30, 23, mat2str(term2, 2), 10, 'g');

pause(0.5);
text(45, 24, '=', 'Color', 'w', 'FontSize', 14, 'FontWeight', 'bold');

draw_text(50, 26, 'NEW CELL STATE', 11, 'y');
draw_text(50, 23, mat2str(C_new, 2), 12, 'y');
rectangle('Position', [49 21 15 7], 'EdgeColor', 'y', 'LineWidth', 2); 

pause(1.5);
draw_text(5, 10, '4. GENERATE OUTPUT', 12, 'y');

ot = 1 ./ (1 + exp(-(Wo * concat_vec + bo)));
h_new = ot .* tanh(C_new);

draw_text(10, 6, 'Output Gate (o_t):', 10, [1 0.7 0.2]);
draw_text(30, 6, mat2str(ot, 2), 10, [1 0.7 0.2]);

pause(1.0);
draw_text(50, 6, '>> FINAL HIDDEN STATE (h_t):', 11, 'm');
t_h = draw_text(80, 6, 'Calculated!', 12, 'm');
delete(t_h);
draw_text(75, 6, mat2str(h_new, 2), 12, 'm');
rectangle('Position', [74 4 15 5], 'EdgeColor', 'm', 'LineWidth', 2, 'LineStyle', '--');

draw_text(5, 1, 'SINGLE STEP COMPLETE.', 10, 'w');