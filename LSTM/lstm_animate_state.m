clc; clear; close all;

steps = 50;
t = linspace(0, 4*pi, steps);

input_signal = sin(t) + 0.2*randn(1, steps); 

Wf = 10; bf = -5;  % Forget 
Wi = 10; bi = -5;  % Input 
Wc = 2;  bc = 0;   % Candidate 
Wo = 10; bo = -5;  % Output 

h = 0; 
c = 0;

figure('Name', 'Real-Time LSTM Simulation', 'Color', 'w', 'Position', [100, 100, 1000, 600]);

subplot(3,1,1);
hPlotInput = plot(NaN, NaN, 'k-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'k');
title('1. Input Signal (x_t)', 'FontSize', 12);
ylabel('Amplitude'); xlim([1 steps]); ylim([-2 2]); grid on;

subplot(3,1,2);
hPlotCell = plot(NaN, NaN, 'b-s', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
title('2. Cell State / Memory (C_t)', 'FontSize', 12);
ylabel('Memory Value'); xlim([1 steps]); ylim([-2 2]); grid on;
legend('Internal Memory', 'Location', 'northeast');

subplot(3,1,3);
hPlotHidden = plot(NaN, NaN, 'r-d', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
title('3. Output / Hidden State (h_t)', 'FontSize', 12);
xlabel('Time Step'); ylabel('Output'); xlim([1 steps]); ylim([-1 1]); grid on;
legend('LSTM Output', 'Location', 'northeast');

history_x = [];
history_c = [];
history_h = [];

disp('Starting Simulation... Watch the Figure Window!');

for k = 1:steps
    xt = input_signal(k);
    
    v = [h; xt];     
    ft = 1 / (1 + exp(-(xt + 0.5))); 
    it = 1 / (1 + exp(-(xt - 0.5)));
    c_tilde = tanh(xt);    
    c_new = (ft * c) + (it * c_tilde);    
    ot = 1 / (1 + exp(-xt));
    h_new = ot * tanh(c_new);
    c = c_new;
    h = h_new;
    
    history_x = [history_x, xt];
    history_c = [history_c, c];
    history_h = [history_h, h];
    
    set(hPlotInput, 'XData', 1:k, 'YData', history_x);
    set(hPlotCell,  'XData', 1:k, 'YData', history_c);
    set(hPlotHidden,'XData', 1:k, 'YData', history_h);
    
    pause(0.1); 
    drawnow;
end

disp('Simulation Complete.');