clc; clear; close all;

steps = 20;
input_data = [zeros(1,5), ones(1,5), zeros(1,5), -ones(1,5)];

Wf = [ 0,  2.0]; bf = -1.0;  % Forget Gate 
Wi = [ 0,  2.0]; bi = -0.5;  % Input Gate 
Wc = [ 0,  1.0]; bc =  0.0;  % Candidate 
Wo = [ 0,  5.0]; bo = -2.0;  % Output Gate 

h = 0;
C = 0;

fig = figure('Name', 'LSTM Internal Mechanics', 'Color', 'w', 'Position', [100, 100, 1000, 700]);

subplot(4, 2, [1 2]);
hPlotInput = plot(NaN, NaN, 'k-o', 'LineWidth', 2);
hold on;
hMarker = plot(NaN, NaN, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
title('1. External Input (x_t)', 'FontSize', 12, 'FontWeight', 'bold');
xlim([1 steps]); ylim([-1.5 1.5]); grid on;
ylabel('Value');

subplot(4, 2, 3);
hBarGates = bar([0, 0, 0]);
set(gca, 'XTickLabel', {'Forget (f)', 'Input (i)', 'Output (o)'});
ylim([0 1.1]);
title('2. The Gates (Sigmoid)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Open (1) / Closed (0)');
grid on;

subplot(4, 2, 4);
hBarCand = bar([0, 0]);
set(gca, 'XTickLabel', {'Candidate (~C)', 'Old Cell (C_{t-1})'});
ylim([-1.1 1.1]);
title('3. Proposal vs History', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

subplot(4, 2, 5);
hBarCell = bar(0, 'FaceColor', 'b');
ylim([-2 2]);
title('4. Updated Cell State (C_t)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Memory Content');
grid on;

subplot(4, 2, 6);
hBarHidden = bar(0, 'FaceColor', 'r');
ylim([-1.1 1.1]);
title('5. Final Output (h_t)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Output');
grid on;

subplot(4, 2, [7 8]);
hPlotCellHist = plot(NaN, NaN, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Cell State'); hold on;
hPlotHiddenHist = plot(NaN, NaN, 'r-', 'LineWidth', 2, 'DisplayName', 'Output');
legend('Location', 'northwest');
title('6. History over Time', 'FontSize', 12);
xlim([1 steps]); ylim([-2 2]); grid on;

hist_C = nan(1, steps);
hist_h = nan(1, steps);

disp('Starting Animation... (Press Ctrl+C to stop)');

for t = 1:steps
    xt = input_data(t);
    
    concat = [h; xt];
    
    ft = 1 / (1 + exp(-(Wf * concat + bf))); % Forget
    it = 1 / (1 + exp(-(Wi * concat + bi))); % Input
    ot = 1 / (1 + exp(-(Wo * concat + bo))); % Output
    
    C_tilde = tanh(Wc * concat + bc);
    C_old = C;
    C = (ft * C) + (it * C_tilde);    
    h = ot * tanh(C);
    
    set(hPlotInput, 'XData', 1:t, 'YData', input_data(1:t));
    set(hMarker, 'XData', t, 'YData', xt);
    
    set(hBarGates, 'YData', [ft, it, ot]);   
    set(hBarCand, 'YData', [C_tilde, C_old]);
    set(hBarCell, 'YData', C);
    set(hBarHidden, 'YData', h);
    
    hist_C(t) = C;
    hist_h(t) = h;
    set(hPlotCellHist, 'XData', 1:steps, 'YData', hist_C);
    set(hPlotHiddenHist, 'XData', 1:steps, 'YData', hist_h);
    
    drawnow;
    pause(0.8); 
end

disp('Simulation Complete.');