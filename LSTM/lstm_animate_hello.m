clc; clear; close all;

chars = ['h', 'e', 'l', 'o'];
char2idx = containers.Map({'h','e','l','o'}, {1, 2, 3, 4});
input_seq_chars  = ['h', 'e', 'l', 'l'];
target_seq_chars = ['e', 'l', 'l', 'o'];

input_size = 4;  
hidden_size = 5;  
output_size = 4;  
learning_rate = 0.1;

rng(42);
scale = 1.0 / sqrt(hidden_size);                                        
W_lstm = randn(hidden_size * 4, input_size + hidden_size + 1) * scale;          
W_out  = randn(output_size, hidden_size + 1) * scale;                       

fprintf('Training LSTM to learn "hello"... (This takes a few seconds)\n');
losses = [];

for epoch = 1:2000
    h = zeros(hidden_size, 1); 
    c = zeros(hidden_size, 1);
    loss = 0;
    cache = cell(1, 4); 
    
    for t = 1:4
        x = zeros(input_size, 1);
        x(char2idx(input_seq_chars(t))) = 1;

        target_idx = char2idx(target_seq_chars(t));
        
        concat = [x; h; 1]; 
        gates = W_lstm * concat;

        f = sigmoid(gates(1:hidden_size));
        i = sigmoid(gates(hidden_size+1:2*hidden_size));
        c_tilde = tanh(gates(2*hidden_size+1:3*hidden_size));
        o = sigmoid(gates(3*hidden_size+1:4*hidden_size));
        
        c_prev = c;
        c = (f .* c_prev) + (i .* c_tilde);
        h = o .* tanh(c);
        y_scores = W_out * [h; 1];
        probs = exp(y_scores) / sum(exp(y_scores));
        loss = loss - log(probs(target_idx));        
        state.x=x; state.h_prev=h; state.c_prev=c_prev; 
        state.gates=[f;i;c_tilde;o]; state.concat=concat; state.probs=probs; state.target=target_idx;
        cache{t} = state;
    end
    
    losses(end+1) = loss;
    
    dW_lstm = zeros(size(W_lstm));
    dW_out = zeros(size(W_out));
    dh_next = zeros(hidden_size, 1);
    dc_next = zeros(hidden_size, 1);
    
    for t = 4:-1:1
        st = cache{t};
        dy = st.probs;
        dy(st.target) = dy(st.target) - 1;
        dW_out = dW_out + dy * [h; 1]';
        
        dh = (W_out(:, 1:end-1)' * dy) + dh_next;        
    end
    
    W_lstm_try = W_lstm + (randn(size(W_lstm)) * 0.05);
    W_out_try  = W_out  + (randn(size(W_out))  * 0.05);
    
    loss_try = 0; h_try=zeros(hidden_size,1); c_try=zeros(hidden_size,1);
    for t2=1:4
        x=zeros(input_size,1); x(char2idx(input_seq_chars(t2)))=1;
        target=char2idx(target_seq_chars(t2));
        con=[x; h_try; 1]; g=W_lstm_try*con;
        f=1./(1+exp(-g(1:hidden_size))); i=1./(1+exp(-g(hidden_size+1:2*hidden_size)));
        ct=tanh(g(2*hidden_size+1:3*hidden_size)); o=1./(1+exp(-g(3*hidden_size+1:4*hidden_size)));
        c_try=(f.*c_try)+(i.*ct); h_try=o.*tanh(c_try);
        ys=W_out_try*[h_try;1]; p=exp(ys)/sum(exp(ys));
        loss_try = loss_try - log(p(target));
    end
    
    if loss_try < loss
        W_lstm = W_lstm_try;
        W_out = W_out_try;
    end
    
    if loss < 0.05, break; end 
end
fprintf('Training Complete. Loss: %.4f\n', losses(end));

f_fig = figure('Name', 'LSTM "Hello" Animation', 'Color', [0.15 0.15 0.15], 'Position', [100 100 1000 600]);

subplot(2,2,1); hPlotInput = bar(zeros(1,4)); title('1. Input (One-Hot)', 'Color','w'); ylim([0 1]);
set(gca, 'XTickLabel', {'h','e','l','o'}, 'Color', 'k', 'YColor','w', 'XColor','w');

subplot(2,2,2); hPlotHidden = bar(zeros(1,hidden_size), 'FaceColor', 'y'); title('2. Hidden State (Neuron Activations)', 'Color','w'); ylim([-1 1]);
set(gca, 'Color', 'k', 'YColor','w', 'XColor','w');

subplot(2,1,2); hPlotProbs = bar(zeros(1,4), 'FaceColor', 'c'); title('3. Output Probabilities (Softmax)', 'Color','w'); ylim([0 1]);
set(gca, 'XTickLabel', {'h','e','l','o'}, 'Color', 'k', 'YColor','w', 'XColor','w');

hText = annotation('textbox', [0.4, 0.45, 0.2, 0.1], 'String', 'Ready...', 'Color', 'w', ...
    'FontSize', 14, 'HorizontalAlignment', 'center', 'EdgeColor', 'none', 'BackgroundColor', 'k');

h = zeros(hidden_size, 1);
c = zeros(hidden_size, 1);

disp('Starting Animation...');
pause(1);

for t = 1:4
    char_in = input_seq_chars(t);
    x = zeros(input_size, 1);
    x(char2idx(char_in)) = 1;
    
    concat = [x; h; 1];
    gates = W_lstm * concat;
    f = sigmoid(gates(1:hidden_size));
    i = sigmoid(gates(hidden_size+1:2*hidden_size));
    c_tilde = tanh(gates(2*hidden_size+1:3*hidden_size));
    o = sigmoid(gates(3*hidden_size+1:4*hidden_size));
    
    c = (f .* c) + (i .* c_tilde);
    h = o .* tanh(c);
    
    y_scores = W_out * [h; 1];
    probs = exp(y_scores) / sum(exp(y_scores));
    
    [max_p, idx_p] = max(probs);
    pred_char = chars(idx_p);
    
    set(hPlotInput, 'YData', x);
    set(hPlotHidden, 'YData', h);
    set(hPlotProbs, 'YData', probs);
    
    msg = sprintf('Input: "%s" --> LSTM thinks: "%s" (%.1f%%)', char_in, pred_char, max_p*100);
    set(hText, 'String', msg);
    
    if pred_char == target_seq_chars(t)
        set(hPlotProbs, 'FaceColor', 'g'); 
    else
        set(hPlotProbs, 'FaceColor', 'r');
    end
    
    drawnow;
    pause(1.5);
    set(hPlotProbs, 'FaceColor', 'c');
end

function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end