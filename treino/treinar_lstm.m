% treinar_lstm.m
% =============================================================
% FASE 3 — Treino da LSTM que imita o controlador PI.
%
% Formulacao SEQUENCIA-A-SEQUENCIA (episodio inteiro):
%   entrada : e(t)   [T x 1]   (erro de pitch, rad)
%   alvo    : dc(t)  [T x 1]   (saida do controlador, rad)
% (convencao trainnet: tempo nas linhas, canais nas colunas)
%
% Por que NAO janela deslizante: dc = Kp*e + Ki*integral(e) desde o inicio
% do episodio. Uma janela finita de e nao determina a integral acumulada
% (problema mal-posto). Com a sequencia inteira, o estado oculto da LSTM
% acumula a acao integral desde t=0 -- exatamente como o bloco Stateful
% Predict fara na malha fechada (reset no inicio, estado persistente
% depois). Treino e inferencia com a MESMA semantica.
%
% Split 70/15/15 por episodio, estratificado por tipo de perfil.
% Normalizacao (z-score) calculada SO no treino.
%
% Saidas (treino/):
%   lstm_pitch.mat        — net, norm_io, split, metricas, config
%   pitch_teste_*.png     — predicao vs alvo em episodios de teste
%
% Uso:  treinar_lstm        (autossuficiente)
% =============================================================

script_dir = fileparts(mfilename('fullpath'));
root_dir   = fileparts(script_dir);
% Banco configuravel: dagger.m define BANCO_FILE (banco aumentado); senao
% usa o banco limpo padrao.
if exist('BANCO_FILE','var') && ~isempty(BANCO_FILE)
  banco_file = BANCO_FILE;
else
  banco_file = fullfile(root_dir, 'datagen', 'banco_pitch.mat');
end
assert(isfile(banco_file), 'Banco nao encontrado: %s (rode rodar_banco antes)', banco_file);

fprintf('Carregando banco: %s\n', banco_file);
S = load(banco_file);
episodios = S.episodios;
config    = S.config;

%% --- filtra aprovados e define grupos (tipo de perfil) ---
aprov  = arrayfun(@(e) e.qc.aprovado, episodios);
eps_ok = episodios(aprov);
n_ok   = numel(eps_ok);
grupo  = string(arrayfun(@(e) string(e.tipo), eps_ok));
grupo  = grupo(:);                          % coluna (estratificacao por tipo)
fprintf('Episodios aprovados: %d/%d\n', n_ok, numel(episodios));

%% --- split 70/15/15 por episodio, estratificado por grupo ---
rng(42);
idx_tr = []; idx_va = []; idx_te = [];
for g = unique(grupo)'
  ig = find(grupo == g);
  ig = ig(randperm(numel(ig)));
  n  = numel(ig);
  n_tr = round(0.70*n); n_va = round(0.15*n);
  idx_tr = [idx_tr; ig(1:n_tr)];                 %#ok<AGROW>
  idx_va = [idx_va; ig(n_tr+1 : n_tr+n_va)];     %#ok<AGROW>
  idx_te = [idx_te; ig(n_tr+n_va+1 : end)];      %#ok<AGROW>
  fprintf('  %-10s: %2d eps -> %d treino / %d val / %d teste\n', g, n, ...
          n_tr, n_va, n - n_tr - n_va);
end

%% --- monta sequencias [T x 1] (trainnet: tempo x canais) ---
seq_x = @(e) single(e.e(:));
seq_y = @(e) single(e.dc(:));
Xtr = arrayfun(seq_x, eps_ok(idx_tr), 'UniformOutput', false)';
Ytr = arrayfun(seq_y, eps_ok(idx_tr), 'UniformOutput', false)';
Xva = arrayfun(seq_x, eps_ok(idx_va), 'UniformOutput', false)';
Yva = arrayfun(seq_y, eps_ok(idx_va), 'UniformOutput', false)';
Xte = arrayfun(seq_x, eps_ok(idx_te), 'UniformOutput', false)';
Yte = arrayfun(seq_y, eps_ok(idx_te), 'UniformOutput', false)';

%% --- normalizacao (SO com dados de treino) ---
all_x = vertcat(Xtr{:}); all_y = vertcat(Ytr{:});
norm_io = struct('mu_x', mean(all_x), 'sg_x', std(all_x), ...
                 'mu_y', mean(all_y), 'sg_y', std(all_y));
nx = @(c) cellfun(@(v) (v - norm_io.mu_x)/norm_io.sg_x, c, 'UniformOutput', false);
ny = @(c) cellfun(@(v) (v - norm_io.mu_y)/norm_io.sg_y, c, 'UniformOutput', false);
Xtr_n = nx(Xtr); Xva_n = nx(Xva); Xte_n = nx(Xte);
Ytr_n = ny(Ytr); Yva_n = ny(Yva);
fprintf('Normalizacao (treino): mu_x=%.2e sg_x=%.3e | mu_y=%.2e sg_y=%.3e\n', ...
        norm_io.mu_x, norm_io.sg_x, norm_io.mu_y, norm_io.sg_y);

%% --- rede ---
% 64->32 (nao 32->16): o canal INTEGRAL (dc = Kp*e + Ki*int(e)) exige
% capacidade p/ manter um acumulador preciso por ~4000 passos; com 32->16
% a validacao plateou em ~0.5 e a rampa (dominada por Ki*int(e)) ficou com
% Pearson 0.39. Licao herdada do DH (treinar_etapa1.m).
camadas = [
  sequenceInputLayer(1)
  lstmLayer(64, 'OutputMode', 'sequence')
  lstmLayer(32, 'OutputMode', 'sequence')
  fullyConnectedLayer(1)
];

opts = trainingOptions('adam', ...
  'MaxEpochs', 350, ...
  'MiniBatchSize', 32, ...
  'InitialLearnRate', 3e-3, ...
  'LearnRateSchedule', 'piecewise', ...
  'LearnRateDropPeriod', 120, ...
  'LearnRateDropFactor', 0.5, ...
  'GradientThreshold', 1, ...
  'Shuffle', 'every-epoch', ...
  'ValidationData', {Xva_n, Yva_n}, ...
  'ValidationFrequency', 25, ...
  'ValidationPatience', 30, ...
  'OutputNetwork', 'best-validation', ...
  'ExecutionEnvironment', 'auto', ...
  'Plots', 'none', ...
  'Verbose', true, 'VerboseFrequency', 25);

fprintf('\nTreinando (%d sequencias de %d passos)...\n', numel(Xtr_n), size(Xtr_n{1},1));
t_tr = tic;
[net, info] = trainnet(Xtr_n, Ytr_n, camadas, 'mse', opts);
fprintf('Treino concluido em %.1f min.\n', toc(t_tr)/60);

%% --- avaliacao no teste (denormalizada, rad) ---
n_te = numel(Xte_n);
rmse_ep = zeros(n_te,1); r_ep = zeros(n_te,1);
pred_te = cell(n_te,1);
for i = 1:n_te
  yp = double(predict(net, Xte_n{i})); yp = yp(:);
  yp = yp * norm_io.sg_y + norm_io.mu_y;
  yt = double(Yte{i}); yt = yt(:);
  pred_te{i} = yp;
  rmse_ep(i) = rms(yp - yt);
  c = corrcoef(yp, yt); r_ep(i) = c(1,2);
end

fprintf('\n=== METRICAS NO TESTE (%d episodios) ===\n', n_te);
fprintf('RMSE(dc): mediana %.3e | p95 %.3e | max %.3e rad\n', ...
        median(rmse_ep), prctile(rmse_ep,95), max(rmse_ep));
fprintf('Pearson(dc_LSTM x dc_PI): mediana %.5f | min %.5f\n', median(r_ep), min(r_ep));
gr_te = grupo(idx_te);
for g = unique(gr_te)'
  m = gr_te == g;
  fprintf('  %-10s: RMSE mediana %.3e rad | r mediana %.5f\n', ...
          g, median(rmse_ep(m)), median(r_ep(m)));
end

%% --- salva rede + contexto ---
treino = struct();
treino.net          = net;
treino.norm_io      = norm_io;
treino.camadas_desc = 'seqInput(1) -> lstm(32,seq) -> lstm(16,seq) -> fc(1)';
treino.split        = struct('idx_tr',idx_tr, 'idx_va',idx_va, 'idx_te',idx_te, ...
                             'ids_globais', arrayfun(@(e) e.id, eps_ok));
treino.metricas     = struct('rmse_ep',rmse_ep, 'r_ep',r_ep, 'grupos_teste',gr_te);
treino.config       = config;
treino.criado_em    = datestr(now);
save(fullfile(script_dir, 'lstm_pitch.mat'), '-struct', 'treino');
% MAT so' com a rede, lido pelo bloco Stateful Predict do pitch_LSTM.slx
save(fullfile(root_dir, 'net_pitch.mat'), 'net');
fprintf('Rede salva em %s (e net_pitch.mat na raiz)\n', fullfile(script_dir, 'lstm_pitch.mat'));

%% --- figuras: melhor, mediano e pior episodio de teste ---
[~, ord] = sort(rmse_ep);
casos = [ord(1), ord(ceil(n_te/2)), ord(end)];
nomes = {'melhor', 'mediano', 'pior'};
prev_vis = get(0, 'DefaultFigureVisible');
set(0, 'DefaultFigureVisible', 'off');
for j = 1:3
  i  = casos(j);
  ep = eps_ok(idx_te(i));
  f = figure('Color','w', 'Position',[100 100 950 520], 'Visible','off');
  subplot(2,1,1); hold on; grid on;
  plot(ep.t, rad2deg(double(Yte{i})), 'k', 'LineWidth', 1.5);
  plot(ep.t, rad2deg(pred_te{i}), '--', 'Color',[0.85 0.33 0.10], 'LineWidth', 1.3);
  ylabel('dc [deg]'); legend({'PI (alvo)','LSTM'}, 'Location','best');
  title(sprintf('Teste %s — ep %d (%s) | RMSE %.2e rad | r %.5f', ...
        nomes{j}, ep.id, gr_te(i), rmse_ep(i), r_ep(i)));
  subplot(2,1,2); grid on;
  plot(ep.t, rad2deg(pred_te{i} - double(Yte{i})), 'Color',[0 0.45 0.74]);
  ylabel('erro dc [deg]'); xlabel('t [s]');
  exportgraphics(f, fullfile(script_dir, sprintf('pitch_teste_%s.png', nomes{j})), ...
                 'BackgroundColor','white', 'Resolution', 150);
  close(f);
end
set(0, 'DefaultFigureVisible', prev_vis);
fprintf('Figuras salvas em %s\n', script_dir);
