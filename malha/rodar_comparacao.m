% rodar_comparacao.m
% =============================================================
% FASE 4 — Comparacao em MALHA FECHADA: PID (pitch_PID) vs LSTM (pitch_LSTM).
%
% Roda os dois modelos nos mesmos cenarios de referencia e compara o
% rastreamento de theta e o comando dc. Inclui um perfil NAO visto no
% treino (multisseno com seed nova). Gera figuras em resultados/.
%
% Metricas por cenario:
%   RMSE_track = rms(theta - theta_ref)         (rad)
%   e_ss       = |theta(fim) - theta_ref(fim)|  (erro de regime)
%   desvio     = rms(theta_LSTM - theta_PID)    (quao perto a LSTM fica do PID)
%
% Uso:  rodar_comparacao        (autossuficiente; roda init_pitch antes)
% =============================================================

script_dir = fileparts(mfilename('fullpath'));
root_dir   = fileparts(script_dir);
res_dir    = fullfile(root_dir, 'resultados');
addpath(root_dir, fullfile(root_dir,'datagen'));
if ~exist(res_dir,'dir'), mkdir(res_dir); end

%% --- inicializacao (planta, Ts) + normalizacao da rede ---
run(fullfile(root_dir, 'init_pitch.m'));
S = load(fullfile(root_dir,'treino','lstm_pitch.mat'),'norm_io');
no = S.norm_io;
nmx = no.mu_x; nsx = no.sg_x; nmy = no.mu_y; nsy = no.sg_y; %#ok<NASGU>

%% --- cenarios de teste ---
T  = 25;  dt = Ts;  N = round(T/dt)+1;  tp = (0:N-1)'*dt;
deg = @(x) deg2rad(x);
mk_step = @(amp,t0) deg(amp) * double(tp >= t0);
cen = {};
cen{end+1} = struct('nome','degrau_+3', 'ref', mk_step(3,2));
cen{end+1} = struct('nome','degrau_+5', 'ref', mk_step(5,2));
cen{end+1} = struct('nome','degrau_+7', 'ref', mk_step(7,2));
cen{end+1} = struct('nome','degrau_-5', 'ref', mk_step(-5,2));
% dois degraus
r = mk_step(5,2) + (deg(-3))*double(tp>=14); cen{end+1} = struct('nome','dois_degraus','ref',r);
% rampa e multisseno NAO vistos (seeds altas)
cen{end+1} = struct('nome','rampa_nova',     'ref', gerar_perfis('rampa',     tp, 9001, CFG.A_MAX));
cen{end+1} = struct('nome','multisseno_novo','ref', gerar_perfis('multisseno',tp, 9002, CFG.A_MAX));
n_cen = numel(cen);

%% --- prepara modelos ---
% (sem Fast Restart: o bloco Stateful Predict reavalia a mask a cada sim e
%  nao permite alternar modelos sob Fast Restart.)
for m = {'pitch_PID','pitch_LSTM'}
  load_system(fullfile(root_dir,[m{1} '.slx']));
  set_param(m{1}, 'StopTime', num2str(T));
end

prev_vis = get(0,'DefaultFigureVisible'); set(0,'DefaultFigureVisible','off');
res = struct('nome',{},'rmse_pid',{},'rmse_lstm',{},'ess_pid',{},'ess_lstm',{},'desvio',{});

fprintf('=== Comparacao PID vs LSTM (%d cenarios, T=%g s) ===\n', n_cen, T);
for c = 1:n_cen
  theta_ref_ts = timeseries(cen{c}.ref, tp);

  oP = sim('pitch_PID');   % PID
  oL = sim('pitch_LSTM');  % LSTM
  to   = oP.theta.Time(:);
  rref = oP.theta_ref.Data(:);
  thP  = oP.theta.Data(:);  dcP = oP.dc.Data(:);
  thL  = oL.theta.Data(:);  dcL = oL.dc.Data(:);

  rmse_pid  = rms(thP - rref);   rmse_lstm = rms(thL - rref);
  ess_pid   = abs(thP(end)-rref(end));  ess_lstm = abs(thL(end)-rref(end));
  desvio    = rms(thL - thP);

  res(c) = struct('nome',cen{c}.nome, 'rmse_pid',rmse_pid, 'rmse_lstm',rmse_lstm, ...
                  'ess_pid',ess_pid, 'ess_lstm',ess_lstm, 'desvio',desvio);
  fprintf('  %-16s | RMSE PID %.3f deg / LSTM %.3f deg | e_ss PID %.3f / LSTM %.3f deg | desvio %.3f deg\n', ...
    cen{c}.nome, rad2deg(rmse_pid), rad2deg(rmse_lstm), rad2deg(ess_pid), rad2deg(ess_lstm), rad2deg(desvio));

  % --- figura ---
  f = figure('Color','w','Position',[100 100 950 600],'Visible','off');
  subplot(2,1,1); hold on; grid on;
  plot(to, rad2deg(rref), 'k:', 'LineWidth',1.2);
  plot(to, rad2deg(thP),  'Color',[0 0.45 0.74], 'LineWidth',1.4);
  plot(to, rad2deg(thL),  '--','Color',[0.85 0.33 0.10], 'LineWidth',1.4);
  ylabel('\theta [deg]'); legend({'\theta_{ref}','PID','LSTM'},'Location','best');
  title(sprintf('%s — RMSE PID %.3f° / LSTM %.3f° | e_{ss} LSTM %.3f° | desvio %.3f°', ...
    strrep(cen{c}.nome,'_','\_'), rad2deg(rmse_pid), rad2deg(rmse_lstm), rad2deg(ess_lstm), rad2deg(desvio)));
  subplot(2,1,2); hold on; grid on;
  plot(to, dcP, 'Color',[0 0.45 0.74], 'LineWidth',1.2);
  plot(to, dcL, '--','Color',[0.85 0.33 0.10], 'LineWidth',1.2);
  ylabel('dc [rad]'); xlabel('t [s]'); legend({'PID','LSTM'},'Location','best');
  exportgraphics(f, fullfile(res_dir, sprintf('comp_%s.png', cen{c}.nome)), ...
                 'BackgroundColor','white','Resolution',150);
  close(f);
end
set(0,'DefaultFigureVisible',prev_vis);

for m = {'pitch_PID','pitch_LSTM'}
  try, set_param(m{1},'SimulationCommand','stop'); catch, end
  try, if strcmp(get_param(m{1},'FastRestart'),'on'), set_param(m{1},'FastRestart','off'); end, catch, end
  try, close_system(m{1},0); catch, end
end

%% --- resumo ---
fprintf('\n=== RESUMO ===\n');
fprintf('RMSE rastreamento (mediana): PID %.3f deg | LSTM %.3f deg\n', ...
  rad2deg(median([res.rmse_pid])), rad2deg(median([res.rmse_lstm])));
fprintf('Erro de regime (mediana):    PID %.3f deg | LSTM %.3f deg\n', ...
  rad2deg(median([res.ess_pid])), rad2deg(median([res.ess_lstm])));
fprintf('Desvio LSTM-PID (mediana):   %.3f deg\n', rad2deg(median([res.desvio])));
save(fullfile(res_dir,'comparacao.mat'), 'res');
fprintf('Figuras e comparacao.mat salvos em %s\n', res_dir);
