% dagger.m
% =============================================================
% FASE 4b — DAgger (Dataset Aggregation) p/ estabilizar a malha fechada.
%
% A LSTM treinada por imitacao pura tem alta fidelidade ABERTA (Pearson
% ~0.98) mas em malha fechada entra em ciclo-limite: os pequenos erros
% levam a estados e(t) oscilatorios que o PID NUNCA visitou (covariate
% shift) e a rede nao sabe amortece-los.
%
% DAgger corrige isto:
%   1. roda a LSTM atual em malha fechada (pitch_LSTM) -> coleta e(t) VISITADO
%   2. rotula esse e(t) com a acao do ESPECIALISTA (PID), analiticamente:
%        ie[k] = ie[k-1] + Ts*e[k-1]   (Forward Euler, = bloco int_e)
%        dc[k] = Kp*e[k] + Ki*ie[k]    (= controlador do pitch_PID)
%   3. agrega (e_visitado -> dc_PID) ao banco e retreina
% Assim a rede aprende a acao correta NOS ESTADOS QUE ELA MESMA PRODUZ.
%
% Saida: datagen/banco_pitch_aug.mat e retreino (lstm_pitch.mat, net_pitch.mat)
%
% Uso:  dagger        (autossuficiente; usa a rede atual como ponto de partida)
% =============================================================

script_dir = fileparts(mfilename('fullpath'));
root_dir   = fileparts(script_dir);
addpath(root_dir, fullfile(root_dir,'datagen'));

run(fullfile(root_dir, 'init_pitch.m'));                 % Ts, Kp, Ki, num_*, den_*, CFG
Sn = load(fullfile(script_dir,'lstm_pitch.mat'),'norm_io'); no = Sn.norm_io;
nmx = no.mu_x; nsx = no.sg_x; nmy = no.mu_y; nsy = no.sg_y; %#ok<NASGU>

%% --- conjunto de referencias para os rollouts da LSTM ---
% (a) varredura de degraus unicos: cobre TODAS as amplitudes (foco do ciclo-limite)
amps = [-10 -9 -8 -7 -6 -5 -4 -3 -2 -1 1 2 3 4 5 6 7 8 9 10];
T_s  = 25; Ns = round(T_s/Ts)+1; ts = (0:Ns-1)'*Ts;
refs = {}; tipos = {};
for a = amps
  r = deg2rad(a) * double(ts >= 2);
  refs{end+1} = struct('t',ts,'ref',r); tipos{end+1} = 'dag_step'; %#ok<SAGROW>
end
% (b) perfis diversos (seeds novas) p/ cobrir manobras
T_p = 20; Np = round(T_p/Ts)+1; tpv = (0:Np-1)'*Ts;
fam = {'degraus','doublet','rampa','multisseno'};
for it = 1:numel(fam)
  for k = 1:12
    r = gerar_perfis(fam{it}, tpv, 7000 + 100*it + k, CFG.A_MAX);
    refs{end+1} = struct('t',tpv,'ref',r); tipos{end+1} = ['dag_' fam{it}]; %#ok<SAGROW>
  end
end
n_roll = numel(refs);

%% --- roda a LSTM em malha fechada e rotula com o PID ---
load_system(fullfile(root_dir,'pitch_LSTM.slx'));
set_param('pitch_LSTM','FastRestart','off');
fprintf('=== DAgger: %d rollouts da LSTM em malha fechada ===\n', n_roll);
dag = struct('id',{},'tipo',{},'seed',{},'meta',{},'t',{},'theta_ref',{},'theta',{},'e',{},'dc',{},'qc',{});
t_d = tic;
for k = 1:n_roll
  theta_ref_ts = timeseries(refs{k}.ref, refs{k}.t);
  set_param('pitch_LSTM','StopTime', num2str(refs{k}.t(end)));
  out = sim('pitch_LSTM');
  e  = out.err.Data(:);
  % acao do especialista (PID discreto) sobre o e VISITADO pela LSTM
  ie = Ts * [0; cumsum(e(1:end-1))];           % Forward Euler (= bloco int_e)
  dc_exp = Kp*e + Ki*ie;
  ep = struct('id',k, 'tipo',tipos{k}, 'seed',7000+k, ...
              'meta',struct('tipo',tipos{k},'dagger',true), ...
              't',out.theta.Time(:), 'theta_ref',out.theta_ref.Data(:), ...
              'theta',out.theta.Data(:), 'e',e, 'dc',dc_exp, ...
              'qc',struct('aprovado',true,'tem_nan',any(isnan([e;dc_exp])), ...
                          'e_final',abs(out.theta.Data(end)-out.theta_ref.Data(end)), ...
                          'max_dc',max(abs(dc_exp))));
  dag(k) = ep;
  if mod(k,10)==0, fprintf('  rollout %d/%d (%s)\n', k, n_roll, tipos{k}); end
end
close_system('pitch_LSTM',0);
fprintf('Rollouts em %.1f s.\n', toc(t_d));

%% --- agrega ao banco (acumula rodadas: usa o aumentado se ja existir) ---
aug_file = fullfile(root_dir,'datagen','banco_pitch_aug.mat');
if isfile(aug_file)
  Sb = load(aug_file,'episodios','config');             % DAgger aggregation
  fprintf('Agregando sobre banco aumentado existente (%d eps).\n', numel(Sb.episodios));
else
  Sb = load(fullfile(root_dir,'datagen','banco_pitch.mat'),'episodios','config');
end
base = Sb.episodios;
% reindexa ids dos dagger p/ nao colidir
for k = 1:numel(dag), dag(k).id = numel(base) + k; end
episodios = [base, dag];
config = Sb.config; config.dagger = struct('n_roll',n_roll, 'amps',amps, 'criado_em',datestr(now));
save(fullfile(root_dir,'datagen','banco_pitch_aug.mat'), 'episodios','config','-v7.3');
fprintf('Banco aumentado: %d (PID) + %d (DAgger) = %d episodios\n', numel(base), numel(dag), numel(episodios));

%% --- retreina com o banco aumentado ---
BANCO_FILE = fullfile(root_dir,'datagen','banco_pitch_aug.mat'); %#ok<NASGU>
run(fullfile(script_dir,'treinar_lstm.m'));
