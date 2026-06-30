% rodar_banco.m
% =============================================================
% FASE 2 — Geracao do banco de dados (imitacao do PI).
%
% Roda o modelo pitch_PID.slx para uma bateria de perfis de theta_ref
% (degraus, doublet, rampa, multisseno) e coleta, por episodio:
%     entrada : e(t)   = theta_ref - theta   (erro de pitch)
%     alvo    : dc(t)  = saida do controlador (comando do elevador)
% que sao os pares (entrada -> alvo) para a LSTM imitar.
%
% Saida: datagen/banco_pitch.mat  (episodios, config)
%
% Uso:  rodar_banco            (autossuficiente; roda init_pitch antes)
% =============================================================

script_dir = fileparts(mfilename('fullpath'));
root_dir   = fileparts(script_dir);
addpath(script_dir); addpath(root_dir);

%% --- inicializacao (planta, Ts, ganhos do PI) ---
run(fullfile(root_dir, 'init_pitch.m'));      % define Ts, Kp, Ki, num_*, den_*, CFG

%% --- configuracao do lote ---
T_EP  = 20;                                   % [s] duracao de cada episodio
dt    = Ts;                                    % (20 s = 2001 passos: integracao
N     = round(T_EP/dt) + 1;                    %  mais facil p/ a LSTM e epocas mais rapidas)
tp    = (0:N-1)' * dt;                         % timestamps = n*dt (bit-exatos)
mdl   = 'pitch_PID';
tipos = {'degraus','doublet','rampa','multisseno'};
N_POR_TIPO = 60;                               % 60 x 4 = 240 episodios

%% --- definicoes dos episodios (seeds deterministicas) ---
ep_defs = {}; s = 0;
for rep = 1:N_POR_TIPO
  for it = 1:numel(tipos)
    s = s + 1;
    ep_defs{end+1} = struct('tipo', tipos{it}, 'seed', s);  %#ok<SAGROW>
  end
end
n_ep = numel(ep_defs);

%% --- prepara modelo (compila 1x via Fast Restart) ---
load_system(fullfile(root_dir, [mdl '.slx']));
set_param(mdl, 'StopTime', num2str(T_EP));
set_param(mdl, 'FastRestart', 'on');

fprintf('=== Banco pitch linear: %d episodios de %g s @ %g Hz ===\n', n_ep, T_EP, 1/dt);
t_lote = tic;

%% --- loop de episodios ---
for k = 1:n_ep
  def = ep_defs{k};
  [vec, meta] = gerar_perfis(def.tipo, tp, def.seed, CFG.A_MAX);
  theta_ref_ts = timeseries(vec, tp);          % lido pelo bloco From Workspace

  out = sim(mdl);

  ep = struct();
  ep.id        = k;
  ep.tipo      = def.tipo;
  ep.seed      = def.seed;
  ep.meta      = meta;
  ep.t         = out.theta.Time(:);
  ep.theta_ref = out.theta_ref.Data(:);
  ep.theta     = out.theta.Data(:);
  ep.e         = out.err.Data(:);
  ep.dc        = out.dc.Data(:);

  % guarda: theta_ref logado == perfil enviado (ZOH bit-exato)
  d_ref = max(abs(ep.theta_ref - vec));
  if d_ref > 1e-9
    warning('Ep %d: theta_ref logado difere do perfil em %.2e (Fast Restart?)', k, d_ref);
  end

  % --- controle de qualidade ---
  qc = struct();
  qc.tem_nan = any(isnan([ep.theta; ep.e; ep.dc]));
  qc.e_final = abs(ep.theta(end) - ep.theta_ref(end));
  qc.max_dc  = max(abs(ep.dc));
  % Sistema linear e estavel: nada diverge. Aceita o episodio se nao ha
  % NaN e dc fica em faixa fisica -- mesmo nao acomodado e' par (e,dc)
  % valido para imitacao. (e_final fica so' como diagnostico.)
  ok_nan = ~qc.tem_nan;
  ok_dc  = qc.max_dc < 10;                        % dc tipico ~3 rad p/ 10 deg
  qc.aprovado = ok_nan && ok_dc;
  ep.qc = qc;

  if k == 1, episodios = ep; else, episodios(k) = ep; end %#ok<SAGROW>

  if ~qc.aprovado || mod(k, 20) == 0
    fprintf('[%3d/%3d] %-10s seed %3d | e_final=%.3f deg | max|dc|=%.2f | %s\n', ...
      k, n_ep, def.tipo, def.seed, rad2deg(qc.e_final), qc.max_dc, ...
      string(ternario(qc.aprovado, 'OK', 'REPROVADO')));
  end
end

set_param(mdl, 'FastRestart', 'off');
close_system(mdl, 0);
fprintf('Lote concluido em %.1f s.\n', toc(t_lote));

%% --- salvamento ---
config = struct('T_EP',T_EP, 'dt',dt, 'N',N, 'Ts',Ts, ...
                'CTRL_TYPE',CFG.CTRL_TYPE, 'Kp',Kp, 'Ki',Ki, 'A_MAX',CFG.A_MAX, ...
                'num_elev',num_elev, 'den_elev',den_elev, ...
                'num_din',num_din, 'den_din',den_din, ...
                'tipos',{tipos}, 'N_POR_TIPO',N_POR_TIPO, ...
                'criado_em',datestr(now));
save(fullfile(script_dir, 'banco_pitch.mat'), 'episodios', 'config', '-v7.3');

%% --- resumo ---
n_ok = sum(arrayfun(@(e) e.qc.aprovado, episodios));
fprintf('\n=== RESUMO DO BANCO ===\n');
fprintf('Aprovados: %d/%d (%.1f%%)\n', n_ok, n_ep, 100*n_ok/n_ep);
for it = 1:numel(tipos)
  idx = arrayfun(@(e) strcmp(e.tipo, tipos{it}), episodios);
  oks = arrayfun(@(e) e.qc.aprovado, episodios(idx));
  fprintf('  %-10s: %2d/%2d aprovados\n', tipos{it}, sum(oks), nnz(idx));
end
fprintf('Amostras totais (aprovados): %d\n', n_ok * N);
fprintf('Salvo em: %s\n', fullfile(script_dir, 'banco_pitch.mat'));

%% --- helper ---
function s = ternario(cond, a, b)
  if cond, s = a; else, s = b; end
end
