% rodar_robustez.m
% =============================================================
% FASE 4c — Robustez: condicoes iniciais e referencias FORA do treino.
%
% A LSTM foi treinada com planta partindo do repouso (theta(0)=0) e
% |theta_ref| <= 10 deg. Aqui testamos generalizacao:
%   (A) CONDICOES INICIAIS: planta comeca em theta(0) != 0 (a malha tem que
%       regular/rastrear a partir de um offset) — via x0_din no bloco State-Space.
%   (B) REFERENCIAS FORA DO TREINO: amplitudes alem de +-10 deg (extrapolacao)
%       e sinais mais rapidos que o treino (0.05-0.5 Hz).
%
% Compara PID vs LSTM e gera figuras em resultados/robustez_*.png.
%
% Uso:  rodar_robustez        (autossuficiente)
% =============================================================

script_dir = fileparts(mfilename('fullpath'));
root_dir   = fileparts(script_dir);
res_dir    = fullfile(root_dir,'resultados');
addpath(root_dir, fullfile(root_dir,'datagen'));
if ~exist(res_dir,'dir'), mkdir(res_dir); end

run(fullfile(root_dir,'init_pitch.m'));
S = load(fullfile(root_dir,'treino','lstm_pitch.mat'),'norm_io'); no=S.norm_io;
nmx=no.mu_x; nsx=no.sg_x; nmy=no.mu_y; nsy=no.sg_y; %#ok<NASGU>

T = 25; N = round(T/Ts)+1; tp = (0:N-1)'*Ts;
deg = @(x) deg2rad(x);
ic  = @(th0) [Cdin; Cdin*Adin] \ [deg(th0); 0];     % x0_din p/ theta(0)=th0 [deg]
stp = @(amp,t0) deg(amp)*double(tp>=t0);

%% --- cenarios ---
cen = {};
% (A) condicoes iniciais
cen{end+1} = struct('nome','IC_+8_reg',  'x0',ic(8),   'ref',zeros(N,1),           'grp','cond. inicial');
cen{end+1} = struct('nome','IC_-10_reg', 'x0',ic(-10), 'ref',zeros(N,1),           'grp','cond. inicial');
cen{end+1} = struct('nome','IC_+10_to_-5','x0',ic(10),  'ref',stp(-5,2),            'grp','cond. inicial');
cen{end+1} = struct('nome','IC_-8_to_+6', 'x0',ic(-8),  'ref',stp(6,2),             'grp','cond. inicial');
% (B) referencias fora do treino (extrapolacao de amplitude) — planta em repouso
cen{end+1} = struct('nome','ref_+13',     'x0',[0;0],   'ref',stp(13,2),            'grp','extrapolacao');
cen{end+1} = struct('nome','ref_-15',     'x0',[0;0],   'ref',stp(-15,2),           'grp','extrapolacao');
% (B) referencia mais rapida que o treino (multisseno 0.4-1.2 Hz)
rng(4242); nc=4; fr=0.4+0.8*rand(1,nc); ph=2*pi*rand(1,nc); am=0.3+0.7*rand(1,nc);
sig=zeros(N,1); for k=1:nc, sig=sig+am(k)*sin(2*pi*fr(k)*(tp-3)+ph(k)); end
sig = sig/max(abs(sig))*deg(8); w=min(1,max(0,(tp-3)/2));
cen{end+1} = struct('nome','ref_rapida',  'x0',[0;0],   'ref',w.*sig,               'grp','extrapolacao');

n_cen = numel(cen);
for m = {'pitch_PID','pitch_LSTM'}
  load_system(fullfile(root_dir,[m{1} '.slx'])); set_param(m{1},'StopTime',num2str(T));
end

prev=get(0,'DefaultFigureVisible'); set(0,'DefaultFigureVisible','off');
res = struct('nome',{},'grp',{},'rmse_pid',{},'rmse_lstm',{},'ess_pid',{},'ess_lstm',{},'desvio',{},'estavel',{});
fprintf('=== Robustez: %d cenarios (T=%g s) ===\n', n_cen, T);
for c = 1:n_cen
  x0_din = cen{c}.x0;                                  %#ok<NASGU> usado pelo State-Space
  theta_ref_ts = timeseries(cen{c}.ref, tp);
  oP = sim('pitch_PID'); oL = sim('pitch_LSTM');
  to=oP.theta.Time(:); rref=oP.theta_ref.Data(:);
  thP=oP.theta.Data(:); dcP=oP.dc.Data(:);
  thL=oL.theta.Data(:); dcL=oL.dc.Data(:);
  rmse_pid=rms(thP-rref); rmse_lstm=rms(thL-rref);
  ess_pid=abs(thP(end)-rref(end)); ess_lstm=abs(thL(end)-rref(end));
  desvio=rms(thL-thP);
  estavel = all(isfinite(thL)) && max(abs(thL)) < deg(60);   % nao divergiu
  res(c)=struct('nome',cen{c}.nome,'grp',cen{c}.grp,'rmse_pid',rmse_pid,'rmse_lstm',rmse_lstm, ...
                'ess_pid',ess_pid,'ess_lstm',ess_lstm,'desvio',desvio,'estavel',estavel);
  fprintf('  %-14s [%-12s] | RMSE PID %.3f / LSTM %.3f deg | e_ss LSTM %.3f deg | desvio %.3f deg | estavel %d\n', ...
    cen{c}.nome, cen{c}.grp, rad2deg(rmse_pid), rad2deg(rmse_lstm), rad2deg(ess_lstm), rad2deg(desvio), estavel);

  f=figure('Color','w','Position',[100 100 950 600],'Visible','off');
  subplot(2,1,1); hold on; grid on;
  plot(to,rad2deg(rref),'k:','LineWidth',1.2);
  plot(to,rad2deg(thP),'Color',[0 0.45 0.74],'LineWidth',1.4);
  plot(to,rad2deg(thL),'--','Color',[0.85 0.33 0.10],'LineWidth',1.4);
  ylabel('\theta [deg]'); legend({'\theta_{ref}','PID','LSTM'},'Location','best');
  title(sprintf('%s [%s] — RMSE PID %.3f° / LSTM %.3f° | e_{ss} LSTM %.3f° | desvio %.3f°', ...
    strrep(cen{c}.nome,'_','\_'), cen{c}.grp, rad2deg(rmse_pid), rad2deg(rmse_lstm), rad2deg(ess_lstm), rad2deg(desvio)));
  subplot(2,1,2); hold on; grid on;
  plot(to,dcP,'Color',[0 0.45 0.74],'LineWidth',1.2);
  plot(to,dcL,'--','Color',[0.85 0.33 0.10],'LineWidth',1.2);
  ylabel('dc [rad]'); xlabel('t [s]'); legend({'PID','LSTM'},'Location','best');
  exportgraphics(f,fullfile(res_dir,sprintf('robustez_%s.png',cen{c}.nome)),'BackgroundColor','white','Resolution',150);
  close(f);
end
set(0,'DefaultFigureVisible',prev);
x0_din = [0;0]; %#ok<NASGU>  restaura default
for m = {'pitch_PID','pitch_LSTM'}
  try, if strcmp(get_param(m{1},'FastRestart'),'on'), set_param(m{1},'FastRestart','off'); end, catch, end
  try, close_system(m{1},0); catch, end
end

%% --- resumo por grupo ---
fprintf('\n=== RESUMO ROBUSTEZ ===\n');
for g = unique(string({res.grp}))
  m = string({res.grp})==g;
  fprintf('%-12s: RMSE mediana PID %.3f / LSTM %.3f deg | desvio mediano %.3f deg | todos estaveis: %d\n', ...
    g, rad2deg(median([res(m).rmse_pid])), rad2deg(median([res(m).rmse_lstm])), ...
    rad2deg(median([res(m).desvio])), all([res(m).estavel]));
end
save(fullfile(res_dir,'robustez.mat'),'res');
fprintf('Figuras e robustez.mat salvos em %s\n', res_dir);
