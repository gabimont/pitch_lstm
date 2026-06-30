% gerar_figuras.m
% =============================================================
% Gera as figuras do relatorio (fundo branco, qualidade de impressao)
% em relatorio/figuras/:
%   planta_malha_aberta.png   — resposta ao degrau da planta em malha aberta
%   malha_fechada_pid.png     — rastreamento com o controlador PID
%   malha_fechada_lstm.png    — rastreamento com o controlador LSTM (vs PID)
%   rob_nominal.png           — robustez: dentro do envelope (LSTM ~ PID)
%   rob_cond_inicial.png      — robustez: condicao inicial theta(0)!=0
%   rob_extrapolacao.png      — robustez: amplitude > +-10 deg (ciclo-limite)
%   rob_ref_rapida.png        — robustez: referencia mais rapida que o treino
%
% Uso:  gerar_figuras        (autossuficiente)
% =============================================================

script_dir = fileparts(mfilename('fullpath'));
root_dir   = fileparts(script_dir);
fig_dir    = fullfile(script_dir,'figuras');
addpath(root_dir, fullfile(root_dir,'datagen'));
if ~exist(fig_dir,'dir'), mkdir(fig_dir); end

run(fullfile(root_dir,'init_pitch.m'));
Sn = load(fullfile(root_dir,'treino','lstm_pitch.mat'),'norm_io'); no=Sn.norm_io;
assignin('base','nmx',no.mu_x); assignin('base','nsx',no.sg_x);
assignin('base','nmy',no.mu_y); assignin('base','nsy',no.sg_y);

%% ===== Figuras dos DADOS DE TREINAMENTO (a partir do banco) =====
Sb = load(fullfile(root_dir,'datagen','banco_pitch.mat'),'episodios');
eps_b = Sb.episodios;
tipos = {'degraus','doublet','rampa','multisseno'};
nomes = {'Degraus','Doublet','Rampa','Multisseno'};

% (i) variedade de theta_ref: 7 amostras reais de cada tipo, sobrepostas
fd = figure('Color','w','Position',[80 80 960 640],'Visible','off');
try, fd.Theme = 'light'; catch, end
for it = 1:4
  subplot(2,2,it); hold on; grid on;
  idx = find(arrayfun(@(e) strcmp(e.tipo, tipos{it}), eps_b));
  idx = idx(1:min(7,numel(idx)));
  for k = idx(:)'
    plot(eps_b(k).t, rad2deg(eps_b(k).theta_ref), 'LineWidth',1.0);
  end
  yline(10,'k--'); yline(-10,'k--');
  ylim([-12 12]); xlabel('t [s]'); ylabel('\theta_{ref} [graus]'); title(nomes{it});
end
sgtitle('Amostras reais de \theta_{ref} usadas para gerar o banco (limite de \pm10^\circ tracejado)');
exportgraphics(fd, fullfile(fig_dir,'dados_perfis.png'),'BackgroundColor','white','Resolution',200);
close(fd);

% (ii) par entrada->alvo de um episodio (degraus): theta, erro e, comando dc
ep = eps_b(find(arrayfun(@(e) strcmp(e.tipo,'degraus'), eps_b), 1, 'last'));
fd = figure('Color','w','Position',[80 80 950 700],'Visible','off');
try, fd.Theme = 'light'; catch, end
subplot(3,1,1); hold on; grid on;
plot(ep.t, rad2deg(ep.theta_ref),'k:','LineWidth',1.3);
plot(ep.t, rad2deg(ep.theta),'Color',[0 0.45 0.74],'LineWidth',1.5);
ylabel('\theta [graus]'); legend({'\theta_{ref}','\theta (resposta)'},'Location','best');
title('Um episodio: o PID controla a planta e gravamos os sinais a 100 Hz');
subplot(3,1,2); grid on;
plot(ep.t, rad2deg(ep.e),'Color',[0.20 0.60 0.25],'LineWidth',1.3);
ylabel('erro e [graus]'); title('ENTRADA da rede:  e(t) = \theta_{ref} - \theta');
subplot(3,1,3); grid on;
plot(ep.t, ep.dc,'Color',[0.85 0.33 0.10],'LineWidth',1.3);
ylabel('comando \delta_c [rad]'); xlabel('t [s]'); title('ALVO da rede:  \delta_c(t) dado pelo PID');
exportgraphics(fd, fullfile(fig_dir,'dados_par.png'),'BackgroundColor','white','Resolution',200);
close(fd);

% modelos carregados uma vez
for m = {'pitch_PID','pitch_LSTM'}
  if ~bdIsLoaded(m{1}), load_system(fullfile(root_dir,[m{1} '.slx'])); end
  try, if strcmp(get_param(m{1},'FastRestart'),'on'), set_param(m{1},'FastRestart','off'); end, catch, end
end

CL_REF=[0 0 0]; CL_PID=[0 0.45 0.74]; CL_LSTM=[0.85 0.33 0.10];
ic = @(th0deg) [Cdin; Cdin*Adin] \ [deg2rad(th0deg); 0];

%% ===== F1: planta em malha aberta (degrau em dc -> theta) =====
to = (0:Ts:6)'; u = ones(size(to));
y  = lsim(Gp, u, to);                  % resposta ao degrau unitario de dc
f = figure('Color','w','Position',[100 100 900 380],'Visible','off');
try, f.Theme = 'light'; catch, end
plot(to, y, 'Color',CL_PID, 'LineWidth',1.8); hold on; grid on;
yline(dcgain(Gp),'k--','ganho DC = 0,06','LabelHorizontalAlignment','left','FontSize',10);
xlabel('t [s]'); ylabel('\theta [rad]');
title('Planta em malha aberta — resposta ao degrau unitario de \delta_c (polos: -10 e -1\pm2j)');
exportgraphics(f, fullfile(fig_dir,'planta_malha_aberta.png'), 'BackgroundColor','white','Resolution',200);
close(f);

%% ===== F1b: planta em MALHA FECHADA com o PI (degrau) =====
Tpi=18; Npi=round(Tpi/Ts)+1; tpi=(0:Npi-1)'*Ts;
rPI = simula('pitch_PID', deg2rad(8)*double(tpi>=2), tpi, [0;0]);
fig_cmp(fullfile(fig_dir,'malha_fechada_pi.png'), ...
  'Planta em malha fechada com o controlador PI — degrau de +8^\circ', ...
  rPI.t, rPI.ref, rPI.th, rPI.dc, [], [], CL_REF, CL_PID, CL_LSTM);

%% ===== referencia comum p/ F2 e F3 (dois degraus, dentro do envelope) =====
T=24; N=round(T/Ts)+1; tp=(0:N-1)'*Ts;
ref = deg2rad(7)*double(tp>=2) + deg2rad(-11)*double(tp>=13);   % +7deg, depois -4deg
ref = max(min(ref,deg2rad(10)),-deg2rad(10));
rP = simula('pitch_PID',  ref, tp, [0;0]);
rL = simula('pitch_LSTM', ref, tp, [0;0]);

%% ===== F2: malha fechada com PID =====
fig_cmp(fullfile(fig_dir,'malha_fechada_pid.png'), ...
  'Malha fechada com PID — rastreamento de \theta_{ref}', ...
  rP.t, rP.ref, rP.th, rP.dc, [], [], CL_REF, CL_PID, CL_LSTM);

%% ===== F3: malha fechada com LSTM (PID em cinza p/ referencia) =====
fig_cmp(fullfile(fig_dir,'malha_fechada_lstm.png'), ...
  'Malha fechada com LSTM — rastreamento de \theta_{ref} (PID em cinza)', ...
  rL.t, rL.ref, rP.th, rP.dc, rL.th, rL.dc, CL_REF, [0.6 0.6 0.6], CL_LSTM);

%% ===== F4: robustez — nominal (dentro do envelope) =====
T=25; N=round(T/Ts)+1; tp=(0:N-1)'*Ts;
ref = deg2rad(5)*double(tp>=2);
rP=simula('pitch_PID',ref,tp,[0;0]); rL=simula('pitch_LSTM',ref,tp,[0;0]);
fig_cmp(fullfile(fig_dir,'rob_nominal.png'), ...
  'Robustez — dentro do envelope (degrau +5°): LSTM \approx PID', ...
  rP.t, rP.ref, rP.th, rP.dc, rL.th, rL.dc, CL_REF, CL_PID, CL_LSTM);

%% ===== F5: robustez — condicao inicial theta(0)=+8 deg, theta_ref=0 =====
ref = zeros(N,1);
rP=simula('pitch_PID',ref,tp,ic(8)); rL=simula('pitch_LSTM',ref,tp,ic(8));
fig_cmp(fullfile(fig_dir,'rob_cond_inicial.png'), ...
  'Robustez — condicao inicial \theta(0)=+8° (regular a 0): transitorio ruim da LSTM', ...
  rP.t, rP.ref, rP.th, rP.dc, rL.th, rL.dc, CL_REF, CL_PID, CL_LSTM);

%% ===== F6: robustez — extrapolacao de amplitude (+13 deg) =====
ref = deg2rad(13)*double(tp>=2);
rP=simula('pitch_PID',ref,tp,[0;0]); rL=simula('pitch_LSTM',ref,tp,[0;0]);
fig_cmp(fullfile(fig_dir,'rob_extrapolacao.png'), ...
  'Robustez — amplitude +13° (> ±10° do treino): ciclo-limite da LSTM', ...
  rP.t, rP.ref, rP.th, rP.dc, rL.th, rL.dc, CL_REF, CL_PID, CL_LSTM);

%% ===== F7: robustez — referencia mais rapida que o treino =====
rng(4242); nc=4; fr=0.4+0.8*rand(1,nc); ph=2*pi*rand(1,nc); am=0.3+0.7*rand(1,nc);
sig=zeros(N,1); for k=1:nc, sig=sig+am(k)*sin(2*pi*fr(k)*(tp-3)+ph(k)); end
sig=sig/max(abs(sig))*deg2rad(8); w=min(1,max(0,(tp-3)/2)); ref=w.*sig;
rP=simula('pitch_PID',ref,tp,[0;0]); rL=simula('pitch_LSTM',ref,tp,[0;0]);
fig_cmp(fullfile(fig_dir,'rob_ref_rapida.png'), ...
  'Robustez — referencia mais rapida (0,4–1,2 Hz): LSTM \approx PID (ambos atrasam)', ...
  rP.t, rP.ref, rP.th, rP.dc, rL.th, rL.dc, CL_REF, CL_PID, CL_LSTM);

%% --- limpeza ---
assignin('base','x0_din',[0;0]);
for m={'pitch_PID','pitch_LSTM'}, try, close_system(m{1},0); catch, end, end
fprintf('Figuras geradas em %s\n', fig_dir);

% =============================================================
function r = simula(model, refvec, tp, x0)
  assignin('base','theta_ref_ts', timeseries(refvec, tp));
  assignin('base','x0_din', x0);
  set_param(model,'StopTime', num2str(tp(end)));
  o = sim(model);
  r.t = o.theta.Time(:); r.th = o.theta.Data(:);
  r.dc = o.dc.Data(:);   r.ref = o.theta_ref.Data(:);
end

function fig_cmp(fname, tit, t, ref, thP, dcP, thL, dcL, cR, cP, cL)
  f = figure('Color','w','Position',[100 100 900 560],'Visible','off');
  try, f.Theme = 'light'; catch, end
  subplot(2,1,1); hold on; grid on;
  plot(t, rad2deg(ref), ':', 'Color',cR, 'LineWidth',1.3);
  leg = {'\theta_{ref}'};
  if ~isempty(thP), plot(t, rad2deg(thP), 'Color',cP, 'LineWidth',1.6); leg{end+1}='PID'; end
  if ~isempty(thL), plot(t, rad2deg(thL), '--','Color',cL, 'LineWidth',1.6); leg{end+1}='LSTM'; end
  ylabel('\theta [graus]'); legend(leg,'Location','best'); title(tit);
  subplot(2,1,2); hold on; grid on;
  if ~isempty(dcP), plot(t, dcP, 'Color',cP, 'LineWidth',1.3); end
  if ~isempty(dcL), plot(t, dcL, '--','Color',cL, 'LineWidth',1.3); end
  ylabel('\delta_c [rad]'); xlabel('t [s]');
  exportgraphics(f, fname, 'BackgroundColor','white','Resolution',200);
  close(f);
end
