% gerar_figura_dagger.m
% =============================================================
% Figura "antes e depois do DAgger": mostra a rede de IMITACAO PURA (v0)
% entrando em ciclo-limite em malha fechada, contra o PID e contra a rede
% FINAL (v2, pos-DAgger), num degrau de +5 graus.
%
% Usa as redes guardadas: net_pitch_v0.mat (imitacao pura) e
% net_pitch_v2.mat (final). O bloco Stateful Predict le sempre
% 'net_pitch.mat'; aqui trocamos o conteudo desse arquivo temporariamente
% (restaurado por onCleanup ao final).
% =============================================================

script_dir = fileparts(mfilename('fullpath'));
root_dir   = fileparts(script_dir);
fig_dir    = fullfile(script_dir,'figuras');
addpath(root_dir);
run(fullfile(root_dir,'init_pitch.m'));

% garante que net_pitch.mat volte a ser a v2 ao final (mesmo se der erro)
restaura = onCleanup(@() copyfile(fullfile(root_dir,'net_pitch_v2.mat'), ...
                                  fullfile(root_dir,'net_pitch.mat')));

% cenario: degrau de +5 graus em t=2, planta partindo do repouso
T = 25; t = (0:Ts:T)';
theta_ref_ts = timeseries(deg2rad(5)*double(t>=2), t);
x0_din = [0;0];

% ---------- PID ----------
load_system(fullfile(root_dir,'pitch_PID.slx'));
set_param('pitch_PID','StopTime',num2str(T));
oP = sim('pitch_PID');

% ---------- LSTM v0 (imitacao pura) ----------
copyfile(fullfile(root_dir,'net_pitch_v0.mat'), fullfile(root_dir,'net_pitch.mat'));
nv = load(fullfile(root_dir,'treino','lstm_pitch_v0.mat'),'norm_io'); no = nv.norm_io;
nmx=no.mu_x; nsx=no.sg_x; nmy=no.mu_y; nsy=no.sg_y;
if bdIsLoaded('pitch_LSTM'), close_system('pitch_LSTM',0); end
load_system(fullfile(root_dir,'pitch_LSTM.slx'));
set_param('pitch_LSTM','StopTime',num2str(T));
oL0 = sim('pitch_LSTM');

% ---------- LSTM v2 (final, pos-DAgger) ----------
copyfile(fullfile(root_dir,'net_pitch_v2.mat'), fullfile(root_dir,'net_pitch.mat'));
nv = load(fullfile(root_dir,'treino','lstm_pitch.mat'),'norm_io'); no = nv.norm_io;
nmx=no.mu_x; nsx=no.sg_x; nmy=no.mu_y; nsy=no.sg_y;
close_system('pitch_LSTM',0);
load_system(fullfile(root_dir,'pitch_LSTM.slx'));
set_param('pitch_LSTM','StopTime',num2str(T));
oL2 = sim('pitch_LSTM');

% ---------- sinais ----------
to  = oP.theta.Time(:);
ref = rad2deg(oP.theta_ref.Data(:));
thP = rad2deg(oP.theta.Data(:));   dcP = oP.dc.Data(:);
th0 = rad2deg(oL0.theta.Data(:));  dc0 = oL0.dc.Data(:);
th2 = rad2deg(oL2.theta.Data(:));  dc2 = oL2.dc.Data(:);
fprintf('RMS rastreamento: PID %.3f | LSTM imitacao pura %.3f | LSTM pos-DAgger %.3f (deg)\n', ...
  rms(thP-ref), rms(th0-ref), rms(th2-ref));

% ---------- figura ----------
CP=[0 0.45 0.74]; C0=[0.85 0.10 0.10]; C2=[0.10 0.60 0.20];
f = figure('Color','w','Position',[80 80 950 620],'Visible','off');
try, f.Theme='light'; catch, end
subplot(2,1,1); hold on; grid on;
plot(to, ref,'k:','LineWidth',1.3);
plot(to, th0,'-','Color',C0,'LineWidth',1.5);
plot(to, thP,'-','Color',CP,'LineWidth',1.6);
plot(to, th2,'--','Color',C2,'LineWidth',1.6);
ylabel('\theta [graus]');
legend({'referência','LSTM só imitação (antes)','PID','LSTM + DAgger (depois)'}, 'Location','best');
title('Antes e depois do DAgger — degrau de +5^\circ');
subplot(2,1,2); hold on; grid on;
plot(to, dc0,'-','Color',C0,'LineWidth',1.3);
plot(to, dcP,'-','Color',CP,'LineWidth',1.3);
plot(to, dc2,'--','Color',C2,'LineWidth',1.3);
ylabel('comando \delta_c [rad]'); xlabel('t [s]');
legend({'LSTM só imitação','PID','LSTM + DAgger'}, 'Location','best');
exportgraphics(f, fullfile(fig_dir,'dagger_antes_depois.png'),'BackgroundColor','white','Resolution',200);
close(f);
close_system('pitch_LSTM',0); close_system('pitch_PID',0);
fprintf('Figura salva: %s\n', fullfile(fig_dir,'dagger_antes_depois.png'));
