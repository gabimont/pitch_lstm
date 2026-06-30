% testar_controladores.m
% =============================================================
% TESTE INTERATIVO — PID vs LSTM na mesma planta.
%
% Edite o bloco "EDITE AQUI" (inclinacao inicial + referencia),
% rode este script no MATLAB (botao Run, ou digite: testar_controladores)
% e veja os graficos dos dois controladores SOBREPOSTOS.
%
% Nao precisa mexer em mais nada: o script carrega a planta, os ganhos
% do PID e a rede LSTM treinada automaticamente.
% =============================================================

%% ========================= EDITE AQUI =========================
theta0_deg = 10;      % inclinacao INICIAL do aviao, em graus (0 = parte nivelado)
T          = 40;     % duracao da simulacao, em segundos

% REFERENCIA desejada de inclinacao (graus) ao longo do tempo t.
% Descomente UM exemplo, ou escreva a sua propria funcao de t:
ref_deg = @(t) -5*(t>=2);                  % degrau: vai para 5 graus em t = 2 s
%ref_deg = @(t) 7*(t>=2) - 11*(t>=13);   % sobe +7 graus e depois desce para -4
%ref_deg = @(t) 8*sin(2*pi*0.1*t);       % oscilacao senoidal de +-8 graus
%ref_deg = @(t) 5*(t>=2 & t<13);         % pulso de 5 graus entre 2 s e 13 s
%% ==============================================================

% ---------- preparacao (nao precisa editar daqui pra baixo) ----------
set(0,'DefaultFigureVisible','on');   % garante que a figura apareca
this_dir = fileparts(mfilename('fullpath'));
run(fullfile(this_dir,'init_pitch.m'));              % planta, Ts, Kp, Ki, Adin...
Sn = load(fullfile(this_dir,'treino','lstm_pitch.mat'),'norm_io'); no = Sn.norm_io;
nmx = no.mu_x; nsx = no.sg_x; nmy = no.mu_y; nsy = no.sg_y;   % normalizacao da rede

t = (0:Ts:T)';
ref_vec = ref_deg(t);                                 % referencia em graus
theta_ref_ts = timeseries(deg2rad(ref_vec), t);       % lida pelos dois modelos
% estado inicial da planta tal que theta(0) = theta0_deg (com velocidade nula)
x0_din = [Cdin; Cdin*Adin] \ [deg2rad(theta0_deg); 0];

% aviso amigavel se sair do envelope de treino da rede (+-10 graus, partindo do repouso)
if max(abs(ref_vec)) > 10 || abs(theta0_deg) > 10
  warning(['Voce esta FORA do envelope de treino da LSTM (referencia ou inclinacao ' ...
           'inicial acima de +-10 graus). A rede pode degradar/oscilar -- e o teste de ' ...
           'robustez. O PID, por ser classico, nao tem esse limite.']);
end

% ---------- roda os dois modelos ----------
for m = {'pitch_PID','pitch_LSTM'}
  if ~bdIsLoaded(m{1}), load_system(fullfile(this_dir,[m{1} '.slx'])); end
  if strcmp(get_param(m{1},'FastRestart'),'on'), set_param(m{1},'FastRestart','off'); end
  set_param(m{1},'StopTime', num2str(T));
end
oP = sim('pitch_PID');     % controlador classico
oL = sim('pitch_LSTM');    % rede neural

to  = oP.theta.Time(:);
ref = rad2deg(oP.theta_ref.Data(:));
thP = rad2deg(oP.theta.Data(:));   dcP = oP.dc.Data(:);
thL = rad2deg(oL.theta.Data(:));   dcL = oL.dc.Data(:);

% ---------- metricas rapidas ----------
rmseP = rms(thP-ref);  rmseL = rms(thL-ref);
essP  = abs(thP(end)-ref(end));  essL = abs(thL(end)-ref(end));
desvio = rms(thL-thP);
fprintf('\n========= RESULTADO  (theta0 = %.1f deg, T = %g s) =========\n', theta0_deg, T);
fprintf('  Erro medio de rastreamento (RMS):  PID %.3f deg  |  LSTM %.3f deg\n', rmseP, rmseL);
fprintf('  Erro de regime (no fim):           PID %.3f deg  |  LSTM %.3f deg\n', essP, essL);
fprintf('  Diferenca entre LSTM e PID (RMS):  %.3f deg\n', desvio);
fprintf('===========================================================\n');

% ---------- grafico sobreposto ----------
f = figure('Color','w','Name','PID vs LSTM','Position',[80 80 950 600],'Visible','on');
try, f.Theme = 'light'; catch, end
subplot(2,1,1); hold on; grid on;
plot(to, ref, 'k:',  'LineWidth',1.3);
plot(to, thP, 'Color',[0 0.45 0.74],  'LineWidth',1.6);
plot(to, thL, '--', 'Color',[0.85 0.33 0.10], 'LineWidth',1.6);
ylabel('inclinacao \theta [graus]');
legend({'referencia','PID','LSTM'}, 'Location','best');
title(sprintf('\\theta(0) = %.1f^\\circ   |   RMS: PID %.2f^\\circ, LSTM %.2f^\\circ   |   desvio %.2f^\\circ', ...
      theta0_deg, rmseP, rmseL, desvio));
subplot(2,1,2); hold on; grid on;
plot(to, dcP, 'Color',[0 0.45 0.74],  'LineWidth',1.3);
plot(to, dcL, '--', 'Color',[0.85 0.33 0.10], 'LineWidth',1.3);
ylabel('comando \delta_c [rad]'); xlabel('t [s]');
legend({'PID','LSTM'}, 'Location','best');

figure(f); shg; drawnow;              % traz a figura para a frente e desenha
