% init_pitch.m
% =============================================================
% Inicializacao do exemplo LINEAR de controle de pitch (Fig. 1 e 3).
% Fonte unica de parametros: planta, periodo de amostragem, e o PID
% de referencia (sintonizado com pidtune). Roda antes de gerar dados,
% treinar a LSTM e validar a malha fechada.
%
% Sistema (realimentacao unitaria, sensor k=1):
%   theta_ref -> (e = theta_ref - theta) -> [C(s)] -> G_elev -> G_din -> theta
%
%   Servo do elevador : G_elev(s) = de/dc   = -0.1 / (0.1 s + 1)
%   Dinamica aeronave : G_din(s)  = theta/de = -3  / (s^2 + 2 s + 5)
%
% A planta vista pelo controlador (de dc ate theta) e':
%   Gp(s) = G_elev*G_din = 0.3 / [(0.1 s + 1)(s^2 + 2 s + 5)]
%   3a ordem, estavel (polos -10 e -1+-2j), ganho DC = +0.06.
%
% Cria as variaveis usadas pelos modelos Simulink (pitch_PID, pitch_LSTM):
%   Ts, num_elev/den_elev, num_din/den_din, Kp, Ki, Kd, Tf, theta0
% e a struct CFG (resumo da configuracao).
%
% Uso:  init_pitch            (autossuficiente)
% =============================================================

%% --- planta (funcoes de transferencia da imagem) ---
num_elev = -0.1;   den_elev = [0.1 1];      % servo do elevador
num_din  = -3;     den_din  = [1 2 5];      % dinamica de pitch
G_elev = tf(num_elev, den_elev);
G_din  = tf(num_din,  den_din);
Gp     = G_elev * G_din;                     % dc -> theta (malha aberta)

%% --- amostragem ---
Ts      = 0.01;                  % [s] passo fixo / sample time (100 Hz)
theta0  = 0;                     % condicao inicial de theta [rad]
A_MAX   = deg2rad(10);           % |theta_ref| <= 10 deg

%% --- realizacao em espaco de estados de G_din (p/ condicao inicial) ---
% G_din e' implementado como bloco State-Space nos modelos (o bloco
% Transfer Fcn nao expoe condicao inicial). x0_din = estado inicial:
%   para theta(0)=theta0 e theta_dot(0)=0 -> x0_din = [Cdin; Cdin*Adin] \ [theta0; 0]
[Adin, Bdin, Cdin, Ddin] = tf2ss(num_din, den_din);
x0_din = [0; 0];                 % default: planta em repouso (theta(0)=0)

%% --- sintonia do controlador de referencia ---
% Projeta PID e PI; o alvo da imitacao deve ser FUNCAO de e(t) apenas
% (decisao: entrada da LSTM = so o erro). PID com derivada-no-erro e
% filtro Tf continua sendo funcao de e(t); PI e' o fallback se a
% derivada gerar "chutes" no degrau. A escolha vai em CTRL_TYPE.
wc = 2.0;                        % banda alvo ~ omega_n da dinamica (rad/s)
opt = pidtuneOptions('PhaseMargin', 60);

C_pid = pidtune(Gp, 'PID', wc, opt);
C_pi  = pidtune(Gp, 'PI',  wc, opt);

[KpP, KiP, KdP, TfP] = piddata(C_pid);
[KpI, KiI]           = piddata(C_pi);

%% --- escolha do controlador ---
% Default PI: nesta planta (bem amortecida, ze~0.45) o PI ja rastreia com
% erro de regime zero e da um dc(t) SUAVE e funcao apenas de e(t) -- alvo
% ideal p/ imitacao com entrada e-only. O termo derivativo do PID injeta
% acao de alta frequencia (Kd/Tf grande) sensivel ao atraso de amostragem
% da malha discreta, dificil de imitar e desnecessaria aqui. (Resultado
% medido: PI -> dc pico/regime=1.01 e OS=2.2%; PID -> dc com transitorio
% derivativo agressivo.) Trocar p/ 'PID' so' se o relatorio exigir.
CTRL_TYPE = 'PI';                % 'PI' ou 'PID'
switch CTRL_TYPE
  case 'PID'
    Kp = KpP; Ki = KiP; Kd = KdP; Tf = TfP;
    C  = C_pid;
  case 'PI'
    Kp = KpI; Ki = KiI; Kd = 0;  Tf = 1;     % Tf irrelevante com Kd=0
    C  = C_pi;
end
if Tf <= 0, Tf = 10*Ts; end      % bloco Discrete PID exige Tf > 0

%% --- malha fechada de referencia (continua, p/ sanidade) ---
T_cl = feedback(C*Gp, 1);        % theta_ref -> theta
info = stepinfo(T_cl);
dc_gain = dcgain(T_cl);          % deve ~ 1 (rastreio com erro de regime ~0)

fprintf('=== init_pitch ===\n');
fprintf('Planta Gp: ganho DC = %.4f | polos = %s\n', ...
        dcgain(Gp), mat2str(round(pole(Gp),3)'));
fprintf('Controlador %s: Kp=%.4f Ki=%.4f Kd=%.4f Tf=%.4g\n', ...
        CTRL_TYPE, Kp, Ki, Kd, Tf);
fprintf('Malha fechada: DCgain=%.4f | overshoot=%.2f%% | ts(2%%)=%.2f s | tr=%.2f s\n', ...
        dc_gain, info.Overshoot, info.SettlingTime, info.RiseTime);
fprintf('Estavel: %d (todos os polos no SPE)\n', isstable(T_cl));

%% --- config resumida (para salvar junto do banco/rede) ---
CFG = struct('Ts',Ts, 'A_MAX',A_MAX, 'theta0',theta0, ...
             'CTRL_TYPE',CTRL_TYPE, 'Kp',Kp, 'Ki',Ki, 'Kd',Kd, 'Tf',Tf, ...
             'num_elev',num_elev, 'den_elev',den_elev, ...
             'num_din',num_din,  'den_din',den_din, ...
             'wc',wc);
