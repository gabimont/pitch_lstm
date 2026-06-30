function [theta_ref_vec, meta] = gerar_perfis(tipo, t, seed, a_max)
% GERAR_PERFIS  Gera um perfil de referencia de pitch para o datagen.
%
%   [vec, meta] = gerar_perfis(tipo, t, seed)
%   [vec, meta] = gerar_perfis(tipo, t, seed, a_max)
%
%   tipo  : 'degraus' | 'doublet' | 'rampa' | 'multisseno'
%   t     : vetor de tempo [s] (coluna ou linha) -- define a duracao T
%   seed  : semente do rng (reprodutibilidade do episodio)
%   a_max : (opcional) amplitude maxima |theta_ref| [rad]; default deg2rad(10)
%
%   theta_ref_vec : coluna, referencia ABSOLUTA de pitch [rad] (trim = 0)
%   meta          : struct com os parametros sorteados
%
% Adaptado de DH/lstm/datagen/gerar_perfil_theta_ref.m (mesmas familias de
% sinal), com trim = 0 e tempos RELATIVOS a T (cabem em qualquer duracao).
% Regras: primeiros T_HOLD s no zero; |theta_ref| <= a_max (clamp duro).

  t = t(:);
  T = t(end);
  rng(seed);

  if nargin < 4 || isempty(a_max), a_max = deg2rad(10); end
  A_MAX  = a_max;
  esc    = a_max / deg2rad(10);          % escala das amplitudes sorteadas
  T_HOLD = max(2, 0.10*T);               % hold inicial (acomodacao)

  delta = zeros(size(t));
  meta  = struct('tipo', tipo, 'seed', seed);

  switch lower(tipo)

    case 'degraus'
      % 2 a 4 degraus em instantes ordenados, espacamento minimo ~ T/6
      n_deg = randi([2 4]);
      tmin  = T_HOLD + 1;
      tmax  = T - 0.15*T;
      gmin  = max(2.5, (tmax - tmin) / (n_deg + 1));
      gaps  = gmin + (gmin) .* rand(1, n_deg);
      times = tmin + cumsum(gaps) - gaps(1);   % 1o degrau logo apos o hold
      times = min(times, tmax);
      levels = (2*rand(1, n_deg) - 1) * A_MAX;
      for k = 1:n_deg
        delta(t >= times(k)) = levels(k);
      end
      meta.n_deg = n_deg; meta.times = times; meta.levels = levels;

    case 'doublet'
      % pulso simetrico +A / -A, retorna a zero
      A  = esc * deg2rad(2 + 8*rand);
      s  = sign(rand - 0.5); if s == 0, s = 1; end
      Td = max(1.2, 0.10*T) + 0.08*T*rand;
      t0 = T_HOLD + 1 + (0.5*T - T_HOLD - 1 - 2*Td) * rand;
      t0 = max(t0, T_HOLD + 1);
      delta(t >= t0      & t < t0 +   Td) =  s*A;
      delta(t >= t0 + Td & t < t0 + 2*Td) = -s*A;
      meta.A = A; meta.t0 = t0; meta.Td = Td; meta.sinal = s;

    case 'rampa'
      % sobe em T_sub, plato, desce em T_desc -- orcamento relativo a T
      A    = sign(rand - 0.5) * esc * deg2rad(2 + 8*rand);
      t0   = T_HOLD + 1;
      budg = (T - 0.10*T) - t0;              % tempo disponivel apos o hold
      f    = [1+rand, 1+rand, 1+rand];       % proporcoes sub/plato/desc
      f    = f / sum(f) * budg;
      T_sub = f(1); T_plato = f(2); T_desc = f(3);
      t1 = t0 + T_sub; t2 = t1 + T_plato; t3 = t2 + T_desc;
      idx = t >= t0 & t < t1;  delta(idx) = A .* (t(idx) - t0) / T_sub;
      idx = t >= t1 & t < t2;  delta(idx) = A;
      idx = t >= t2 & t < t3;  delta(idx) = A .* (1 - (t(idx) - t2) / T_desc);
      meta.A = A; meta.t0 = t0; meta.T_sub = T_sub;
      meta.T_plato = T_plato; meta.T_desc = T_desc;

    case 'multisseno'
      % soma de 3-5 senoides (0.05-0.5 Hz), entrada suave em 2 s
      n_c    = randi([3 5]);
      f      = 0.05 + 0.45*rand(1, n_c);
      ph     = 2*pi*rand(1, n_c);
      amp    = 0.3 + 0.7*rand(1, n_c);
      A_alvo = esc * deg2rad(2 + 8*rand);
      sig = zeros(size(t));
      for k = 1:n_c
        sig = sig + amp(k) * sin(2*pi*f(k)*(t - T_HOLD) + ph(k));
      end
      sig = sig / max(abs(sig)) * A_alvo;
      w   = min(1, max(0, (t - T_HOLD)/2));     % rampa de entrada
      delta = w .* sig;
      meta.n_c = n_c; meta.f = f; meta.fase = ph; meta.A_alvo = A_alvo;

    otherwise
      error('gerar_perfis: tipo desconhecido ''%s''', tipo);
  end

  % seguranca: clamp duro no limite de projeto
  delta = max(-A_MAX, min(A_MAX, delta));

  theta_ref_vec = delta;            % trim = 0 -> referencia absoluta = delta
  meta.A_MAX = A_MAX; meta.T_HOLD = T_HOLD;

end
