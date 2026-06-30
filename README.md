# Controle de pitch (planta linear) — LSTM substituindo o PID

Exemplo didático e autocontido: substituir o controlador PID da malha de pitch
por um controlador **LSTM** treinado por imitação, e comparar os dois em malha
fechada. Tudo em **MATLAB/Simulink** (R2025b + Deep Learning Toolbox), sem Python.

**Repositório:** <https://github.com/gabimont/pitch_lstm>
**Relatório completo (PDF):** [`relatorio/relatorio.pdf`](relatorio/relatorio.pdf) — teoria,
metodologia, figuras do Simulink por dentro e análise de robustez.

## Sistema (das figuras do enunciado)

Malha fechada com realimentação unitária (sensor k = 1):

```
theta_ref --> (e = theta_ref - theta) --> [C] --> G_elev --> G_din --> theta
                       ^------------------------------------------------|
```

- Servo do elevador:  `G_elev(s) = de/dc   = -0.1 / (0.1 s + 1)`
- Dinâmica da aeronave: `G_din(s) = theta/de = -3  / (s^2 + 2 s + 5)`
- Planta vista pelo controlador: `Gp = G_elev*G_din = 0.3/[(0.1s+1)(s^2+2s+5)]`
  (3ª ordem, estável, ganho DC = +0.06).

O controlador de referência **C** é um **PI** sintonizado com `pidtune`
(`Kp=11.79, Ki=15.15`). Usa-se PI (não PID): a planta é bem amortecida e a derivada
injetaria ação de alta frequência difícil de imitar a partir só do erro.

## Metodologia

1. **Imitação seq2seq** — a LSTM recebe a sequência do erro `e(t)` e aprende a
   reproduzir o comando do controlador `dc(t)`. Episódio inteiro (não janela
   deslizante): o termo integral `Ki*∫e` depende do histórico desde `t=0`, e o
   estado oculto da LSTM acumula essa ação — mesma semântica do bloco *Stateful
   Predict* na malha fechada.
2. **DAgger** (2 rodadas) — a imitação pura tem fidelidade aberta alta
   (Pearson ~0.99) mas, em malha fechada, entrava em **ciclo-limite** (os erros
   levam a estados que o PID nunca visitou). O DAgger roda a LSTM em malha
   fechada, coleta os `e(t)` visitados, rotula com a ação **exata** do PID
   (`dc = Kp*e + Ki*∫e`) e reagrega ao banco. Isso elimina a oscilação.

## Arquivos e ordem de execução

| Passo | Script / modelo | O que faz |
|------|------------------|-----------|
| 0 | `init_pitch.m` | Define planta, `Ts=0.01`, sintoniza o PI (fonte única de parâmetros). |
| 1 | `pitch_PID.slx` | Malha fechada da Fig. 3 com o PI (entregável + gerador de dados). |
| 2 | `datagen/gerar_perfis.m`, `datagen/rodar_banco.m` | Gera 240 episódios (degraus, doublet, rampa, multisseno) → `banco_pitch.mat`. |
| 3 | `treino/treinar_lstm.m` | Treina a LSTM `seqInput(1)→lstm(64)→lstm(32)→fc(1)` → `lstm_pitch.mat` + `net_pitch.mat`. |
| 4 | `pitch_LSTM.slx` | Cópia do `pitch_PID` com o PI trocado por `norm → Stateful Predict → denorm`. |
| 5 | `treino/dagger.m` | DAgger: rollouts da LSTM + rótulo PID → `banco_pitch_aug.mat`, retreina. (rodar 2×) |
| 6 | `malha/rodar_comparacao.m` | PID vs LSTM em 7 cenários (envelope nominal) → métricas + figuras. |
| 7 | `malha/rodar_robustez.m` | Robustez: condições iniciais θ(0)≠0 e referências fora do treino → `robustez_*.png`. |

Recomeçar do zero: rode `init_pitch`, depois `rodar_banco`, `treinar_lstm`,
`dagger` (duas vezes) e `rodar_comparacao`.

**Teste rápido (interativo):** `testar_controladores.m` na raiz. Edite no topo a
inclinação inicial (`theta0_deg`) e a referência (`ref_deg`), rode, e veja PID e LSTM
**sobrepostos** (θ e comando) + as métricas no console. É o jeito mais fácil de
experimentar e de explorar os limites da rede (basta pôr `theta0_deg` alto ou
referência acima de ±10°).

## Resultado (rede final = pós-DAgger round 2)

A LSTM **empata com o PID** em todos os cenários (inclusive um perfil novo não
visto no treino):

- RMSE de rastreamento (mediana): **LSTM 0.914° vs PID 0.919°**
- Erro de regime (mediana): LSTM 0.13° (PID ~0)
- Desvio LSTM↔PID (mediana): 0.13°
- A oscilação em malha fechada da imitação pura foi **eliminada** pelo DAgger.

## Robustez e limites do método (`rodar_robustez.m`)

Testou-se a rede final FORA do envelope de treino (que era: planta partindo do
repouso e |θ_ref| ≤ 10°). Em **todos** os casos a malha permaneceu **estável**
(nunca divergiu), mas a qualidade degrada fora da distribuição dos dados:

- **Condições iniciais θ(0) ≠ 0** (planta começa deslocada): a LSTM é estável e
  chega ao regime correto (e_ss ≈ 0), mas com **transitório ruim** — partindo de
  θ₀=+8° ela oscila +16°/−22° antes de assentar, enquanto o PID volta suave. Causa:
  o estado oculto da LSTM parte do zero (*cold start*) e a combinação "estado oculto
  zerado + planta deslocada" nunca apareceu no treino (RMSE 2–3× o do PID).
- **Amplitude > ±10°** (extrapolação): o **ciclo-limite reaparece** (em +13° a θ
  oscila 10↔16,5°; em −15° sobra ~2,5° de erro). O DAgger só cobriu até ±10°.
- **Referência mais rápida** (0,4–1,2 Hz vs 0,05–0,5 Hz do treino): LSTM ≈ PID
  (desvio 0,09°); os dois atrasam igual.

**Lição:** a fronteira de competência do controlador aprendido = a fronteira dos
dados. Dentro do envelope de imitação+DAgger ele empata com o PID; fora dele,
degrada de forma previsível mas SEM instabilizar. Para estender o envelope bastaria
aumentar o datagen com θ(0) aleatório e amplitudes maiores (a planta já está
parametrizada em condição inicial via `x0_din`) e dar mais uma passada de DAgger.

## Notas técnicas

- Solver `ode4` passo fixo `Ts=0.01` (planta contínua + controlador discreto a 100 Hz).
- `G_din` é um bloco **State-Space** (não Transfer Fcn) p/ expor a condição inicial
  `x0_din` (`theta(0)=theta0` ⇒ `x0_din = [Cdin; Cdin*Adin] \ [theta0; 0]`); em
  `x0_din=[0;0]` a dinâmica é idêntica à TF original.
- Bloco *Stateful Predict* (`deeplib/Stateful Predict`): carrega `net_pitch.mat`,
  com **Force interpreted simulation = on** (necessário p/ simulações repetidas) e
  conversão `single→double` na saída (rede treina em `single`, malha é `double`).
- Versões da rede: `*_v0` (imitação pura, oscilava), `*_v1` (DAgger r1),
  `*_v2` (DAgger r2 = final). `net_pitch.mat`/`lstm_pitch.mat` = v2.
- **Não usar Fast Restart** alternando os dois modelos: o *Stateful Predict*
  reavalia a máscara a cada simulação.
