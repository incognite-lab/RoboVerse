# Audit reward pipeline ChairMan multi — 2026-09-15

## Závěr

V aktuálním pracovním stromu se jednorázový bonus **+500 za dokončení stage 1
použije pro učení policy 1**. Bonus se neztrácí při přechodu do stage 2 ani při
okamžitém resetu v režimu `train_only`. Závažný problém je poměr průběžných
kladných rewardů vůči terminálnímu bonusu: maximalizace návratnosti může odměňovat
odkládání dokončení. To je konkrétní mechanismus možného propadu SR, nikoli
prokázaná příčina historického běhu bez jeho časových logů a tehdejšího kódu.

Audit nemění trénovací kód ani váhy rewardů. Přidává CPU regresní testy.

## Cesta od úspěchu k optimalizéru

1. `metasim/sim/env_wrapper.py:step` provede simulaci a zavolá checker.
2. `metasim/cfg/checkers/checkers.py:_ChairManChecker.check` uloží původní
   stage do `reward_stage`, zapíše `completed_stage_events = 1`, zvýší
   `actual_stage` na 2 a nastaví `completed_stages = 1`. Úspěšný přechod
   vyřadí z neúspěšných terminací, takže nedostane současně penalizaci selhání.
3. `metasim/wrapper/gym_vec_env.py:_calculate_rewards` dočasně přiřadí
   rewardům původní stage 1. Shaping poslední akce proto patří ke stage 1.
   Sčítá `weight * reward_fn(...)`; není tu clipping, normalizace ani násobení dt.
4. `ChairMan_multi.py:MultiPolicyStageCompletionReward` vrátí jedničku,
   spotřebuje pouze `completed_stages` a zachová samostatnou událost pro trainer.
   Jeho váha je 500. `ContinuousStageReward` není v aktivním seznamu rewardů.
5. `SB3_chairman_multi_env.py:torch_step` zkopíruje události před případným
   resetem. Odměna už je vypočtená; reset ji nenuluje.
6. `multi_ppo_trainer.py:_collect_rollout` směruje reward podle
   **stages_before**, které reálný wrapper vrací jako kopii. Do bufferu policy 1
   přidá celou odměnu včetně bonusu. Dokončení označí jako lokální terminál.
7. `RaggedStageRollout.finish` na terminálu počítá `advantage = reward - V(s)`
   a `return = reward`. Nepřenáší hodnotu další policy nebo resetovaného stavu.
   Do předchozích souvislých kroků stejné policy propaguje GAE.
8. `_update_ready_policies` předá batch do `_ppo_update`; advantages vstupují
   do PPO actor loss a returns do critic loss, poté proběhne `backward/step`.

Výjimky jsou záměrné: zmražené policies a policies mimo `train_stage` se neučí;
nedostatečný batch čeká v pending bufferu. `train_only` vybranou policy odmrazí
a vypne automatické zmražení. Běžný souběžný režim používá manifestové frozen flagy.

## Proč +500 nemusí stačit

Aktuální stage 1 má například tyto kladné složky za **každý řídicí krok**:

| Složka | Váha | Potřebuje dokončení stage? |
| --- | ---: | --- |
| StayNearAnchorReward | 6 | Ne; stačí malý XY drift pánve |
| WaistStraightReward | 2 | Ne; stačí neutrální pas |
| OpenGraspReward | 1 | Ne; otevřené prsty |
| ReachChairProgressReward | 2 | Ne; spojitá odměna za vzdálenost rukou |
| HandTargetStillnessReward | 5 | Ne; podmínky se liší od checkeru |
| HandOrientationProgressReward | 1 | Ne |
| PreciseHandTargetReward | 2 | Ne |

Váhy nejsou vždy maximem výsledné složky; například výškový bonus může
zvětšit některé výstupy. Zároveň se odečítají pohybové a jiné penalizace.

Při ověřeném `gamma = 0.995` je návratnost konstantního čistého rewardu r
po N krocích `r * (1 - gamma**N) / (1 - gamma)`. Ilustrační čistých +8 za krok
má nekonečnou hodnotu 1600. Již 100 kroků čekání s +8 a potom bonus +500
dá přibližně 934, zatímco bonus ihned dá 500. Jde o srovnání stejného
rozhodovacího bodu, nikoli součtů epizod různých počátečních stavů.

Obecně, je-li celkový terminální reward T, jeden krok odkladu se vyplatí,
když `r + gamma*T > T`, tedy `r > (1-gamma)*T`. Pro T přibližně 500
je hranice jen **2.5 bodu za krok**. Terminální shaping ji trochu zvýší.
Ani penalizace -50 až na vzdáleném timeoutu nemusí tuto motivaci vyrovnat.
Čistých +8 není naměřený reward dané policy; příklad ukazuje problém měřítka.

Úspěch navíc potřebuje současně obě ruce do 7 cm, quaternionovou chybu
`1-abs(dot) < 0.03`, rychlosti obou rukou pod 0.15 m/s a splnění ve dvou
po sobě jdoucích krocích, bez společné terminace a nepovoleného pohybu židle.
Velký shaping proto ještě neznamená úspěch. Konstanta timeoutu stage 1 je 800
referenčních kroků po 0.02 s = **16 s**; komentář „10 s“ je zastaralý.

## Dotrénování a interpretace SR

- `PPO.load` přebírá learning rate/schedule, entropy coefficient, KL limit,
  critic i stav optimalizéru z checkpointu. Pouhá změna `learning_rate` v YAML
  pro resume zde learning rate načteného modelu nezmění. Collector bere
  `gamma`, `gae_lambda`, délku rolloutu, batch size a počet epoch z nového YAML.
- Rewardy se při resume počítají z aktuálního zdrojového kódu. Checkpoint
  nezachovává jejich původní definici. Lokální diff například mění reach
  váhu 10 → 2, orientation 4 → 1, anchor 1 → 6 a precision oblast 10 → 5 cm.
  To dokládá změnu proti HEAD, nikoli přesné nastavení při historickém běhu.
  Načtený critic tak může být pro novou návratnost špatně kalibrovaný.
- Actor se při trénování neodmražené policy vzorkuje stochasticky; deterministic
  evaluace není přímo srovnatelná se SR při sběru trénovacích dat.
- Rolling SR používá poslední dokončené pokusy (výchozí okno 1000).
  Po resetu mohou nejprve skončit krátké úspěšné pokusy a dlouhé neúspěšné až
  později. Pokles této metriky sám o sobě nedokazuje zapomínání policy.
- Snapshot curriculum mění rozdělení počátečních stavů. `train_only` resetuje
  vybranou stage; souběžné trénování také přijímá stavy z předchozí policy.
- GAE používá gamma*lambda = 0.96515; přímý příspěvek terminálního bonusu
  do advantage 63 kroků před koncem je zhruba 10.7 %. Přes hranici rolloutu
  se budoucnost odhaduje criticem. To přenos bonusu oslabuje, ale neruší.

## Ověřený uložený běh

`output/ppo_models_multi/run_2026-09-10_13-39-34_chairmanmulti_concurrent`:

- Manifest: 5 000 010 000 globálních přechodů.
- Stage 1: 3 350 252 345 samples, 5209 updates, 4 413 880 attempts,
  **0 successes**, rolling SR 0, frozen false.
- `stage_1/model_final.zip`: gamma 0.995, lambda 0.97, learning rate 0.0003,
  ent_coef 0.002, target_kl 0.02, normalize_advantage true.
- ZIPové n_steps=2 a batch_size=2 jsou záměrný malý nepoužívaný SB3 buffer;
  skutečný sběr řídí vlastní trainer. Nejde o chybu délky rolloutu.
- Lokální TensorBoard soubor tohoto běhu není dostupný. Počátečních 99 %
  nelze z manifestu potvrdit; může jít o jiný checkpoint, evaluaci nebo resume.

## Doporučený směr opravy

Priorita je omezit kladný reward za pouhé setrvávání: anchor/waist/stillness
formulovat jako penalizace odchylky s nulou v cíli, případně použít potenciálové
shaping `gamma*Phi(s_next)-Phi(s)` s nulovým potenciálem na terminálu.
Samotné zvýšení bonusu vyžaduje porovnání s čistým per-step rewardem a gamma;
velký skok měřítka navíc zatíží načtený critic. Není zde slepě aplikován.

Pro další experiment zaznamenávat jednotlivé vážené složky, completion bonus,
returns/advantages zvlášť na úspěchu, příčiny terminace a SR na pevné sadě
snapshotů se stejným režimem vzorkování. Resume by měl explicitně podporovat
změnu learning rate a ukládat konfiguraci rewardů spolu s checkpointem.

## Testy a meze ověření

Nový `test_chairman_multi_reward_pipeline.py` ověřuje reálný checker a součet
rewardů se syntetickými predikáty úspěchu, reálný torch wrapper při resetu,
přidělení terminálních returns policy 1 v obou režimech a PPO update.
Používá malé CPU policies; neověřuje fyziku ani SR robota v Genesis.

Spuštění v dostupném prostředí:

```bash
MPLCONFIGDIR=/tmp/mpl-chairman /home/roboversepc/miniconda3/envs/roboverse_train/bin/python -m unittest config_run.test_chairman_multi_reward_pipeline config_run.test_multi_ppo_trainer config_run.test_SB3_chairman_multi_env -v
```

Samostatně prošly všechny 3 nové testy. Původní dvě unittest sady mají 16 testů,
z nichž 15 prošlo a jeden CUDA test byl přeskočen. Pytest v tomto prostředí
není instalován; funkční pytest testy nebyly vydávány za spuštěné unittestem.
