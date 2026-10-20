# ChairMan: jedna policy pro celou úlohu

`metasim/cfg/tasks/humanoidbench/ChairMan.py` obsahuje samostatné implementace
stejných aktivních pohybových rewardů jako upravený `ChairMan_multi.py`.
Neimportuje rewardy z multi ani separate. Tyto dva moduly se touto úpravou nemění.

## Jedna policy, různé cíle podle stage

| Stage | Aktivní cíl |
| --- | --- |
| 0 | Přijít, zastavit, udržovat výchozí paže a otevřené prsty; nehýbat židlí |
| 1 | Dosáhnout oběma rukama na cíle se správnou orientací, prsty otevřené |
| 2 | Zavřít prsty do cílového úchopu, zachovat polohu a orientaci rukou |
| 3 | Táhnout židli dozadu, zachovat úchop a zabránit skluzu rukou |
| 4 | Otevřít prsty, udržet ruce na cílech a vytaženou židli v klidu |
| 5 | Spustit paže, zachovat otevřené prsty a klid židle |

Stabilita těžiště nad pánví, natočení k židli a penalizace prudkých změn
řízení platí společně. `stage_reward_weights` mění složení rewardu podle
fáze úlohy; všechny fáze řídí stejná policy a její návratnost pokračuje přes
mezilehlé přechody. Nevznikají stage policies ani lokální terminály PPO.

## Váhy a completion bonus

Husté váhy jsou stejné jako v [přehledu multi](chairman_multi_adapted_rewards.md):
faktor 0.25 vůči výchozím vahám v separate YAML, navíc poloviční příspěvek
každé ruky u párových složek a společné členy židle započítané jednou.
Ve stages 0–2 je penalizace neúspěšného ukončení −10, ve stages 3–5 −250.
Jsou to výchozí váhy pro ladění, bez ověření úspěšnosti PPO ve fyzice.

Pro tuto architekturu je aktivní `StageCompletionReward` s váhou **500 ve
všech šesti stages**. Nezávisí na čísle aktuální stage: odmění i přechod
0 → 1, spotřebuje jednorázový completion flag a zachová událost checkeru
pro logování. Dokončení celé úlohy 5 → 6 také dostane +500.

`ContinuousStageReward` a původní `StageProgressCfg` nejsou v aktivním
seznamu. Policy tedy nedostává konstantní bonus jen za setrvání v pozdější
stage. Všechny základní `reward_weights` jsou nulové; aktivní váhy vybírá
`stage_reward_weights`. Na úspěšném přechodu se počítají podle stage před
akcí (`reward_stage`), poté se obnoví stage následujícího kroku. Historie
stavových rewardů se po přechodu resetuje.

Checker a single-policy SB3 wrapper nadále pokračují přes stages 0–4 a po
úspěšném dokončení stage 5 označí celou úlohu za dokončenou. Nastavení délky
epizody a resetovacího curriculum tato změna neupravuje.

## Ověření

Pět CPU testů kontroluje shodu pohybových výpočtů s multi včetně historie
a částečných resetů, registraci vah, skutečný checker se syntetickými stage
predikáty a skutečný součet rewardů ve wrapperu, penalizace selhání a
jednorázovost bonusu včetně stage 0 a finále. Fyzika a trénink PPO nejsou
součástí těchto testů.

```bash
MPLCONFIGDIR=/tmp/mpl-chairman /home/roboversepc/miniconda3/envs/roboverse_train/bin/python -m unittest metasim.test.test_chairman_single_adapted_rewards -q
```
