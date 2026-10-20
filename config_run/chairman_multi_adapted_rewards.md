# Rewardy ChairMan multi odvozené ze separate

`metasim/cfg/tasks/humanoidbench/ChairMan_multi.py` používá stejné aktivní
výpočty pro pas, ruce, prsty a chůzi jako `ChairMan_separate.py`, ale jejich
příspěvky skládá podle stage policy, která provedla akci. Implementace jsou
samostatné; modul multi neimportuje separate. `ChairMan_separate.py` ani jeho
trénovací konfigurace se touto úpravou nemění.

## Převod vah

Výchozí husté váhy vycházejí z `configs/chairman_separate/train_ppo.yaml`:

- faktor 0.25 pro přechod od několika souběžných rewardů k jednomu součtu;
- u párových rewardů dostává každá ruka navíc faktor 0.5;
- společné rewardy židle se počítají jednou; pro `PullChairReward` se použije
  směrová váha 1.25 × 0.25, pro její klid −0.1 × 0.25;
- `PolicyDeltaAction` patří separate traineru. Multi používá existující
  `DeltaActionRateCfg` pro fyzické cíle horní části těla a
  `LocomotionCommandPenalty` pro příkazy chůze. Tyto penalizace nejsou numericky
  totožné s lokální penalizací akcí každé separate policy.

Jde o výchozí nastavení pro ladění, nikoli váhy ověřené tréninkem v simulaci.
Původní completion bonus +500 zůstává. Selhání má váhu −10 ve stages 0–2
a −250 ve stages 3–5.

V každé stage platí společné váhy:

| Reward | Váha |
| --- | ---: |
| CenteredUpperBodyCOMReward | 0.25 |
| FaceChairAllStagesReward | 0.075 |
| PelvisFacingChairReward | 0.0375 |
| WaistVelocityPenalty | −0.05 |
| DeltaActionRateCfg | −0.05 |
| LocomotionCommandPenalty | −0.025 |
| MultiPolicyStageCompletionReward | 500 |

Doplňkové váhy; „L/R“ znamená uvedenou váhu **pro každou** ruku, pomlčka nulový příspěvek:

| Reward | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ApproachAndStandReward | 0.5 | 0.5 | 0.5 | — | — | — |
| L/R Stage0ArmPoseReward | 0.125 | — | — | — | — | — |
| L/R OpenFingersReward | 0.0625 | 0.0625 | — | — | — | — |
| L/R HandDistanceReward | — | 0.15 | 0.15 | 0.15 | 0.15 | — |
| L/R HandOrientationReward | — | 0.1 | 0.1 | 0.1 | 0.1 | — |
| L/R ArmVelocityPenalty | — | −0.0125 | −0.0125 | −0.0125 | — | — |
| L/R Stage2FingerJointPositionReward | — | — | 0.125 | 0.125 | — | — |
| L/R Stage3HandDriftPenalty | — | — | — | −0.0125 | — | — |
| L/R ReleaseFingersReward | — | — | — | — | 0.125 | — |
| StayNearAnchorReward | — | — | 0.0375 | — | 0.0375 | — |
| Stage3ReverseVelocityReward | — | — | — | 0.5 | — | — |
| PullChairReward | — | — | — | 0.3125 | — | — |
| PulledChairStillnessReward | — | — | — | — | −0.025 | −0.025 |
| KeepChairStillPenaltyWalk | −0.25 | — | — | — | — | — |
| KeepChairStillPenaltyArms | — | −0.25 | — | — | — | — |
| ArmDownReward | — | — | — | — | — | 0.25 |
| KeepFingersOpenPenalty | — | — | — | — | — | −0.0125 |

Stage 5 zachovává původní aktivní cíl multi pro spuštění paží. Separate v jeho
aktivním YAML nemá odpovídající arm-lowering reward.

## Přechody stages a ověření

Průběžné `Waist/Arm/Finger/DirectionStageProgressReward` nejsou aktivní:
v jedné stage mají konstantní hodnotu a odměňovaly by odkládání přechodu.
Úspěch odměňuje jednorázový `MultiPolicyStageCompletionReward`. Wrapper
vybírá váhy podle `reward_stage`, tedy stage před akcí i na úspěšném přechodu.
Nedefinované příspěvky jsou nulové díky nulovým základním `reward_weights`.

CPU testy porovnávají výstupy s implementací separate v šesti stages včetně
pohybu a částečných resetů historie. Další test používá skutečnou metodu
`MetaSimVecEnv._calculate_rewards` a ověřuje výběr vah, jednorázový bonus
a zachování události pro trainer. Testy neověřují fyziku ani úspěšnost PPO.

```bash
MPLCONFIGDIR=/tmp/mpl-chairman /home/roboversepc/miniconda3/envs/roboverse_train/bin/python -m unittest metasim.test.test_chairman_multi_adapted_rewards metasim.test.test_chairman_multi_simple_rewards metasim.test.test_chairman_multi_com_reward metasim.test.test_chairman_stage2_finger_reward -q
```

Výsledek: všech 16 testů výše prošlo. Rozšířená sada
`metasim.test.test_chairman_stage2_checker` má jeden existující neúspěšný test
`test_stage3_reuses_pose_goal_without_requiring_contact`; jeho selhání bylo
reprodukováno i s původním modulem `ChairMan_multi.py` z HEAD. Checker tato
úprava nemění.
