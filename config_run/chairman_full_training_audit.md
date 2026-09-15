# Audit celého ChairMan Multi-PPO tréninku — 2026-09-15

> Historický nález před úpravami. Implementované opravy, nové váhy a zbývající
> omezení popisuje [souhrn změn](chairman_training_changes.md).

## Závěr

Aktuální konfiguraci nelze označit za připravenou na spolehlivé naučení celé
sekvence. Přenos rewardu do správné policy a základní PPO/GAE mechanismus
fungují, ale existují konkrétní konflikty cílů rewardů a curriculum.
Tento audit nemění produkční kód ani uživatelovy váhy.

Kontrolováno: `configs/chairman_multi/train_ppo.yaml`, `main_multi.py`,
`multi_ppo_trainer.py`, oba SB3 Chairman wrappery, `gym_vec_env.py`,
`env_wrapper.py`, `_ChairManChecker`, `stages_chairman.py`, `ChairMan_multi.py`
a posuvné klouby modelu židle. CPU scénáře neověřují dynamickou dosažitelnost
pohybu v Genesis ani výslednou success rate.

## Aktuální konfigurace

- Čerstvé modely: `train_or_eval: train`, `train_only: false`.
- 15 000 prostředí, 64 kroků/rollout, až 960 000 vzorků za globální rollout.
- Minimum pro update stage 131 072, minibatch 16 384, 4 epochy.
- Gamma 0.995, lambda 0.97, LR 0.0003, entropy coefficient 0.002.
- Přenos actorů a zmrazování úspěšných policies zapnuté.
- Snapshot curriculum implicitně zapnuté; disk obsahuje 79 snapshotů stage 1,
  10 snapshotů stage 2, žádné pro stages 3–5.
- Bonus +500, **aktuální penalizace selhání -1**.
- Stage-1 vzdálenost pro úspěch 5 cm; sdílený drift limit stages 2/3 je 25 cm.
  Některé komentáře stále uvádějí staré tolerance 7/10 cm.

## 1. Vysoká priorita: diskové snapshoty obejdou dědění actorů

`stages_chairman.py:init_ram_buffer` automaticky načte starší snapshoty.
`reset_chairman` vybírá stage rovnoměrně od 0 do poslední souvisle dostupné.
V současném čerstvém běhu proto již první reset může obsadit stages 0, 1 a 2.

`MultiPPOTrainer._initialize_stage_actor` při nulových datech předchůdce
nastaví cíl na `independent/source_has_no_data` a označí inicializaci hotovou.
Po pozdějším naučení předchůdce se přenos neopakuje. To je konflikt nově
přidaného přenosu se starým diskovým curriculum, nikoli ztráta checkpointu.

CPU reprodukce: zavolat inicializaci stage 1 s nulovými čítači, pak dát source
10 000 samples a 5 updates a inicializaci zavolat znovu. Metadata zůstávají
`independent/source_has_no_data`.

**Oprava:** odemykání snapshotových stages svázat s aktuálním během a stavem
actorů. Čerstvý souběžný trénink má začít ve stage 0; staré snapshoty nesmějí
odemknout dosud neinicializovanou downstream policy. Pro resume odemknout
stages podle skutečně načtených modelů. Samotné nastavení počátečního resetu
na stage 0 nestačí, pokud následné resety opět zpřístupní diskové stages.

## 2. Vysoká priorita: stage 0 odměňuje čekání mimo úspěšnou oblast

Stage0ArmPos dává až +3 za krok a FaceChairReward při stabilním správném
natočení +1.4. To jsou kladné odměny i při nulovém postupu k židli.

Numerický scénář se správnými cíli kloubů, nulovými rychlostmi a správným
natočením (celý agregovaný reward, nikoli jen jednotlivé složky):

- 16 cm od finálního bodu: **+4.48716/krok**, mimo toleranci úspěchu 15 cm.
- 14 cm od finálního bodu: **+4.69395/krok**, uvnitř poziční tolerance.

Tyto hodnoty samy nedokazují fyzickou stabilitu pózy, ale dokazují rewardový
motiv čekat. Pro terminální návratnost T je krok odkladu výhodný při
`r_wait > (1-gamma)*T`. Pro T kolem 500 je hranice přibližně 2.5.
Stage 0 ji překračuje i těsně před vstupem do úspěšné oblasti.

**Oprava:** stage-0 rozpočet kladného shapingu stáhnout pod diskontní cenu
odkladu bonusu s rezervou. Převést držení paží/orientace na přiměřené
penalizace odchylky nebo správně terminované potenciálové shaping.
Nezvyšovat pouze bonus bez kontroly ostatních stages a criticu.

## 3. Vysoká priorita: stages 4/5 mají výhodný stav těsně před úspěchem

ReleaseFingersReward i ArmDownReward obsahují `0.20 * goal_score` a váhu 14.
To je až +2.8 za každý krok klidného čekání. Ostatní penalizace mohou být
v ideálním statickém stavu nulové.

CPU scénáře s židlí v cíli, klidným robotem a nulovými rychlostmi:

| Stage | Stav mimo úspěch | Celý průběžný reward | Výhoda kroku odkladu* |
| --- | --- | ---: | ---: |
| 4 | Úhly prstů 0.151 rad, limit <0.15 | +2.799628 | +0.285628 |
| 5 | Úhly paží 0.351 rad, limit <0.35 | +2.799791 | +0.285791 |

*Srovnání s terminálním rewardem 502.8: `r_wait + 0.995*502.8 - 502.8`.
Přibližně správný neúspěšný stav tak získává vyšší návratnost odkladem.

**Oprava:** zmenšit per-step goal složku nebo celkové stage-4/5 váhy;
vyhradit hlavní signál pokroku a terminálnímu bonusu. Testovat statické stavy
těsně na neúspěšné straně prahů, ne pouze perfektní cílovou pózu.

## 4. Vysoká priorita: stage 3 má silnou motivaci k časnému selhání

PullChairReward při stojící židli na začátku tahu vrací po váze 16 přibližně
**-7.1090 za krok**. Další negativní složky (úchop, drift, pohybové penalizace)
jej mohou dále snížit. Současná jednorázová penalizace selhání je pouze -1.

Policy přenesená ze stage 2 umí zejména držet a svírat. Než objeví užitečné
tahání, setrvávání v tomto chování je velmi drahé. Ztráta kontaktu ukončuje
stage 3 ihned. Při podobném rewardu terminálního kroku je selhání levnější
než pokračování v dlouhé neúspěšné trajektorii.

To je důvod pro riziko učení předčasného selhání, nikoli důkaz, že jej každá
policy skutečně objeví. Bonus +500 za dostupný úspěch může preference změnit.

**Oprava:** zmenšit průběžné penalizace stage 3, začít s dosažitelným krátkým
tahem, prodlužovat vzdálenost a postupně zavést nárok na souvislý pohyb.
Penalizaci selhání kalibrovat vůči kumulovaným nákladům, ne izolovaně.
Zmenšení -50 na -1 sice opticky zlepší mean reward, ale oslabí motivaci
vyhnout se selhání.

## 5. Další rizika v přechodech a train loopu

- **Přenos po prvním vzácném úspěchu:** podmínka zdroje kontroluje samples,
  nikoli mastery. Přenos může nastat před prvním PPO updatem zdroje. Přenos
  není chyba sám o sobě (odpovídá požadovanému prvnímu přechodu), ale nesmí
  být interpretován jako převzetí spolehlivé dovednosti.
- **Přenesené log_std:** velmi malý naučený průzkum může bránit zavírání
  prstů ve stage 2 či otevírání ve stage 4. Ověřit a případně zavést minimum
  průzkumu na relevantních kloubech; actor mean přitom lze zachovat.
- **Okamžité failure ve stages 3–5:** stage 3 ukončí i jednokrokovou ztrátu
  kontaktu jedné ruky; stages 4/5 ukončí při normě 3D rychlosti pánve nad
  0.2 m/s. Vertikální pohyb při balancování je zahrnut, zatímco stage 0
  používá XY. Fyzická dosažitelnost stabilního navazujícího stavu není CPU
  testy prokázána. Zvážit hysterézi/krátkou toleranci po měření příčin resetů.
- **Zamrazení podle starších dat:** learn() nejprve updatuje policy a potom
  ji může zmrazit podle úspěchů před updatem. Zároveň jsou nejdříve dostupné
  výsledky krátkých pokusů, nikoli dlouhých neúspěšných. Vhodnější je potvrdit
  výkon nové policy více okny nebo evaluací na pevné sadě stavů.
- **Globální časový limit:** při task dt 0.01 s je 2500 kroků 25 s, zatímco
  součet stage timeoutů je 44 s. Není to důkaz nemožnosti rychlého úspěchu,
  ale pomalá úspěšná sekvence může skončit globálním timeoutem. Sladit limity
  s cílem evaluace celé úlohy a měřit dobu stage i zbytek fyzické epizody.
- **Stage 5 nekontroluje otevřené prsty v success predicate:** jejich zavření
  má penalizaci, ale arms-down success jej nezakazuje. Pokud je otevřená ruka
  požadavek finálního stavu, musí jej obsahovat také checker.

## Co kontrolou prošlo

- Reward je přidělen podle stage před akcí; úspěch jde odcházející policy.
- GAE se přeruší na dokončení stage a nebootstrapuje z další policy/resetu.
- Pending stage batch nečeká přes update téže policy: při updatu se bere celý.
- Akce jsou pro PPO uloženy před clippingem; výpočet log-prob odpovídá sběru.
- Přenos actoru kopíruje nezávislé tensory a remapuje stage one-hot; critic
  a stav optimalizéru se nepřebírají ze zdroje. Metadata přežijí resume.
- Stage 1 má horní mez kladného shapingu 1.525 a stage 2 přibližně 2.02,
  tedy pod základní cenou 2.5 za odložení +500 při gamma 0.995. To eliminuje
  konkrétní problém měřítka, nikoli všechny možné lokální optima.
- `fix_base_link=True` u židle není zákaz tahu: URDF obsahuje posuvné
  `floor_slide_x/y` klouby mezi pevnou kotvou a pohyblivou židlí.
- Parametry 15k prostředí skutečně dávají rozsáhlé rollouts; dřívější problém
  čtyř prostředí čekajících stovky rolloutů na update zde obecně neplatí.
  Pozdější stage s malým obsazením může přesto čekat na dost dat.

## Validace a pořadí oprav

Spuštěno 34 CPU unittest testů: **32 prošlo, 1 přeskočen (CUDA), 1 selhal**.
Selhání `test_terminal_bonus_and_failure_penalty_survive_reweighting`:
test očekává -50, aktuální konfigurace dává -1. Není to ztráta rewardu po
cestě k PPO. Dále byly provedeny uvedené numerické kontrapříklady.

Priorita před dlouhým během: (1) odemykání stages versus dědění actorů,
(2) anti-waiting rozpočet stage 0/4/5, (3) stage-3 náklady a dosažitelnost
prvního tahu, (4) diagnostika jednotlivých vážených rewardů a důvodů selhání,
(5) teprve potom delší běh a tuning PPO hyperparametrů.

Bez simulace nelze slíbit naučení kontaktů, stabilního tahu ani výslednou SR.
Úspěšné unit testy mají potvrdit kontrakty pipeline, ne nahrazovat evaluaci
celé sekvence na nezávislých počátečních stavech.
