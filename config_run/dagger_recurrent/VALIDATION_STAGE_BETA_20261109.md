# Ověření distilace neúplného teachera (9. 11. 2026)

Nový režim `stage_sawtooth` nepodmiňuje distilaci úspěchem celé šestifázové
úlohy. Všechny čtyři train konfigurace cílí na dokončení stage 4 včetně.
Podrobná pravidla a nastavení jsou v [README](README.md).

## Automatické testy

Ve finální regresní sadě prošlo všech **91 testů**. Z toho je 13 nových testů
v `test_stage_beta`. Regrese zahrnuje model, sekvenční replay, fine-tuning,
checkpointy, starý beta regulátor, metriky, škálování a checkery. Fixture
staršího testu randomizace dostala chybějící `task.randomization_seed` a CPU
device, které produkční handler už poskytuje. Log místního ověření je
`/tmp/dagger_stage_final_regression.log`.

```bash
python -m unittest config_run.dagger_recurrent.test_stage_beta
```

Nové kontroly ověřují:

- přesnou délku warmupu a sklon 1 → 0 za 1000 kroků;
- jednu korekci při více špatných stages, hranici tolerance a obnovu poklesu;
- ignorování nedostatečných, nedokončených a neznámých měření;
- nemožnost zablokovat betu na 1 opakovaným prodlužováním pauzy;
- kopii a neměnnost reference, obnovení křivky včetně rozběhnuté pauzy;
- ukončení po stage 4, původní šestifázový režim a omezení curriculum;
- obnovení dočasných eval parametrů i při výjimce;
- skutečný runner s odděleným testovacím prostředím: stage probes nikdy
  neresetují tréninkové prostředí, po resume se reference znovu neměří;
- komunikaci se skutečně spuštěným procesem: aktuální váhy, přenos chyb
  a ukončení pracovníka;
- stejná nastavení regulátoru ve všech čtyřech train konfiguracích.

## Skutečný GPU běh

Na RTX 3060 proběhl produkční runner se skutečnými šesti PPO politikami
z vybraného teacher bundle, předtrénovaným ResNetem a kamerou 640×360:
2 tréninkové envy + 2 envy v samostatném eval procesu. Trénink dokončil
32 kroků, obě gradientní aktualizace a uložil finální checkpoint. Warmup
teachera byl pro tuto zkoušku zkrácen na 2 kroky, warmup ResNetu na 1 update,
minimum pokusů na 2 a vnější limit eval na 300 kroků. Produkční YAML má
nadále 5000 / 1000 / 30 / 6000.

- Finální beta: 0,97, přesně dle 30 kroků poklesu po zkráceném warmupu.
- ResNet se po prvním updatu odemkl; LR policy 1e-4, ResNet 1e-5.
- Nenulové gradienty: ResNet 0,8700, GRU 0,2971, proprio 0,1300, hlavy 0,3710.
- Naměřená saturace vstupu GRU i hidden state: 0.
- Teacher prefix v krátkém limitu: 0/2, oba pokusy nedokončené. Trénink
  přesto pokračoval. Stages 1 a 2: teacher 2/2; stage 4: 0/2 přirozená
  selhání; stages 0 a 3 v limitu nedokončené, tedy neznámé. Jde o velmi malý
  vzorek a zkrácené evaluace, nikoli odhad produkční kvality teachera.
- V posledním běhu žádná hlášení Nyx `ASSERT_FAILURE` z kolize singletonů.
  Původní varianta s oběma scénami v jednom procesu tato hlášení vyvolala;
  nahrazena je izolovaným eval procesem.
- Po poslední úpravě záchranného checkpointu prošlo ještě všech 19 cílených
  testů (`test_stage_beta` + `test_training_audit`). Ověřují i přerušení
  evaluace po dokončeném tréninkovém kroku a obnovitelný stav bety.

Malý výpis s přesnými hodnotami je v
[validation_stage_beta_results.json](validation_stage_beta_results.json).
Místní úplný log: `/tmp/dagger_stage_gpu_isolated.log`, checkpointy:
`/tmp/dagger_stage_gpu_isolated/recurrent/`. Diagnostický spouštěč použitý
při ověřování: `/tmp/stage_sawtooth_smoke.py`.

Pro opakování se stejnými parametry je přidaný spouštěč (zvol nový výstupní
adresář; běh vyžaduje GPU, Nyx a lokální teacherovy váhy):

```bash
QD_DEVICE_MEMORY_FRACTION=0.15 python -m config_run.dagger_recurrent.verify_stage_training --output /tmp/dagger_stage_repeat
```

Eval proces navíc potřebuje vlastní GPU kontext, kopii modelů a simulaci.
Limit paměti Quadrants z proměnných prostředí platí v každém procesu zvlášť.
U 400-env varianty je potřeba počítat s touto režií vedle CPU replay.

## Omezení důkazu

Testy dokazují uvedené vlastnosti implementace. Úspěch studentského průchodu
0–4 musí změřit prefixová evaluace po dostatečně dlouhém tréninku. Samostatné
stage probes začínají ze snapshotů, proto jejich úspěch sám nestačí. Běh
400 tréninkových + 32 eval prostředí na L40 zde nebyl proveden.
