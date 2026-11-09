# Oprava a měření rekurentního DAggeru — 9. 11. 2026

Tato zpráva zachycuje první opravu a tehdejší měření. Následné změny
předtrénování jsou popsány v [README](README.md); další nalezené chyby
a jejich reprodukce v [druhém auditu](AUDIT_20261109.md).

Oprava odstranila naměřený kolaps encoderu a GRU. Skutečné učení na datech
teachera, pokles chyby na oddělených datech a běh produkční pipeline byly
ověřeny na NVIDIA RTX 3060, při plném vstupu 640×360 a délce sekvence
16 učených kroků + 8 burn-in. **Dosažení úspěchu celého úkolu zatím prokázáno
není.** Krátký diagnostický student ani poslední teacher neměly v měřených
čtyřech epizodách úspěch.

## Provedené změny

- V2 ResNet-18 používá GroupNorm namísto zmrazeného, nekalibrovaného
  BatchNorm. Obrazový i proprioceptivní embedding před GRU používají LayerNorm.
  Sekvence a jednokroková inference mají stejné chování bez sdílení statistik
  mezi prostředími či časovými kroky.
- Nové tréninky používají AdamW, learning rate 1e-4 a nový výstupní adresář.
  Staré v1 checkpointy lze stále evaluovat; pokračování jejich tréninku je
  odmítnuto s instrukcí začít nový v2 model a optimizer.
- Replay uchovává časově souvislé teacherovy demonstrace v oddělených
  rezervoárech podle fází. Fáze 0 nemůže přepsat demonstrace pozdějších fází.
  Velmi krátké fáze se zachovají i mezi intervaly archivace; přiřazení fáze
  vychází ze všech učených kroků okna, nikoli pouze jeho posledního snímku.
- Část ukládaných prostředí zůstává trvale pod teacherem. Ostatní prostředí
  vybírají teachera nezávisle. Archiv přijímá pouze souvislá teacherova okna.
  Archiv se ukládá a obnovuje odděleně od krátkého online replaye.
- Počet aktualizací automaticky respektuje minimální poměr učených a nových
  vzorků. V konfiguraci 400/400, batch 16, délka 16 a interval 20 se provede
  63 aktualizací namísto původních 10; poměr je alespoň 2.
- Trénovací konfigurace povolují snapshot curriculum. Standardní lokální
  konfigurace ukládá 16 ze 80 prostředí, aby samotný online replay nezabral
  přibližně 26 GiB RAM. Explicitní konfigurace L40 400/400 zůstává 400/400;
  vyžaduje přibližně 93 GiB RAM pouze pro online replay, plus archiv a simulaci.
- Beta klesá až po úspěšných samostatných evaluacích studenta, po warmupu.
  Při opakovaném zhoršení se vrací výš. Smíšené epizody nejsou důkazem
  úspěšnosti samotného studenta.
- Evaluace používá předem přidělené kvóty prostředí; rychlá selhání nemohou
  vytlačit pomalejší úspěchy. Průběžné evaluace používají nejvýše jednu epizodu
  na prostředí. Po evaluaci se trénovací simulace a GRU explicitně resetují.
- Kontrolují se gradienty, saturace, nečíselné hodnoty, montáž/FOV kamery,
  geometrie scény, renderer a řídicí perioda. Checkpointy a archiv se zapisují
  přes dočasný soubor a atomické přejmenování.

## Naměřené výsledky

### Skutečné demonstrace teachera

Použit byl bundle
`ppo_models_separate/run_2026-10-14_19-21-47_chairman_separate`.
Zkontrolována byla metadata všech šesti `model_final.zip`: každý má
`num_timesteps = 5 000 000 000`. Je to pozdější checkpoint než poslední
číslovaný společný checkpoint 4 954 880 000.

Sběr proběhl na čtyřech prostředích po 1024 krocích. Prostředí 0 tvořilo
oddělená validační data; prostředí 1–3 sloužila pro učení. Model provedl
300 aktualizací na skutečných RGB, propriocepci a teacherových akcích.
Chyba před a po učení byla počítána na přesně stejných oknech.
V tomto malém sběru archiv trénovacích prostředí obsahoval fáze 1–3;
validační archiv obsahoval fáze 0–3. Napříč prostředími byly navštíveny fáze
0–4, fáze 5 nebyla navštívena. Test tedy nedokládá naučení všech fází;
zachování všech šesti kategorií a velmi krátkých fází ověřují regresní testy.

| Metrika | Před učením | Po 300 aktualizacích |
|---|---:|---:|
| MSE na pevných trénovacích oknech | 1,157644 | **0,217755** |
| MSE na oknech z odděleného prostředí | 0,971280 | **0,481422** |
| Pokles trénovací chyby | | **81,2 %** |
| Pokles validační chyby | | **50,4 %** |

Poslední naměřené normy gradientů byly: obrazový encoder 0,1181,
proprioceptivní encoder 0,5994, GRU 0,3739. Všechny tři skupiny vah se změnily.
RMS obrazového embeddingu byla 1,0013, maximum 4,3615; podíl vstupních
preaktivací GRU s absolutní hodnotou nad 10 byl **0 %**.

Pro srovnání: u původního checkpointu 185 000 dosahoval obrazový embedding
maxima přibližně 3,78 miliardy, encoder/propriocepce/GRU měly v diagnostickém
backwardu nulový gradient a jejich parametry se od kroku 64 000 nezměnily.

První kratší diagnostika se 100 aktualizacemi zlepšila pouze trénovací chybu,
zatímco validační chybu zhoršila. Tento negativní výsledek zůstává uložen;
není vydáván za úspěšný test generalizace. Větší test ověřil také směs
online replaye a archivu vyváženého podle dostupných fází. Krátké testy nejsou
řízenou studií jednotlivých hyperparametrů ani zárukou generalizace na jiné scény.

Výsledky a graf:

- [JSON většího měření](../output/dagger_verification_20261109_balanced/results.json)
- [Graf učení](../output/dagger_verification_20261109_balanced/learning_proof.png)
- [JSON prvního kratšího měření](../output/dagger_verification_20261109_startup/results.json)
- [Syntetický test při 640×360](../output/dagger_verification_20261109_synthetic/results.json)

### Produkční pipeline v reálné simulaci

Poslední krátký test provedl 64 vektorových kroků, 256 přechodů a 10 učicích
aktualizací. Uložil v2 studenta, optimizer i archiv demonstrací. Vstup měl
skutečně `[3, 360, 640]`, špička alokací PyTorch byla přibližně 836 MiB.
Tento údaj nezahrnuje veškerou paměť Genesis/Nyx ani desktopu.

- [Výsledky produkčního testu](../output/dagger_verification_20261109_pipeline_final/summary.json)
- [Studijní checkpoint](../output/dagger_verification_20261109_pipeline_final/student/recurrent/student_model_final.pth)

### Samostatná evaluace celého úkolu

| Politika | Úspěch celého úkolu | Výsledek |
|---|---:|---|
| Finální teacher 5 miliard vzorků | **0/4** | Všechny epizody dokončily fáze 0–2; tři selhaly ve fázi 3 a jedna ve fázi 4. |
| Student po 300 diagnostických aktualizacích | **0/4** | Všechny epizody skončily ve fázi 0. |

U obou politik byly všechny přidělené epizody skutečně dokončeny; nešlo
o doplnění nedoběhlých pokusů při vyčerpání limitu evaluace.

Pokles MSE tedy dokládá opravený mechanismus učení, **nikoli hotovou politiku**.
300 aktualizací na krátkém sběru nenahrazuje plný DAgger trénink. Navíc je
potřeba nejprve vyřešit spolehlivost teachera v současném prostředí. Není
potvrzeno, že dnešní prostředí přesně odpovídá prostředí jeho původního tréninku.
Už před touto opravou byly v pracovním stromu změny URDF a resetu fáze 0;
tato práce je nepřepisovala.

Standardní konfigurace proto před učením měří teachera a při nedostatečné
úspěšnosti trénink zastaví. Práh je `dagger_teacher_min_success: 0.01`;
nastavení 0 dovolí experiment s částečně úspěšným teacherem, ale neřeší jeho
selhávání a není důkazem zvládnutí celého úkolu. Beta se při nedostatečné
samostatné úspěšnosti studenta nesnižuje pouze plynutím času.

## Regresní ověření

**50 testů OK.** Pokrývají učení obou senzorových větví a GRU, plné rozlišení,
shodu sekvence a streamingu, časové zarovnání replaye, resety, přetečení ringu,
uchování krátkých i pozdních fází, čistě teacherova okna, trvalé teacherovy
sloty při beta=0, obnovení archivu a optimizeru, evaluaci bez zkreslení rychlými
selháními, beta řízenou evaluací, odmítnutí starého kolabovaného resume,
kontrolu kamery a zastavení při selhávajícím baseline teacherovi.

```bash
python -m unittest config_run.dagger_recurrent.test_student config_run.dagger_recurrent.test_pipeline config_run.dagger_recurrent.test_sequence_integrity config_run.dagger_recurrent.test_learning metasim.test.test_dagger_metrics metasim.test.test_dagger_scaling metasim.test.test_dagger_beta_schedule
```

Reprodukce skutečného učicího testu do nového adresáře:

```bash
QD_DEVICE_MEMORY_FRACTION=0.25 python -m config_run.dagger_recurrent.verify_learning --output config_run/output/dagger_verification_repeat --steps 1024 --updates 300 --eval-steps 6000
```

Na této instalaci bylo potřeba spouštět Genesis před alokací modelu na GPU;
verifikační skript to dodržuje stejně jako produkční launcher. První pokusy
s opačným pořadím selhaly při inicializaci paměti Quadrants/CUDA, nikoli při
učení. Uvedený podíl paměti byl použit při úspěšném ověření.

Nový standardní trénink se spouští stávajícím příkazem:

```bash
python config_run/main_separate.py chairman_separate/train_dagger
```

Na místní RTX 3060 je menší varianta
`chairman_separate/train_dagger_rtx3060_profiled`; její dřívější rychlostní
měření neplatí automaticky pro novou architekturu v2. Pro cluster existuje
`train_dagger_cluster_l40`, případně explicitní RAM náročná varianta
`train_dagger_l40_400`. Začíná se novými vahami, nikoli resume původního
checkpointu 185 000.
