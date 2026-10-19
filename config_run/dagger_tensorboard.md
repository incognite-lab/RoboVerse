# TensorBoard pro chairman_separate DAgger

Spuštění na clusteru z kořene repozitáře:

```bash
python config_run/main_separate.py chairman_separate/train_dagger_cluster_l40
tensorboard --logdir config_run/output/dagger_separate_cluster_l40_200/tensorboard
```

Každé spuštění včetně resume má samostatný podadresář `run_<UTC>_<pid>`.
Osa Step nových grafů je počet dokončených **vektorových** kroků (při resume
navazuje na checkpoint). Počítadla s `_total` a `_this_session` začínají v novém
běhu od nuly. Staré grafy `DAgger/*` zůstávají dostupné ve svém původním formátu.
Konfiguraci a význam metrik najdeš také na kartě Text pod `Run/*`.

## Stages/stage_0 až stage_5

| Metrika | Význam |
|---|---|
| `completed_total`, `completed_recent` | Počet skutečně pozorovaných dokončení stage celkem / od posledního zápisu. Jedno prostředí může dokončit stage opakovaně. |
| `completed_env_slots_total` | Kolik různých fyzických env slotů už tuto stage alespoň jednou dokončilo; nejvýše `num_envs`. |
| `entered_total` | Počet vstupů do stage, včetně curriculum startů a opakovaných resetů. |
| `active_envs` | Aktuální počet prostředí v této stage po resetech. |
| `failed_total`, `failed_recent` | Ukončené pokusy bez úspěšného dokončení této stage, včetně timeoutů. |
| `success_rate_total`, `success_rate_recent` | Dokončení / (dokončení + neúspěšná ukončení); rozpracované pokusy nejsou ve jmenovateli. |
| `completion_steps_mean` | Průměrná délka úspěšného průchodu od vstupu do stage. |
| `completion_sim_seconds_mean` | Stejná délka v simulačním čase podle skutečného `scene.dt × decimation`. |
| `completion_wall_seconds_mean` | Průměrný reálný čas od vstupu do úspěšného dokončení; zahrnuje i čekání na rendering a učení. |
| `completion_steps_recent_p50/p95` | Medián a 95. percentil posledních nejvýše 256 dokončení stage. |
| `active_age_steps_mean` | Jak dlouho v průměru běží aktuálně rozpracované pokusy. |

Start ze snapshotu stage 3 nepřičítá dokončení stages 0–2. Čas se měří od
aktuálního resetu/vstupu, nezahrnuje historii před vytvořením snapshotu.
Délky úspěšných průchodů se začnou zobrazovat až při prvním skutečném dokončení;
podíl úspěšnosti až po prvním ukončeném pokusu. Nedefinované hodnoty se nenahrazují
nulou. `recent` u počtů a úspěšnosti znamená interval od posledního zápisu,
u percentilů jde o omezenou historii 256 dokončení.

## Další důležité skupiny

- **Episodes/**: dokončené epizody/resetované pokusy, úspěšné konce, nedávná
  úspěšnost a délky v krocích, simulačních a reálných sekundách.
- **Student/**: loss za interval, MSE/MAE shody studenta s teacherem na právě
  navštívených stavech, podíl studentských akcí mimo limity před clippingem,
  beta, skutečný podíl kroků řízených teacherem a learning rate.
- **Buffer/**: velikost, kapacita, zaplnění a alokovaná velikost v MiB.
- **Failures/**: kumulativní příčiny neúspěšných ukončení, například pád,
  stage timeout nebo posun židle. Jeden pokus může mít více příčin.
- **Progress/**: kroky, přechody, uložené příklady, gradientní aktualizace
  a počet vzorků použitých při učení za tento běh.

Úspěšnost během DAggeru zahrnuje **směs teacher/student a curriculum starty**.
Ani úspěšný konec ze snapshotu stage 5 sám o sobě nedokazuje, že student umí
celou úlohu od stage 0. MSE shody s teacherem doplňuje průběžnou informaci o
studentovi; výslednou kvalitu je třeba posuzovat samostatnou studentskou evaluací.

## Rychlost a hardware

- **Speed/vector_steps_per_second**: kroky celé dávky/s za interval.
- **Speed/transitions_per_second**: kroky/s × počet prostředí.
- **Speed/stored_samples_per_second**: nově uložené příklady/s.
- **Speed/gradient_updates_per_second**, **trained_samples_per_second**:
  průchodnost učení; omezení dávky velikostí zatím naplněného bufferu je zohledněné.
- **Timing/**: průměrné ms na vektorový krok a podíl času pro kameru a přípravu
  vstupů, inference/akce, ukládání do bufferu, fyzikální krok s checkery a resety,
  gradientní aktualizace, logging a checkpointy. Zahrnuje interval i bez učení.
- **Time/**: uběhlý reálný čas od zahájení smyčky, simulační čas na prostředí,
  poměr simulačního/reálného času a průběžný odhad zbývajícího času. ETA není
  záruka dokončení před Slurm limitem.
- **GPU/**: vzorkované vytížení GPU a paměťového řadiče, celková použitá a
  dostupná kapacita VRAM zařízení, příkon a teplota; dále aktuální alokace,
  rezervace a maximální alokace PyTorch tohoto procesu.
- **Host/**: RSS RAM procesu a využití CPU procesu. CPU procenta jsou vůči
  jednomu jádru a při práci více vláken mohou přesáhnout 100 %.

Časy fází jsou wall time bez dodatečných `torch.cuda.synchronize()`, takže při
asynchronní práci se část čekání může účtovat až následující fázi. Součet fází
nemusí zahrnout veškerou obsluhu smyčky; hlavní ukazatel výkonu je celková
průchodnost. První načtení teachera, alokace bufferu a inicializace prostředí
nejsou zahrnuté do rychlosti trénovací smyčky.

GPU údaje čte pomocné vlákno přes `nvidia-smi`, s výběrem UUID skutečného CUDA
zařízení, aby `CUDA_VISIBLE_DEVICES` ve Slurmu nezaměnilo GPU 0 za jinou kartu.
Dotaz má timeout a neběží v trénovacím vlákně. `monitor_available` hlásí dostupnost,
`sample_age_seconds` stáří posledního vzorku. Nepodporované hodnoty se vynechají;
bez dostupného driverového monitoru zůstávají metriky PyTorch. Driverové údaje
jsou za celé zařízení a zahrnují další procesy. Paměťový řadič utilization
není procento zaplnění VRAM.

V obou trénovacích YAML:

```yaml
dagger_log_every: 100          # nové metriky každých 100 vektorových kroků
dagger_gpu_poll_seconds: 10    # GPU vzorek na pozadí každých 10 s
```

Na konci tréninku se zapíše i poslední kratší interval. Pro rychlejší náhled
lze nastavit `dagger_log_every: 20`; hardware zůstává vzorkovaný zvlášť.
Nevykreslují se další obrazy ani video jen kvůli monitoringu.

Ověření: prošlo 12 cílených testů (události stages, curriculum starty, opakované
úspěchy stejného slotu, finální stage, timeouty, časování a okna statistik,
parsování GPU údajů a existující DAgger/kamerové testy). Proběhly také dva
45krokové tréninky se skutečným teacherem: 100 env / batch 512 a L40 config
s 50 env / batch 1024, na lokální RTX 3060 se zmenšeným bufferem. Zapsané event
soubory byly znovu načteny; poslední běh obsahoval 130 skalárních tagů včetně
GPU vytížení, rollout MSE a správných počtů 2 250 uložených a 10 240 učicích
vzorků. Zápisy proběhly na krocích 20, 40 a v závěrečném krátkém intervalu 45.
L40S samotná tímto ověřena nebyla.

## Lineární beta

Nový výchozí průběh je `beta = max(0, 1 - step / total_timesteps)`:

```yaml
total_timesteps: 1_000_000
beta_schedule: linear
beta_start: 1.0
beta_end: 0.0
```

Na kroku 0 je beta 1, na 200 000 je 0,8, na 500 000 je 0,5 a po dokončení
1 000 000 kroků přesně 0. `beta_decay` už lineární průběh neovlivňuje;
exponenciální průběh vyžaduje výslovné `beta_schedule: exponential`.

`total_timesteps` je nyní absolutní konečný vektorový krok i při resume,
nikoli počet dalších kroků. Checkpoint na 200 000 a cíl 1 000 000 tedy znamená
800 000 dalších kroků a obnovenou betu 0,8. Beta se přepočítává i při načtení
starého exponenciálního checkpointu, zatímco váhy a optimizer se obnovují
beze změny. Pokud se změní cílový horizont, beta se řídí novým horizontem.

Lokální train config má cíl 1 000 000. L40 config si zachovává dosavadní
rozpočet 250 000 vektorových kroků × 200 env, a beta tedy končí na nule při
250 000. Nastavení `total_timesteps: 1_000_000` v tomto configu posune konec
bety na milion, ale také zvýší trénovací rozpočet na 200 milionů přechodů.
Automatické škálování počtu env zachovává průběh podle množství dat přepočtem
cílového počtu kroků. Krátký profiler drží betu na 1, aby jeho zkrácení na
201 kroků nenahradilo produkční průběh nevytrénovaným studentským rolloutem.
