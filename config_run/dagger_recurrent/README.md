# Rekurentní student pro ChairMan separate

Samostatný model, sekvenční replay, trainer a evaluace. V `main_separate.py`
se přepínají pouze přes `train_or_eval`; původní režimy stále používají
`separate_dagger.py` a `dagger_vp/student_net.py`.

Kamera Nyx podporuje samostatné `fov_horizontal` a `fov_vertical`; konfigurace,
vztah k deployment cropu a interní renderování jsou popsané v [camera_fov.md](../camera_fov.md).

## Spuštění přes stávající konfigurace

| `train_or_eval` | Chování |
|---|---|
| `train_dagger_new` | Nový ResNet-18 + GRU student; šest PPO politik jako teacher. |
| `load_and_train_dagger_new` | Obnova nového studenta a optimizeru z `load_student_path`. |
| `eval_dagger_new` | Studentská evaluace checkpointů, CSV a tři grafy. |
| `eval_dagger_new_video` | Studentská evaluace jednoho checkpointu s RGB videem. |
| `train_dagger`, `load_and_train_dagger`, `eval_dagger`, `eval_dagger_video` | Původní student a původní chování. |

Použij například existující `chairman_separate/train_dagger.yaml`, změň jeho
`train_or_eval` na `train_dagger_new` a spusť:

```bash
python config_run/main_separate.py chairman_separate/train_dagger
```

Povinné nastavení navíc není potřeba. Teacher zůstává v `expert_model_path`,
`policies` nadále popisují šest teacherových politik, kamera v `cameras`.
Pro evaluaci zvol nový režim ve stávajícím eval configu. Musí používat stejnou
kameru (geometrii, montáž, FOV, rozlišení) a řídicí periodu jako trénink.
U v2 se kontrolují rozlišení, perioda, montáž, FOV i nastavení scény.
Montáž a FOV se automaticky nepřepisují; neshoda vyvolá chybu.

Nové váhy se ukládají do **`model_save_path/recurrent/`**. Tím samotné přepnutí
režimu nepřepíše starého studenta. Pokud cesta už končí `recurrent`, další
podsložka nevzniká. Stejně tak evaluace při `load_model_path` ukazujícím na
rodičovský adresář automaticky vybere jeho `recurrent/`; lze zadat i přímo
konkrétní nový `.pth`. Starý `.pth` není kompatibilní a vyvolá jasnou chybu.
Video z adresáře vybere `student_model_final.pth`, případně poslední číslovaný
checkpoint. Výsledky evaluace mají názvy `eval_dagger_new_results.csv` a
`eval_dagger_new_{mean_reward,success_rate,mean_length}.png`.

Checkpoint nového studenta je slovník s vahami, parametry architektury,
názvy/pořadím kloubů a akcí, konvencí IMU, řídicí periodou a velikostí RGB.
Optimizer a RNG se ukládají vedle do `.pth.training.pth`. Resume obnoví
váhy/optimizer a absolutní krok. U `beta_schedule: evaluation` se beta obnoví
z checkpointu a další snížení čeká na nový warmup a evaluace.
**Simulace, online replay, předchozí akce a rollout paměť začínají znovu**;
nejde o bitově přesné pokračování rozehrané epizody. Buffer se znovu naplní
před prvním gradientním updatem. Pro resume ponech `expert_model_path` na
teacher bundle a nastav `load_student_path` na nový studentský checkpoint.
Archiv `teacher_demonstrations.pth` se obnovuje vedle checkpointu.
Nové váhy a optimizer mají společné ID generace; při nesouladu se resume
odmítne ještě před načtením optimizeru, i když mají stejné číslo kroku.
Při chybě před prvním pokrokem se existující modely nepřepisují. Přerušený
běh s pokrokem uloží `student_model_interrupted_step_N.pth` a případný vlastní
archiv `.pth.demonstrations.pth`. Resume z této cesty vybere jeho archiv
automaticky; běžná adresářová evaluace vybírá standardní checkpointy.
Staré checkpointy `chairman_recurrent_v1` lze evaluovat, ale nelze z nich
pokračovat v novém tréninku: jejich encoder/GRU mohou být saturované.

### Volitelné nové parametry

```yaml
# Není nutné přidávat: tyto hodnoty jsou výchozí.
dagger_new_sequence_length: 16      # kroky se započteným lossem
dagger_new_burn_in: 8              # předcházející kroky bez gradientu
dagger_new_batch_size: 8           # počet sekvencí, nikoli počet snímků
dagger_new_hidden_size: 256
dagger_new_num_layers: 1
dagger_new_proprio_embedding_size: 128
dagger_new_grad_clip: 1.0
dagger_new_pretrained_visual: false
dagger_new_finetune_visual: true   # výchozí pro nový běh bez tohoto klíče
dagger_new_buffer_device: cpu      # záměrně nezávislé na legacy dagger_buffer_device
dagger_new_seed: 0
```

Přesný default batch size je `min(8, max(1, dagger_batch_size // sequence_length))`;
u dosavadního `dagger_batch_size: 512` tedy 8 sekvencí po 16 učených krocích.
Počet aktualizací/frekvence, kapacita, počet ukládaných envs, learning rate,
beta, TensorBoard a ukládání používají stávající klíče `dagger_*`.
`dagger_buffer_steps` musí pojmout burn-in + učenou sekvenci (standardně 24).
Pokud `dagger_store_per_step` převyšuje aktuální `num_envs` (např. při ladění
s jedním robotem), nový trainer jej automaticky omezí na `num_envs` a vypíše
upozornění. Nulové/záporné hodnoty zůstávají chybou.
`dagger_buffer_pin_memory` se používá jen pro CPU replay. Sekvenční ResNet
potřebuje více výpočtů než původní CNN; neměň význam starého batch size na
512 sekvencí. Pro omezení RAM lze zmenšit `dagger_buffer_steps` (např. 512);
samotné RGB při aktuálních 512 krocích, 8 envs a 640×360 zabírá přibližně 2,64 GiB.
TensorBoard má vlastní podsložku `recurrent`; velikost replay včetně paměti
GRU se loguje jako `Recurrent/buffer_total_mib`.

## Stabilní trénink v2 a ověření

Standardní konfigurace používají nový adresář `dagger_separate_stable_v2`
(pro L40 `dagger_separate_l40_stable_v2_640x360`). Začni přes
`train_dagger_new`, nikoli obnovením vah starého zkolabovaného modelu.
Teacher se načítá z manifestu; `model_final.zip` má v použitém bundle
5 000 000 000 trénovacích vzorků a je pozdější než číslované checkpointy.

- `dagger_demonstrations_per_stage: 16`: CPU reservoir pro každou z šesti
  fází; délka okna je burn-in + sequence. Archiv obsahuje celé souvislé
  sekvence a stejné reset masky jako online replay, nikoli náhodné snímky.
  Okna mají krok `sequence_length`, takže se burn-in části překrývají:
  po prvním burn-in nejsou mezi učenými částmi oken časové mezery.
- `dagger_demonstration_fraction: 0.5`: přibližně polovina učicích sekvencí
  pochází z archivu; dostupné fáze se vybírají rovnoměrně.
- `dagger_teacher_anchor_fraction: 0.25`: čtvrtina ukládaných env slotů
  zůstává pod teacherem i při beta=0. Ostatní envs volí teachera nezávisle.
  Archiv přijímá pouze okna, ve kterých všechny vykonané akce pocházely od
  teachera. Při 640×360 a délce 24 stojí plný archiv přibližně 1,49 GiB RAM.
- `dagger_min_train_sample_ratio: 2.0`: počet aktualizací je nejméně
  `ceil(train_every * stored_envs * ratio / (batch_sequences * sequence_length))`.
  U 400 ukládaných envs, batch 16, délky 16 a intervalu 20 je to 63 aktualizací.
- `reset_to_stage0: false` a `use_snapshot_curriculum: true` dovolují obnovovat
  pokročilé fáze. Snapshot musí být dostupný; samy tyto volby nevytvoří
  kvalitní demonstraci fáze, kterou teacher nezvládá.
- `beta_schedule: stage_sawtooth` je nyní ve všech čtyřech train configách.
  `dagger_target_stage: 4` znamená stages 0–4 **včetně**: dokončení stage 4
  ukončí úspěšnou epizodu, stage 5 se do curriculum nevybírá. Eval configy mají
  stejný cíl. Šest souborů teacherových politik se stále načítá jako celek.
- Prvních `dagger_teacher_warmup_steps: 5000` vektorových kroků řídí teacher.
  Na konci se jednou změří a zmrazí jeho reference samostatně pro každou stage.
  U stages 1–4 se začíná ze snapshotu a epizoda končí dokončením měřené stage.
  Reference je v `teacher_stage_reference.json`, `.pth` a v checkpointu.
  Úspěšnost celého prefixu je diagnostika; nulová úspěšnost nezastaví distilaci.
- Potom beta klesá o 0,001 za vektorový krok (`dagger_beta_decay_steps: 1000`).
  Každých `dagger_eval_every: 500` kroků se změří čistý student po stages.
  Pokud u některé stage zaostává za teacherovou referencí o **více než 0,1**,
  beta jednou vzroste o 0,1 (nejvýše na 1) a pokles se pozastaví na 500 kroků.
  Více špatných stages neznamená více zvýšení za stejné měření. Pauza se
  neprodlužuje, pokud zvýšení na horní mezi už nebylo možné. Bez korekcí beta
  dosáhne nuly po 1000 krocích od warmupu; s korekcemi později.
- Platné porovnání vyžaduje alespoň 30 dokončených pokusů u teachera i studenta;
  standardně se měří 32 epizod na stage. Nedostupný snapshot, nedostatek pokusů
  nebo dosažení vnějšího limitu evaluace znamená neznámý výsledek a nevyvolá
  korekci. Přirozený timeout epizody je selhání. Limit je 6000 kroků na kohortu.
  Neznámé teacherovy reference se automaticky nedoplňují; sleduj warningy.
- Probes používají **samostatný proces se scénou s 32 envy**. Načítají stejné
  teacherovy váhy a před měřením dostanou aktuální studentské váhy. Mají vlastní
  historii, resetování i renderer (Nyx obsahuje globální singletony). Tréninkové epizody, replay a GRU
  zůstávají během evaluací nedotčené. Přidává to paměťovou a časovou režii.
  Každých 2500 kroků po warmupu se navíc měří studentský prefix 0–4.
- Resume zachová referenci, warmup, beta i zbývající pauzu; warmup nezačíná
  znovu. Změna regulátoru nebo cílové stage při resume je odmítnuta.
  Starší `beta_schedule: evaluation` zůstává podporovaný pro staré běhy.

Tyto lokální stage probes měří schopnost provést fázi ze snapshotu. Úspěch
souvislého průchodu 0–4 ověřuje zvlášť prefixová evaluace; úspěch jednotlivých
stages jej sám nedokazuje. Hodnoty 30/32 a tolerance 0,1 jsou praktické nastavení,
nikoli statistická záruka. Chráněných 25 % teacherových slotů zůstává i při beta=0.

TensorBoard odděluje `Evaluation/teacher/*`, `Evaluation/student/*`,
`Health/*`, `Replay/stage_N_teacher_windows` a `Student/trained_to_stored_ratio`.
Tři učicí bloky s nulovým gradientem encoderu i GRU při nenulové chybě zastaví trénink.
Nečíselná loss, gradient nebo akce vyvolá chybu. Přehled evaluací je také
v `recurrent/training_evaluations.jsonl`.

Reprodukovatelný krátký důkaz optimalizace a nezávislá evaluace:

```bash
python -m config_run.dagger_recurrent.verify_learning --output config_run/output/dagger_verification
python -m unittest config_run.dagger_recurrent.test_student config_run.dagger_recurrent.test_pipeline config_run.dagger_recurrent.test_sequence_integrity config_run.dagger_recurrent.test_learning
```

Ověření používá skutečný ResNet, snímky 640×360 a produkční délku 16 + burn-in 8.
Měří chybu na pevných trénovacích i oddělených validačních oknech, změnu vah
obou encoderů/GRU a gradienty. `--synthetic` ověří pouze optimalizaci bez
simulátoru. `results.json` záměrně odděluje `optimization_checks_passed`
od `full_task_success_proven`; pokles MSE není důkaz dokončení úkolu.

## Architektura a rozhraní

Aktuální standardní train/eval konfigurace používají vstup **640×360**:

```text
Nyx RGB 640×360 → RGB [B,3,360,640] / 255 → ResNet-18 → GRU → 34 akcí
```

Společné nastavení je `dagger_image_width: 640`, `dagger_image_height: 360`.
`config_run/student_vision.py` při shodné velikosti obrazu a vstupu pouze
převede pořadí os a zařízení. Žádný crop ani resize neproběhne. Pro odlišné
rozměry podporuje ořez a resize. Stejná funkce je zapojená i do původního studenta. Bez obou klíčů
zůstává kvůli starým konfiguracím výchozí čtvercový vstup 128×128.

ResNet zpracuje plný obraz a až jeho závěrečný adaptive global pooling
vytvoří 512 příznaků. GRU, propriocepce a šest akčních hlav se nemění.
`dagger_new_visual_batch_size: 4` rozdělí snímky envů/sekvencí do menších
dávek při zachování pořadí. Při zapnutém `dagger_new_checkpoint_visual` (výchozí
`true`) se při učení aktivace encoderu přepočítají během backward pomocí
gradient checkpointingu; gradienty i přesnost vstupu zůstávají
zachované za cenu dalšího výpočtu. Hodnota 0 vypíná dělení i checkpointing.
L40 benchmark porovnává i `dagger_new_checkpoint_visual: false`, které může
být na kartě s větší VRAM rychlejší. Viz [dagger_scaling.md](../dagger_scaling.md).

640×360 má 14,0625× více pixelů než 128×128. Aktuální konzervativní config
proto ukládá 8 envů × 512 kroků na CPU, přibližně 2,64 GiB RGB + malá režie
propriocepce/GRU. Trénuje 2 sekvence v dávce. Jde o výchozí nastavení paměti,
nikoli nově změřené optimum rychlosti pro RTX 3060.

Nové checkpointy ukládají `image_shape: [3,360,640]`, verzi preprocessingu
a velikost encoderové dávky. Evaluace použije rozměry uložené v checkpointu,
takže starý rekurentní checkpoint 128×128 lze stále vyhodnotit s původním
rozměrem. Resume odmítne změnu rozměru obrazu; pro nové rozlišení spusť nový
trénink. Replay a GRU se při spuštění obnovují prázdné jako dosud.

Reálný deploy nyní ořezává podle cílového poměru stran. Pro obdélníkový
vstup použij `--image-width 640 --image-height 360`: celý snímek 1280×720
zmenší na 640×360 bez ořezu. Vstupní snímek 640×360 už rozměry nemění.
Výchozí 128×128 je zachované.
Stávající deploy skript nadále načítá původní VisionStudent; tato úprava
nepřidává podporu rekurentního 34akčního modelu do jeho robotického rozhraní.

```text
RGB [3,360,640] → ResNet-18 → 512 ────────────────┐
                                                ├→ GRU (256) → šest hlav → 34 akcí
q (43) + dq (43) + IMU (6) + a_previous (34)       │
             → MLP 126 → 256 → 128 ──────────────┘
```

`RecurrentVisionStudent` standardně používá 43 kloubů, jednu vrstvu GRU a 256
hodnot paměti. Počet kloubů a velikosti skrytých vrstev jsou konfigurovatelné.

Vstupy jednoho kroku:

| Vstup | Tvar | Význam |
|---|---|---|
| `images` | `[B,3,360,640]` | Float RGB 0–1; rozměry jsou konfigurovatelné, síť přijímá H,W >= 32. |
| `joint_positions` | `[B,43]` | Všechny skutečné polohy kloubů v rad. |
| `joint_velocities` | `[B,43]` | Stejné klouby ve stejném pořadí, rad/s. |
| `imu` | `[B,6]` | Úhlová rychlost (3, rad/s) a směr jednotkové gravitace (3), obojí v lokálním rámci linku `imu_in_pelvis`. |
| `previous_actions` | `[B,34]` | Předchozí skutečně aplikovaný povel po teacher/student výběru, clippingu a případném omezení/zpoždění. |
| `hidden` | `[L,B,hidden_size]` | Paměť GRU; `None` znamená nuly. |
| `reset_mask` | `[B]`, bool | `True` vynuluje paměť daného prostředí před zpracováním observations. |

Pro sekvence přidej všem vstupům kromě `hidden` rozměr T za B:
`images [B,T,3,H,W]`, senzory `[B,T,D]`, maska `[B,T]`.
Výstup je dvojice `(actions, next_hidden)`, akce mají `[B,34]` nebo `[B,T,34]`.
Model si paměť neukládá interně a sám ji neodpojuje od autograd grafu.

Pořadí výstupů přesně odpovídá současnému `SB3_chairman_env`:

| Hlava | Indexy | Výstup |
|---|---|---|
| `waist` | 0–2 | yaw, roll, pitch |
| `left_arm` | 3–9 | shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw |
| `right_arm` | 10–16 | Stejné pořadí vpravo |
| `left_fingers` | 17–23 | thumb 0/1/2, middle 0/1, index 0/1 |
| `right_fingers` | 24–30 | Stejné pořadí vpravo |
| `direction` | 31–33 | walk vx [m/s], vy [m/s], yaw rate [rad/s] |

Výstup je bez tanh: prvních 31 hodnot jsou absolutní cílové úhly, poslední tři
jsou povely chůze. Clipping zůstává odpovědností řídicího wrapperu. Výstup
neobsahuje odhad nejistoty; jednotlivé hlavy rozdělují řízení částí těla.

### Normalizace a váhy ResNetu

Model interně normalizuje RGB pomocí ImageNet mean/std. Vstup tedy pouze
převeď na RGB, CHW a vyděl 255; ImageNet normalizaci nedělej podruhé.
V dodaných `chairman_separate/train_dagger*.yaml` je nyní předtrénování
zapnuté a fine-tuning vypnutý. Samostatný konstruktor modelu stále standardně
nestahuje váhy. Pokud v těchto konfiguracích přepneš na náhodnou inicializaci,
nastav také `dagger_new_finetune_visual: true`, aby se náhodný encoder učil.

```yaml
dagger_new_pretrained_visual: true
dagger_new_finetune_visual: false        # zmrazit ResNet i při resume již fine-tunovaného checkpointu
learning_rate: 0.0001                    # GRU, propriocepce, akční hlavy a nové LayerNorm
dagger_new_visual_learning_rate: 0.00001  # pouze ResNet, včetně jeho affine parametrů
dagger_new_visual_warmup_updates: 1000    # dokončené gradientní aktualizace, nikoli kroky simulace
```

Následující dvě varianty popisují `dagger_new_pretrained_visual`; fáze učení
encoderu se uplatní pouze při `dagger_new_finetune_visual: true`.

- **`true`:** načte ImageNet ResNet-18 (`ResNet18_Weights.DEFAULT`). Zachová
  původní BatchNorm a všechny naučené running mean/variance. Tyto statistiky
  se nikdy nemění, ani při fine-tuningu. Prvních 1000 aktualizací se učí nové
  části modelu, zatímco celý ResNet včetně affine parametrů zůstává zmrazený.
  Aktualizace 1001 již fine-tunuje ResNet s menším LR. Optimizer nových částí
  se při odemčení znovu neinicializuje. Hodnota `0` zapne fine-tuning ihned.
- **`false`:** začíná s náhodnými vahami, BatchNorm nahrazuje GroupNorm se
  32 skupinami. ResNet se učí od první aktualizace, také s nastaveným menším LR.
  `dagger_new_visual_warmup_updates` se v tomto režimu neuplatní: náhodný
  encoder nezůstane po dobu warmupu zmrazený.

Obě varianty používají LayerNorm obrazového i proprioceptivního embeddingu
před GRU. Normalizace nezávisí na ostatních snímcích ani prostředích. Nové
části pracují už během warmupu s obrazovými příznaky zmrazeného ResNetu.
První předtrénovaný běh může stáhnout přibližně 45 MB vah do standardní cache
torchvision. Na stroji bez internetu lze cache připravit předem přes `TORCH_HOME`.

Warmup vizuálního encoderu je nezávislý na `dagger_teacher_warmup_steps`, který
řídí změny beta. Počet aktualizací je zde použit záměrně: jeho význam zůstává
přímý i při změně počtu prostředí a aktualizací v jednom trénovacím bloku.
Nastavení 1000 je nastavitelné výchozí číslo, nikoli empiricky nalezené optimum.

Ve všech čtyřech dodaných train konfiguracích je nyní
`dagger_new_finetune_visual: false`: váhy celého ResNetu zůstávají pevné,
včetně affine parametrů BatchNorm. Jeho gradienty a efektivní LR jsou nulové,
počet aktualizací jej neodemkne. Učí se dál propriocepční síť, GRU, akční
hlavy a navazující LayerNorm. ResNet stále počítá obrazové příznaky při
každém průchodu; cache příznaků v replay není součástí této změny.

Checkpoint uchovává typ normalizace, původ vah, dokončené aktualizace, warmup
i oba learning rates. Při `load_and_train_dagger_new` se obnoví tato uložená
nastavení a fáze: warmup nezačíná znovu. Výjimkou je explicitní
`dagger_new_finetune_visual` v konfiguraci, který má přednost před uloženým
příznakem: `false` zmrazí i již odemčený encoder při zachování jeho vah a
momentů Adamu. Pozdější `true` umožní pokračovat podle uloženého počtu
aktualizací. Pokud klíč vynecháš, převezme se uložený příznak; u starších
checkpointů bez příznaku a u nového běhu bez tohoto klíče je výchozí `true`.
Pro evaluaci i resume se váhy a statistiky načtou přímo ze studentského
checkpointu, bez opětovného stahování ImageNetu. Přepínač předtrénování nelze
změnit uprostřed resume; pro změnu inicializace začni nový běh a použij nový
výstupní adresář. Starší v2 checkpointy s jedinou skupinou optimizeru se při
obnově rozdělí na dvě skupiny při zachování momentů Adamu; v1 resume zůstává
odmítnuto.

Při `eval_dagger_new` a `eval_dagger_new_video` odlišná kalibrace kamery nebo
transformace splatu pouze vypíše warning; evaluace pokračuje s aktuální
konfigurací scény. Checkpoint se nepřepisuje. Při pokračování tréninku je
kontrola shody kalibrace nadále striktní. Kontroly pořadí akcí, kloubů,
IMU a periody řízení zůstávají striktní i během evaluace.

TensorBoard obsahuje `Student/visual_finetuning_enabled`, `Student/visual_finetuning_active`,
`Student/visual_learning_rate`, `Student/completed_gradient_updates` a
`Student/visual_warmup_updates_remaining`. Během warmupu má skutečný ResNet LR
0 a jeho gradient je nulový; po odemčení se použije nastavený vizuální LR.
Profilovací skript standardně používá vizuální warmup 0, ale respektuje
vypnutý fine-tuning v konfiguraci. Pro měření fine-tuningu jej nejprve povol.
Volbou `--visual-warmup-updates N` lze explicitně měřit i warmup.

Ověření obou variant při 640×360 na GPU (syntetická úloha, skutečné ImageNet
váhy pro zapnuté předtrénování, 60 aktualizací, zkrácený warmup 10):

| Inicializace | Trénovací MSE před → po | Validační MSE před → po |
|---|---:|---:|
| ImageNet | 0,03344 → 0,00897 | 0,03287 → 0,01786 |
| Náhodná + GroupNorm | 0,03508 → 0,00867 | 0,03558 → 0,01798 |

V obou testech byly konečné gradienty encoderu nenulové a naměřená saturace
vstupů GRU 0 %. Testy ověřují funkčnost obou cest optimalizace; tato čísla
nejsou srovnáním kvality obou inicializací na úloze ChairMan.
Podrobnosti jsou v
[ImageNet JSON](../output/resnet_finetuning_20261109/pretrained/results.json) a
[scratch JSON](../output/resnet_finetuning_20261109/scratch/results.json).
Regresní testy navíc kontrolují neměnnost všech vah encoderu během warmupu,
neměnnost statistik BatchNorm po odemčení, odemčení ve správné aktualizaci
a přesnou shodu pokračujícího tréninku s obnovením z checkpointu.

Samotný model má identity normalizaci. Runner nastavuje pevné fyzikální škály:
kloubové rychlosti násobí 0,05 a úhlovou rychlost IMU 0,25, ostatní vstupy
ponechává v uvedených jednotkách. To nemění normalizaci během aktivní historie
GRU. Všechny škály jsou součástí `state_dict` a evaluace je znovu použije.
Samostatné použití modelu může nastavit tréninkové statistiky přes
`model.set_proprio_normalization(mean, std)` v pořadí `q, dq, imu, previous_actions`.
Loss runneru je MSE fyzických 34 akcí, gradienty se omezují podle `grad_clip`.

### Jednokroková inference

```python
import torch
from config_run.dagger_recurrent import RecurrentVisionStudent

model = RecurrentVisionStudent().eval()
B = 2
hidden = None
previous_actions = torch.zeros(B, 34)  # definovaný počáteční povel po resetu
with torch.no_grad():
    actions, hidden = model(
        images=torch.zeros(B, 3, 360, 640),
        joint_positions=torch.zeros(B, 43),
        joint_velocities=torch.zeros(B, 43),
        imu=torch.tensor([0., 0., 0., 0., 0., -1.]).expand(B, -1),
        previous_actions=previous_actions,
        hidden=hidden,
        reset_mask=torch.ones(B, dtype=torch.bool),
    )
```

Při dalších krocích předej vrácené `hidden`. Masku nastav jen pro envs, jejichž
aktuální observation už patří nové epizodě. Při tréninku paměť odpoj na hranici
tréninkových úseků (`hidden.detach()`), nikoli mezi všemi snímky v úseku.
Model ukládej spolu s parametry konstruktoru, seznamem názvů kloubů a akcí,
konvencí IMU, rozlišením a periodou řízení. Samotný `state_dict` pořadí kloubů
ani tyto konvence nepopisuje. Checkpoint starého studenta není kompatibilní.

## Správa vstupů, paměti a replay

- `inputs.py`: q/dq se mapují podle uložených názvů všech kloubů, nikoli podle
  pořadí akcí. IMU se počítá z body state linku `imu_in_pelvis`: world-frame
  gyro a gravitace `[0,0,-1]` se otočí jeho inverzní orientací WXYZ do rámce
  tohoto IMU linku. Pokud link chybí, adaptér skončí s jasnou chybou; nepoužije
  potichu `pelvis`. Není to syrová
  akcelerace ani další renderovaný senzor; není potřeba přidávat IMU do YAML.
- `RolloutState`: GRU se aktualizuje každý krok, i při provedení teacherovy
  akce. Předchozí akce je skutečně vybraný a oříznutý povel, ne studentův
  návrh. Neobsahuje 12 nízkoúrovňových leg targets. Na `done` se paměť i
  previous action příslušného envu ihned vynulují. Platí pro úspěch, selhání,
  timeout i reset do snapshotu. Samotný přechod stage bez resetu paměť nemaže.
- `buffer.py`: na začátku běhu náhodně vybere `dagger_store_per_step` envs
  a pak ukládá stejné envs každý krok. Nelze přehazovat náhodné env IDs jako
  u nesekvenčního bufferu. Ring uchovává souvislé časové stopy, expert labels,
  reset masky a vstupní rollout hidden state. Přes jeho nejnovější/nejstarší
  hranici se nikdy nevytváří nesouvislá sekvence.
- Ukončení jedné epizody **nemaže celý replay**. Data jsou stále platná a
  maska na první observation nové epizody přeruší její historii uvnitř GRU.
  `clear()` invaliduje celý buffer pro nový běh; resume vytváří nový prázdný
  buffer. Nepoužívá se padding ani ztráta na neexistujících observations.
- Učení používá uloženou vstupní paměť, poté burn-in s aktuální sítí bez
  gradientu a nakonec učený úsek s gradientem. Je to **truncated BPTT**:
  rollout paměť ze starší verze vah je aproximace, burn-in ji aktualizuje;
  nepřepočítává se celý prefix epizody od resetu. Reset masky zůstávají přesné.
  Také aktivní rollout paměť pokračuje po gradientním updatu, nenuluje se
  uměle při každém učení. Velikost burn-in a sekvence lze později ladit.
- `checkpoint.py`: formát, kontrola pořadí akcí, kloubů, IMU a periody; obnova
  architektury přímo z metadat. `runner.py`: trénink, CSV/grafy a RGB video.
  Nové checkpointy ukládají také `imu_body_name: imu_in_pelvis`. Starší
  rekurentní checkpoint bez tohoto pole lze načíst, pouze pokud aktuální
  orientace a úhlové rychlosti `pelvis` a `imu_in_pelvis` souhlasí. V současném
  G1 URDF je spojuje fixed joint s `rpy="0 0 0"`, proto jsou gyro/gravity
  vstupy shodné a není nutné kvůli tomuto přesunu začínat nový trénink.

Reálný deployment zde připojen není. Potřebuje stejné mapování kloubů,
IMU konvence, normalizaci, časování a správu paměti. Starý deployment s
mapováním 17 akcí pro tento model použít nelze.

### Kontrola časového pořadí dat

Testy v `test_sequence_integrity.py` označují snímky i senzory číslem kroku
a prostředí. Kontrolují všechna dostupná 24kroková okna před zaplněním
i po několika přetočeních ring bufferu, souběžné indexování všech vstupů,
teacherových cílů, vstupní paměti a reset masek. Test sběru používá skutečný
runner a adaptér se syntetickým prostředím, pevně přeházeným výběrem envů,
částečnými resety a teacher/student řízením včetně clippingu.

Další test porovnává skutečný ResNet + GRU po jednotlivých krocích s celou
sekvencí a s burn-in 8 + učenými 16 kroky; vizuální dávka 5 přitom protíná
hranice sekvencí. Kontrola Nyx spouští produkční funkci dávkování izolovaně
s náhradním rendererem, který opakovaně přepisuje stejnou výstupní paměť.
Ověřuje kopírování do správných env/camera řádků, nikoli fyzický render GPU.

```bash
python -m unittest config_run.dagger_recurrent.test_sequence_integrity config_run.dagger_recurrent.test_student config_run.dagger_recurrent.test_pipeline
```

Při kontrole 29. 10. 2026 prošlo všech 21 testů na CPU. Jde o kontrolu
indexování a rekurentního zpracování, nikoli zkoušku dlouhého běhu 400 envů.

## Návrh soustředění randomizace do jednoho místa

Následující struktura je návrh pro další krok, nikoli implementovaná změna:

```text
config_run/domain_randomization/
    config.py       # jeden typovaný config: rozsahy, pravděpodobnosti, seed
    manager.py      # reset / observation / training_batch / action
    visual.py       # obrazové transformace + parametry kamery/scény
    sensors.py      # bias, šum, zpoždění a výpadky; historie po jednotlivých envs
    physics.py      # adaptér změn parametrů podporovaných Genesis/Nyx
```

Jedna konfigurace runu (`student_randomization:` v YAML) a jeden manager vlastní
pravidla i RNG. Volající místa pouze zavolají jeho metody. Náhodný stav a config
ukládej s checkpointem; při resume definuj i chování stavových front zpoždění.
Neřeš konfiguraci náhodnými konstantami uvnitř CNN, rewardů a checkerů.

| Hook | Kde se bude volat v současném kódu | Co vlastní manager |
|---|---|---|
| `on_reset(env_ids)` | Reset checkeru `stages_chairman.py` / budoucí lifecycle hook po resetu, včetně snapshot resetů | Počáteční póza, parametry kamery, bias senzorů, fyzika; vyčištění front pouze resetovaných envs. |
| `observe(clean_inputs, env_ids)` | Za `StudentInputs.read()` v `runner.py`; starý student má obdobné místo za `_student_inputs` v `separate_dagger.py` | Šum, offset, zpoždění a výpadky studentových senzorů během rolloutů. |
| `augment_training_batch(sequence)` | `buffer.py:train_sequence_step()` po `buffer.sample()` a před burn-in i učenou částí | Opakovaná obrazová augmentace uložených čistých dat, konzistentně pro celou sekvenci. |
| `process_action(selected_action)` | V `runner.py` po teacher/student volbě a clippingu, před `env.step` | Případná simulace akčního zpoždění a omezení. Musí vrátit i skutečně odeslanou akci pro `RolloutState.after_step()`. |

Existující `_randomize_stage0_batch()` a jeho rozsahy v
`metasim/cfg/checkers/stages_chairman.py` už randomizují polohu robota a židle.
Při integraci je přesměruj na společný config/manager, aby nevznikla dvojitá
randomizace. Checker má volat hook; nemá vlastnit další sadu rozsahů. Snapshot
reset musí také aktualizovat stav manageru, ne pouze obnovit stav simulace.

### Co randomizovat a kdy

- **Při resetu, konstantní v epizodě:** montáž kamery, její kalibrace/FOV,
  scénu/pozadí, bias kloubů, hmotnost/tření a parametry serv. Některé parametry
  Genesis nebo Nyx může být nutné nastavit při vytvoření scény nebo rozdělit
  mezi samostatné runy; existující obecný `RandomizationCfg` sám nedokládá
  podporu všech změn pro tyto backendy. Ověř účinek v daném rendereru/solveru.
- **Během rolloutů:** snímkový šum, měřené zpoždění a jitter, výpadky dat.
  Teacher dostává čistý skutečný stav simulace, student degradované senzory.
  U zpožděných observations teacherova cílová akce stále odpovídá aktuálnímu
  stavu; paměť musí pomoci rozdíl odhadnout. Je to odlišné od zpoždění actuatorů.
- **Při učení z bufferu:** jas, kontrast, barvy, mírné rozmazání, šum,
  kompresi a drobné zakrytí. Kalibrační/crop posun aplikuj konzistentně na
  celou sekvenci, aby nevznikalo neexistující chvění kamery. Obrazové změny
  nesmějí měnit význam akčního labelu; horizontal flip vyžaduje transformovat
  levou/pravou stranu, senzory i cílové akce.

Ukládej ideálně čistá data a skutečně provedené akce. Rozlišuj augmentaci
vzhledu (lze opakovat při každém vzorkování) od stavového šumu/zpoždění
(musí zachovat časovou historii). Stejný bias nebo zpoždění náhodně neaplikuj
podruhé na data, která už byla degradována. Rozsahy odvozuj od reálných logů,
nejprve mírné; při změnách fyziky ověř, že teacher nadále poskytuje dobré akce.

Pro evaluaci měj explicitní profily `clean`, `held_out_randomized`, `real`,
vypnuté náhodné tréninkové augmentace a oddělené seedy. Měř úspěšnost celých
studentských epizod a jednotlivých fází, nejen MSE na teacherových akcích.

## Implementovaná obrazová augmentace

Společný modul `config_run/visual_augmentation.py` obsahuje rozsahy na začátku
souboru: jas/kontrast/gamma 0,9–1,1, RGB gains 0,95–1,05, směrodatnou odchylku
šumu 0–2/255 a pravděpodobnost mírného Gaussian blur 0,1. Polovina epizod nebo
sekvencí zůstává čistá. Jde o počáteční mírné rozsahy; uprav je podle reálných
obrazů a po změně restartuj proces.

Všech pět train/resume konfigurací `chairman_separate/*train_dagger*.yaml`
zapíná `dagger_visual_augmentation: true`. Nastavením `false` augmentaci vypneš;
chybějící klíč také znamená vypnuto. Volitelný `dagger_augmentation_seed`
řídí oddělené generátory augmentace. Funguje i se zmrazeným ResNetem.

`runner.py` volá `VisualAugmentation.rollout()` před studentem: vzhled se losuje
po resetu jednotlivého prostředí. `buffer.py` volá `sequence()` po vzorkování:
vzhled je stejný po celé sekvenci včetně burn-inu. Snímací šum je nezávislý pro
každý pixel a snímek. Replay i archiv uchovávají čisté uint8 RGB; teacherovy
labely, propriocepce a akce se nemění. Augmentace běží nad float RGB [0,1]
před stávající ImageNet normalizací. Generátory se ukládají s checkpointem;
při resume začínají nové epizody s nově losovaným vzhledem.

Standardní evaluace, stage probes pro betu a evaluační video zůstávají čisté.
V TensorBoardu sleduj `Augmentation/enabled`, `Augmentation/rollout_env_fraction`
a zaznamenané rozsahy v `Run/visual_augmentation`. Generalizaci je potřeba
ověřit samostatnou evaluací na odlišných podmínkách a reálných datech.

V `calibrate_nyx` klávesa **G** losuje nový vzhled a **V** přepíná čistý obraz.
Podrobnosti jsou v [dokumentaci kamery](../camera_fov.md).

## Ověření

```bash
python -m unittest config_run.dagger_recurrent.test_student
python -m unittest config_run.dagger_recurrent.test_pipeline
python -m unittest metasim.test.test_chairman_separate_dagger
```

Testy nového modelu zahrnují shodu sekvenční a krokové inference, nezávislé
resety envs uvnitř sekvence, pořadí akčních hlav, zpětný průchod přes ResNet,
MLP, GRU a všechny hlavy a obnovu normalizace/predikcí z checkpointu.
Pipeline testy používají falešné fyzikální prostředí a malý obrazový encoder;
ověřují routing režimů, DAgger příznaky prostředí, převod senzorů, kontinuitu
replay po přetečení, resety, trénink/resume/evaluaci/video a kontrolu metadat.
Nenahrazují plný rollout s Genesis/Nyx a skutečným PPO teacherem.

Ověření nové stage bety a krátký GPU integrační test jsou popsané v
[VALIDATION_STAGE_BETA_20261109.md](VALIDATION_STAGE_BETA_20261109.md).
