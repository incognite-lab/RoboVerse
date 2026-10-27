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
Rozlišení a perioda se kontrolují; montáž a FOV se automaticky nepřepisují.

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
váhy/optimizer a absolutní krok; beta se přepočítá ze stejného sdíleného
schedule. **Simulace, replay, předchozí akce a rollout paměť začínají znovu**;
nejde o bitově přesné pokračování rozehrané epizody. Buffer se znovu naplní
před prvním gradientním updatem. Pro resume ponech `expert_model_path` na
teacher bundle a nastav `load_student_path` na nový studentský checkpoint.

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
`pretrained_visual=False` je výchozí nastavení bez stahování vah.
`pretrained_visual=True` načte ImageNet ResNet-18 a může potřebovat síť nebo
lokální cache torchvision. BatchNorm průběžné statistiky jsou vždy zmrazené,
aby výsledek nezávisel na budoucích snímcích v sekvenci ani na ostatních envs;
konvoluce i affine parametry BatchNorm zůstávají trénovatelné. To platí i při
tréninku od náhodné inicializace.

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
