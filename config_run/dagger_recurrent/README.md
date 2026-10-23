# Rekurentní student pro ChairMan separate

Samostatný model; stávající `dagger_vp/student_net.py`, DAgger trainer, buffery,
konfigurace a deployment jej zatím neimportují. Spuštění `main_separate.py`
proto stále používá dosavadního studenta. Tento adresář není nový hotový trainer.

## Architektura a rozhraní

```text
RGB [3,H,W] → ResNet-18 → 512 ────────────────────┐
                                                ├→ GRU (256) → šest hlav → 34 akcí
q (43) + dq (43) + IMU (6) + a_previous (34)       │
             → MLP 126 → 256 → 128 ──────────────┘
```

`RecurrentVisionStudent` standardně používá 43 kloubů, jednu vrstvu GRU a 256
hodnot paměti. Počet kloubů a velikosti skrytých vrstev jsou konfigurovatelné.

Vstupy jednoho kroku:

| Vstup | Tvar | Význam |
|---|---|---|
| `images` | `[B,3,H,W]` | Float RGB 0–1, minimálně 32×32; pro začátek 128×128. |
| `joint_positions` | `[B,43]` | Všechny skutečné polohy kloubů v rad. |
| `joint_velocities` | `[B,43]` | Stejné klouby ve stejném pořadí, rad/s. |
| `imu` | `[B,6]` | Úhlová rychlost těla v body frame (3, rad/s), pak směr jednotkové gravitace v body frame (3). |
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

Propriocepce má ve výchozím stavu identity normalizaci. Před delším tréninkem
nastav `model.set_proprio_normalization(mean, std)` ze statistik tréninkových
dat v pořadí `q, dq, imu, previous_actions`, se spodním omezením std pro téměř
konstantní příznaky. Statistiky jsou součástí `state_dict`. Při evaluaci je
neměň. Normalizace vstupu není normalizace cílových akcí; loss bude vhodné
škálovat/vážit podle jednotlivých akčních skupin.

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
        images=torch.zeros(B, 3, 128, 128),
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

## Co bude potřeba při budoucím napojení

1. **Jeden adaptér vstupů:** v samostatném modulu sestavit q/dq z
   `handler.get_states().robots[robot_name]`, RGB a IMU. Pro sim2real používej
   explicitní seznam názvů kloubů a mapování, ne předpoklad shodného pořadí
   senzorů. Současný sim vrací q v pořadí simulátoru, které se liší od pořadí
   akčních kloubů. World-frame úhlovou rychlost pelvisu otoč do body frame;
   gravitaci `[0,0,-1]` otoč inverzní orientací pelvisu (kvaternion WXYZ).
   V realitě musí orientace i gyro odpovídat stejnému tělesovému rámci; není
   to syrové zrychlení akcelerometru. Respektuj také časové značky senzorů.
2. **Collector:** udržovat paměť pro každé prostředí, aktualizovat ji každý
   krok i při vykonání teacherovy akce. Ukládat předchozí provedenou akci,
   nikoli vždy studentův návrh. Při resetu nastavit paměť a počáteční previous
   action podle dohodnuté konvence (například nulový sentinel). Těchto 34
   hodnot neobsahuje 12 nízkoúrovňových leg targets z G1MotionPolicy.
3. **Sekvenční buffer/trainer:** nynější `DAggerBuffer.sample()` vzorkuje
   nezávislé snímky. Pro GRU potřebuje souvislé úseky z jednoho envu,
   identifikaci epizod a reset masky. Na náhodně vybraný prostředek epizody
   nestačí vždy inicializovat GRU nulami: použij prefix pro rozehřátí paměti
   (burn-in) nebo sekvence od začátku epizody. Návrh k ověření: historie
   0,5–1 s, počet snímků určit skutečnou řídicí periodou. Padding nezapočítávej
   do lossu. Teacher labels mohou zůstat jednotlivé 34rozměrné akce.
4. **Evaluace/deployment:** stejné mapování, normalizace a správa paměti.
   Starý deployment s mapováním 17 akcí nelze použít pro tento model.

Tyto integrace zde záměrně nejsou provedeny.

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
| `observe(clean_inputs, env_ids)` | Budoucí adaptér nahrazující `_student_inputs` v `separate_dagger.py` | Šum, offset, zpoždění a výpadky studentových senzorů během rolloutů. |
| `augment_training_batch(sequence)` | Budoucí sekvenční trainer; dnes obdobné místo je mezi `buffer.sample()` a `student(...)` v `dagger_vp/dagger_trainer.py` | Opakovaná obrazová augmentace uložených čistých dat. |
| `process_action(selected_action)` | V DAgger collectoru po teacher/student volbě a před `env.step` | Případná simulace akčního zpoždění a omezení. Musí vrátit i skutečně odeslanou akci pro příští observation. |

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
python -m unittest metasim.test.test_chairman_separate_dagger
```

Testy nového modelu zahrnují shodu sekvenční a krokové inference, nezávislé
resety envs uvnitř sekvence, pořadí akčních hlav, zpětný průchod přes ResNet,
MLP, GRU a všechny hlavy a obnovu normalizace/predikcí z checkpointu.
