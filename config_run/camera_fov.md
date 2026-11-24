# Samostatné vodorovné a svislé FOV v Genesis/Nyx

## Ovladatelný Nyx náhled pro zarovnání splatu

```bash
python config_run/main_separate.py chairman_separate/calibrate_nyx
```

Konfigurace `chairman_separate/calibrate_nyx.yaml` používá jedno prostředí
a jedno ovladatelné GLFW okno s obrazem vykresleným Nyxem. Zobrazuje splat
i skutečnou geometrii robota, židle a podlahy v jedné scéně. Režim
`preview_nyx` nepotřebuje váhy studenta ani teachera; po resetu pravidelně
renderuje bez krokování fyziky a spouštění politik.

Klikni do okna, aby přijímalo klávesy. Držení pohybových kláves umožňuje
plynulý pohyb podle uplynulého času; mimo aktivní okno se kamera nepohybuje.

| Klávesy | Akce |
|---|---|
| W / S | Dopředu / dozadu ve směru pohledu |
| A / D | Posun doleva / doprava |
| Q / E | Otočení pohledu doleva / doprava |
| R / F | Výška nahoru / dolů ve světových souřadnicích |
| Šipky nahoru / dolů | Naklonění pohledu nahoru / dolů |
| Šipky doleva / doprava | Otočení jako Q / E |
| Shift + pohyb | Trojnásobná rychlost |
| C nebo Tab | Přepnutí volná kamera / kamera robota |
| 1 / 2 | Přímo volná kamera / kamera robota |
| H | Návrat do počátečního volného pohledu |
| P | Výpis aktuální polohy, směru a up vektoru |
| Esc nebo zavření okna | Ukončení náhledu |

Při návratu z robotí kamery se obnoví poslední volný pohled. V robotím
pohledu pohybové klávesy nemění montáž ani kalibraci. Poloha a orientace se
počítají ze skutečné světové pózy `head_link`, montážního posunu a quaternionu
kamery; optická osa je lokální −Z, image up lokální +Y jako v Nyx pluginu.
Pro oba pohledy slouží jediný renderovací senzor, aby se splat nenačetl dvakrát.

```yaml
nyx_preview_controls: true
nyx_preview_move_speed: 1.0       # m/s
nyx_preview_turn_speed: 75.0      # stupně/s
nyx_preview_robot_camera_config: chairman_separate/train_dagger
nyx_preview_robot_camera_name: camera0
```

Montáž robotí kamery se čte přímo z uvedené train/eval konfigurace,
nepřepisuje se v ní. FOV kalibrační kamery musí odpovídat robotí kameře;
náhled zachovává své rozlišení (výchozí 1280×720, tedy vyšší než policy).
Po změně zdrojového configu náhled restartuj.

Nainstalované nativní GUI Nyxu neposkytuje Python callbacky pro vlastní
klávesy. Proto je při `nyx_preview_controls: true` použité vlastní GLFW okno
s přesným RGB výstupem Nyx kamery a `open_nyx_window: false`, aby se neotevírala
dvě různá okna. Vyžaduje `glfw` a `PyOpenGL` (součást zdejšího prostředí).
Pokud chceš původní nativní GUI bez našeho ovládání, nastav
`nyx_preview_controls: false` a `open_nyx_window: true` pod kamerou.

Ověření 23. 11. 2026: prošlo 31 testů náhledu, kamerových transformací,
renderovacího adaptéru a původní pipeline. GPU test vykreslil šest snímků
1280×720 při pohybu, přepnutí na robota, návratu a resetu. Robotí póza
souhlasila s výpočtem Nyx adaptéru s tolerancí 1e-5; návrat obnovil volnou
pózu s tolerancí 1e-8. RGB přečtené z OpenGL okna odpovídalo výstupu Nyxu
pixelově přesně. Vstupy GPU testu byly zadány programově; klávesové mapování
a práce s focusem jsou pokryté samostatnými testy.

Přepínač původního nativního GUI existuje i v běžných train/eval konfiguracích, pod kamerou:

```yaml
cameras:
  camera1:
    type: NyxGaussianSplatCameraCfg
    params:
      open_nyx_window: true    # false = vypnout nativní okno
      render_sim_geometry: true
```

`headless: true` vypíná samostatné okno Genesis rasterizeru; Nyx okno řídí
výhradně `open_nyx_window`; ovladatelné okno řídí `nyx_preview_controls`.
Náhled vyžaduje funkční GPU a grafickou relaci
(například místní desktop nebo VNC); samotný headless Slurm job okno na
přihlašovacím počítači neotevře. Kamera se musí pravidelně číst, aby Nyx
obsluhoval okno; kalibrační režim to zajišťuje i při stojící fyzice.

Pro registraci upravuj v kalibračním YAML:

- `gaussian_position`: posun v souřadnicích Genesis (Z nahoru);
- `gaussian_rotation_xyzw`: rotaci splatu jako quaternion XYZW;
- `gaussian_scale`: měřítko, typicky stejné ve všech třech osách.

### Skrýt simulovanou podlahu a ponechat kolize

V kalibračním configu i dodaných separate DAgger train/eval konfiguracích je:

```yaml
render_ground: false
```

Jde o parametr na nejvyšší úrovni YAML, vedle `headless`. Genesis vytvoří
rovinu s `visualization=False, collision=True`: robot a židle se o ni dál
fyzicky opírají, ale její síť se neexportuje do Nyxu. Obraz podlahy pak
pochází pouze z Gaussian splatu. Robot, židle a měřicí kvádr zůstávají
viditelní (`render_sim_geometry: true`). Skrytí platí i pro Genesis viewer.
`render_ground: true` síť podlahy znovu zobrazí; mimo tyto konfigurace
zůstává výchozí `true`. Nastavení nyní používá builder `main_separate.py`
a backend Genesis; ostatní backendy se tímto parametrem nemění.

Po změně restartuj scénu. Již uložené obrázky v demonstračním archivu se
zpětně nezmění. Viditelnost neřeší rozdíl výšky mezi kolizní rovinou a
rekonstruovanou podlahou: ten je potřeba doladit registrací splatu.

GPU ověření: skrytá rovina měla 0 vizuálních a 1 kolizní geometrii, nebyla
v seznamu instancí exportovaných do Nyxu. Po 29 fyzikálních krocích bylo
naměřeno 8 kontaktů robota s touto rovinou a nenulová svislá kontaktní síla.
Sada náhledu, měřidla, renderovacího adaptéru a pipeline prošla 39 testy.

### Měřicí kvádr a numerická klávesnice

Kalibrační konfigurace má zapnuté `nyx_ruler.enabled: true`. V Nyx scéně
je skutečná 3D tyč (bez kolizí) o výchozích rozměrech **1 m × 2 cm × 2 cm**.
Její podélná osa je lokální X. Červený střed značky A je začátek, modrý střed
B konec; měřená délka je vzdálenost **středů** těchto značek, nikoli jejich
vnějších okrajů. Při změně délky A zůstává na místě. Tyč je vidět i při
přepnutí do kamery robota a podléhá skutečné perspektivě a zakrytí v Nyxu.
Kalibrační světlo umožňuje rozeznat barvy simulovaných objektů; tréninkové
scény toto světlo ani měřidlo nepřidávají.

Klikni do okna; ideálně zapni Num Lock. Horní číslice 1/2 nadále přepínají
kameru, numerická klávesnice má oddělené ovládání měřidla:

| Numerické klávesy | Akce |
|---|---|
| 4 / 6 | Posun v ose světa X − / + |
| 2 / 8 | Posun v ose světa Y − / + |
| 3 / 9 | Posun v ose světa Z − / + |
| 7 / 1 | Otočení kolem světové osy Z + / − |
| `/` / `*` | Otočení kolem světové osy Y + / − |
| `+` / `−` | Délka + / − 1 cm; funguje i opakování při držení |
| Ctrl + ovládání měřidla | Jemný posun/rotace; změna délky po 1 mm |
| Enter | Uložit aktuální délku jako známou skutečnou délku |
| 5 | Vypsat a uložit měření a doporučenou transformaci |
| 0 | Reset polohy, rotace, délky, reference i viditelnosti měřidla |
| Desetinná tečka | Skrýt / zobrazit měřidlo |

Délka `L`, uložená skutečná délka `ref` a korekce `k` jsou v titulku okna.
Výchozí rozsah tyče je 1 mm až 10 m, rozlišení délky 1 mm. Rozlišení nástroje
není záruka přesnosti vizuálního měření. Nyx neumí měnit rozměry rigidního
meshe za běhu; tyč je proto složená ze souvislých pevných dílů. Jejich
přesouváním se změní délka skutečné geometrie bez restartování scény.

Postup pro získání měřítka:

1. Změř objekt v realitě. Nastav tyč na tuto délku a stiskni NumEnter,
   nebo vyplň `nyx_ruler.reference_length_m` v YAML před spuštěním.
2. Přesuň a natoč tyč na odpovídající objekt ve splatu. Umísti A na jeden
   konec a změň délku tak, aby B odpovídal druhému konci. Uložená reference
   se při těchto změnách nepřepisuje.
3. Ověř hloubku i konce z několika pohledů. Shoda v jediném 2D snímku nestačí:
   tyč blíž ke kameře může zdánlivě odpovídat většímu vzdálenému objektu.
4. Stiskni Num5. JSON se uloží do
   `config_run/output/nyx_calibration/measurement.json`; doporučené YAML
   hodnoty se také vypíšou do terminálu.

Výpočet je `k = skutečná_délka / naměřená_délka` a
`nové_gaussian_scale = současné_gaussian_scale × k`. Například skutečný
1 m při naměřených 1,25 m a původním měřítku `[4,4,4]` dává `[3.2,3.2,3.2]`.
Report obsahuje i doporučený `gaussian_position`, který při této korekci
zachová bod A na místě. Je to volba kotvy, nikoli automatický důkaz správného
zarovnání s realitou. Hodnoty přenes do YAML a restartuj náhled; měření samo
konfiguraci nepřepisuje.

`nyx_ruler` v YAML umožňuje změnit počáteční `size_m`, `position`,
`rotation_deg` (XYZ stupně), `reference_length_m`, `max_length_m`,
rychlosti, krok délky, kalibrační světlo a cestu reportu. Délky/rozsah/krok
musí být násobky 1 mm; maximální podporovaný rozsah je 20 m.

Měřidlo prošlo spolu s náhledem a původní pipeline 37 testy. GPU zkouška
vykreslila devět snímků včetně změn délky 1 → 1,5 → 2 m, přesunu, skrytí,
resetu a přepnutí na robotí pohled. Rozměry skutečných vizuálních vrcholů
souhlasily s nastavenou délkou a průřezem 0,02 × 0,02 m s tolerancí 1e-4 m.
Ověřeno bylo i uložení JSON a korekce měřítka 4 → 3 pro referenci 1,5 m
a měřenou délku 2 m. Ovládání GPU zkoušky bylo zadáno programově.

Po každé změně splat transformace náhled znovu spusť: Nyx tyto parametry
načítá při `scene.build()`, nikoli živě z YAML. Nativní okno samo není
editor splat transformací. `pos` a `look_at` volné kamery slouží pouze
k inspekci; její změnou nekompenzuj špatnou registraci splatu.
Ověř zarovnání z více pohledů. Výsledné tři `gaussian_*` hodnoty přenes
i do příslušných train/eval konfigurací. Splat poskytuje obraz, kolize
nadále určuje geometrie simulace.

`nyx_preview_fps` je horní limit frekvence obnovování (výchozí 30), nikoli
garantovaná rychlost renderu. `nyx_preview_max_frames: 0` znamená neomezený
náhled; kladný počet snímků slouží pro krátké ověření.

Oficiální dokumentace: [nativní okno](https://genesis-embodied-ai.github.io/genesis-nyx/latest/api_reference/generated/gs_nyx_plugin.nyx_renderer.NyxPyRenderer.__init__.html),
[renderování při čtení kamery](https://genesis-embodied-ai.github.io/genesis-nyx/latest/user_guide/advanced/sensor_lifecycle.html),
[omezení změn umístění splatu](https://genesis-embodied-ai.github.io/genesis-nyx/latest/user_guide/advanced/splats.html).

## Nastavení FOV

Pro `NyxGaussianSplatCameraCfg` lze v `cameras.*.params` zadat úhly ve stupních:

```yaml
cameras:
  camera1:
    type: NyxGaussianSplatCameraCfg
    params:
      name: camera0
      width: 128
      height: 128
      fov_horizontal: 65.0
      fov_vertical: 65.0
      gaussian_splat_path: ./config_run/splat/splat_2.ply
      # Ostatní montážní/renderovací parametry zůstávají jako dosud.
```

- `fov_vertical`: svislé FOV výsledného obrazu.
- `fov_horizontal`: vodorovné FOV výsledného obrazu.
- Při zadání pouze jedné osy se druhá dopočítá z `width/height` pro čtvercové
  pixely. Při zadání obou se obě respektují i tehdy, když neodpovídají tomuto
  poměru stran.
- Staré `fov` zůstává aliasem **svislého** FOV; původně nešlo o dvě stejné
  hodnoty, druhá osa se odvozovala z poměru stran. Pokud zadáš zároveň `fov`
  a `fov_vertical`, musí souhlasit. Při přechodu je vhodné staré `fov` odstranit.
- Bez explicitních FOV zůstává projekce odvozená z původního focal length /
  aperture. Obyčejné `PinholeCameraCfg` a jiné backendy tato nová pole nepoužívají.

## Výřez z Full HD

Pro ideální kameru se stejnou pózou, vystředěným optickým středem a čtvercovými
pixely:

```text
1920×1080, VFOV 65° a odvozené HFOV přibližně 97,114°
    → středový crop 1080×1080
    → resize 128×128
```

má stejnou projekci jako přímý render 128×128 s oběma FOV 65°.
Rozdíly vyhlazení a šumu mezi Full HD a přímým malým renderem tím nezmizí.

**Do čtvercového výřezu nezadávej původní vodorovné FOV celého širokého obrazu.**
Například HFOV 97,114° + VFOV 65° při 128×128 zobrazí široký pohled stlačený
do čtverce, tedy jinou geometrii než středový crop. Nezávislé FOV dovolují i
takovou projekci; automaticky neprovádějí deployment crop ani korekci čočky.

Pokud výrobce uvádí horizontální FOV 65° pro celý 1920×1080 obraz, odpovídá
mu při těchto předpokladech vertikální FOV přibližně 39,430°. Středový čtvercový
výřez pak má přibližně 39,430° v obou osách, nikoli 65°.

## Náhled augmentace

`chairman_separate/calibrate_nyx.yaml` má zapnuté
`nyx_preview_augmentation: true`. Okno začíná čistým obrazem. **G** vylosuje
nové parametry jasu, kontrastu, gamma, RGB gains, šumu a rozmazání a zapne jejich
náhled; **V** přepíná mezi čistým a posledním vylosovaným vzhledem. Parametry se
vypíšou do terminálu. Změny vzhledu drží do dalšího G, snímací šum se mění každý
snímek. Ovládání platí pro volnou i robotovu kameru v aktivním ovládacím okně.

Náhled používá stejné funkce a rozsahy z `config_run/visual_augmentation.py`
jako DAgger. G vybírá z augmentovaných případů; při tréninku zůstává podle
`CLEAN_PROBABILITY` polovina epizod/sekvencí čistá. Základní render ani splat se
neupravuje. Augmentovaný náhled se počítá v rozlišení sítě podle
`nyx_preview_augmentation_size: [640,360]` a zvětší do okna, aby šum a blur měly
odpovídající velikost pixelů. Čistý náhled používá původní rozlišení kamery.
Při změně vstupního rozlišení studenta uprav i tuto dvojici.

## Jak se obě osy skutečně vykreslují

Nyx nativně přijímá svislé FOV a rozlišení, nikoli dvě nezávislé ohniskové
vzdálenosti. `metasim/utils/camera_projection.py` z požadovaných úhlů spočítá
poměr stran frusta `tan(HFOV/2) / tan(VFOV/2)` a interní renderovací rozlišení.
Renderer dostane tuto velikost a požadované svislé FOV. Interní obraz pokryje
požadovaný výřez a má alespoň tolik vzorků v každé ose jako výsledný obraz.

`GenesisHandler.get_camera_states()` následně přes
`NyxGaussianSplatRuntime.project_camera_rgb()` převede RGB na požadované
`height×width`. Převod je na stejném zařízení, používá bilineární vzorkování
středů pixelů a opravuje i zlomkový výřez vzniklý celočíselným zaokrouhlením
interního rozlišení. Mřížka se kešuje; uint8 RGB i pořadí envs zůstávají
zachované. Pokud jsou úhly a poměr stran kompatibilní, převod se přeskočí.

Například výstup 128×128 s HFOV 90° a VFOV 60° potřebuje interně 222×128,
nikoli Full HD. V tomto příkladu má výstup kamery 128×128.
Extrémní kombinace vyžadující interní rozměr nad 16384 se odmítnou před
alokací rendereru. Podpora se týká RGB; Nyx zde stejně jako dosud nevrací depth.

`camera.horizontal_fov`, `camera.vertical_fov` a `camera.intrinsics` popisují
výsledný obraz. Změna FOV se musí stejně promítnout do train/eval konfigurací;
současná studentská metadata nekontrolují úhly FOV, pouze rozměry a periodu.

## Ověření

Aktuální standardní train/eval konfigurace však používají široký studentský
vstup **640×360**. Kamera renderuje 640×360 s celkovým HFOV 90° a VFOV
58,715507°. Společný adaptér `config_run/student_vision.py` předá celý obraz
bez ořezu a bez zmenšení; změní pouze pořadí os na NCHW a případně zařízení.
Výsledný vstup má stejné HFOV 90° a VFOV 58,715507° jako raw render.
Tyto úhly jsou stále odhad D455; přesné hodnoty nahraď kalibrací RGB streamu.

Velikost studenta řídí `dagger_image_width` a `dagger_image_height` mimo sekci
kamery. Umožňuje to ponechat raw render oddělený od vstupu sítě. Adaptér ořezává
na cílový poměr stran a případný resize používá plošné filtrování. Například
reálný snímek 1280×720 se pro 640×360 zmenší 2× bez ořezu.
128×128 checkpointy při rekurentní evaluaci použijí původní rozměr z metadat;
nový trénink se standardním configem uloží [3,360,640]. Viz také
[README studenta](dagger_recurrent/README.md#architektura-a-rozhraní).

```bash
python -m unittest metasim.test.test_student_vision config_run.dagger_recurrent.test_student config_run.dagger_recurrent.test_pipeline
python -m unittest metasim.test.test_camera_fov
RUN_GENESIS_CAMERA_TESTS=1 python -m unittest metasim.test.test_genesis_camera_fov
```

CPU testy ověřují převody obou os, intrinsics, starý `fov`, shodu přímé čtvercové
projekce s intrinsics Full-HD cropu a výsledné paprsky proti analytické projekci.
Volitelný GPU test používá čtyři skutečné Nyx kamery a ověřuje i render
640×360 → přímý vstup 640×360 → skutečný ResNet forward/backward na GPU.
Kontroluje parametry rendereru, velikost/dtype obrazů a deferred cache.
Potřebuje lokální G1/scénu/splat a funkční Genesis/Nyx.
