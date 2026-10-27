# Samostatné vodorovné a svislé FOV v Genesis/Nyx

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
