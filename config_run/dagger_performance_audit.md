# Měření DAgger: chairman_separate

Lokální měření 19. 10. 2026, RTX 3060 12 GB, Python prostředí
`roboverse_train`, PyTorch 2.8.0+cu126. Použit skutečný šestipolicy teacher
`run_2026-10-14_19-34-03_chairman_separate` (5 106 560 000 přechodů).

## Výsledek

Největší položkou je vykreslování Nyx. Vedle něj byla v kompatibilitní vrstvě
chyba, která při každém čtení stavu třikrát znovu inicializovala stejnou vizuální
geometrii. Odstranění této práce a dávkové čtení stavů linků zrychlilo skutečný
krátký trénink se 100 prostředími **1,57×**, při zachování kamery, rozlišení,
SPP, počtu gradientních aktualizací i architektury studenta.

| Běh | Čas 45 vektorových kroků | Vektorové kroky/s | Přechody/s |
|---|---:|---:|---:|
| Původní, 100 env | 36,35 s | 1,24 | 124 |
| Opravený, 100 env | 23,14 s | 1,94 | 194 |
| Opravený, 200 env | 43,41 s | 1,04 | 207 |
| Opravený, 50 env | 13,33 s | 3,38 | 169 |
| Opravený, 50 env, FastPathTracer | 14,25 s | 3,16 | 158 |

U 100 env byl původní rozpad času následující. Jde o kumulativní wall time
profilovaných funkcí; CUDA čekání se účtuje funkci, která synchronizuje GPU.
Vnořené časy nelze sčítat s časy jejich rodičů.

| Operace | Původní čas | Podíl celku | Po opravě |
|---|---:|---:|---:|
| Vykreslení kamer Nyx | 19,15 s | 52,7 % | 19,12 s |
| Kompatibilita solveru, zejména inicializace mesh dat | 10,54 s | 29,0 % | <0,001 s |
| Čtení stavů těl | 2,59 s | 7,1 % | 0,07 s |
| Simulace, 5 fyzikálních kroků na akci | 1,03 s | 2,8 % | 1,02 s |
| Učení studenta, 20 gradientních aktualizací | 0,48 s | 1,3 % | 0,49 s |
| Teacher inference | 0,35 s | 1,0 % | 0,35 s |

Po opravě představuje kamera přibližně 83 % času se 100 env a 90 % s 200 env.
Samotný replay buffer, ukládání checkpointu a backpropagation nebyly v tomto
měření dominantní. Dodatečné rendery vznikají i při resetech: 45 akcí se 100 env
vyvolalo 53 renderů dávky, tedy **5 300 jednotlivých renderů**.

## Proč 200 env nepřineslo očekávané zrychlení

Nyx plugin (`NyxCameraSensor._render_current_state`) obsahuje smyčku přes
prostředí a vykresluje je jednotlivě. Každý `NyxPyRenderer.render` navíc provádí
`torch.cuda.synchronize()`. Větší počet prostředí tudíž nezajišťuje paralelní
vykreslování jako u batched fyziky. Žádnou z těchto synchronizací jsem
neodstranil: chrání sdílené buffery a předávání dat mezi simulací a rendererem.

`dagger_store_per_step: 50` zároveň znamená, že 100 i 200 env ukládá stejných
50 nových vzorků za vektorový krok. Počet aktualizací studenta se také nemění.
Při 200 env se tak vykresluje více obrazů bez zvýšení počtu uložených příkladů.
Zvýšení počtu prostředí může pomoci rozmanitosti trajektorií, ale zde výrazně
zhoršuje rychlost sběru těchto 50 vzorků.

Při 50 env se stále ukládá 50 vzorků za krok a student dostává stejných
10 aktualizací po 20 krocích. Naměřený sběr je přibližně 169 nových vzorků/s,
oproti 97 při 100 env a 52 při 200 env. To je **2,73× více uložených vzorků/s
než původní kód se 100 env**. Menší počet prostředí mění rozmanitost souběžných
trajektorií, proto tento výsledek není zárukou stejné výsledné úspěšnosti.

Pro další běh bych začal s `num_envs: 50` a ponechal `dagger_store_per_step: 50`,
batch size 512, 10 aktualizací každých 20 kroků i současný RefPathTracer/SPP 8.
Samotné přepnutí na FastPathTracer se v krátkém kontrolním běhu nevyplatilo
(14,25 s proti 13,33 s). Výchozí trénovací YAML jsem proto neměnil. Větší další
zrychlení vyžaduje úpravu vykreslovací cesty: skutečný batch rendering nebo
omezení renderů při sběru/resetech. Snížení SPP nebylo změřeno a mění obraz;
stejně tak nebyl ověřen vliv jiného rendereru na výslednou úspěšnost studenta.

DAgger `total_timesteps` v `separate_dagger.py` počítá vektorové kroky, zatímco
PPO trainer vykazuje jednotlivé přechody. Student na 370 000 krocích při 100 env
už viděl 37 milionů přechodů a při ukládání 50 vzorků za krok prošlo bufferem
18,5 milionu učitelských příkladů. Pro srovnání různých konfigurací používej
kroky/s, přechody/s a uložené vzorky/s odděleně.

Historický TensorBoard tohoto studenta potvrzuje přibližně 1,36 vektorového
kroku/s v posledních 2 000 krocích víkendového běhu. Krátký původní benchmark
s profilerem tedy odpovídá pozorované pomalosti.

## Změny a ověření

- `metasim/sim/genesis/gaussian_splat.py`: solver s veřejnými
  `get_vgeoms_pos/get_vgeoms_quat` už nezkouší vytvářet stará pole
  `vgeoms_state/vverts_state`. V aktuálním Genesis je tyto inicializace nikdy
  nevytvořily, takže se opakovaly při každém snímku. Renderer už pro tuto verzi
  používal veřejné gettery; starší cesta zůstává zachována.
- `metasim/sim/genesis/genesis.py`: čtyři dávkové gettery entity místo čtyř
  getterů pro každý link. Pořadí linků, souřadné rámce a výběr prostředí zůstávají
  shodné. Ověřeno na skutečném robotu i židli, před a po simulaci a pro
  přeházenou podmnožinu prostředí.
- `metasim/test/test_nyx_solver_compat.py`: regresní test veřejného API bez
  opakované inicializace a zachování staré cesty. Spolu s existujícími DAgger
  integračními testy prošlo 7 testů.
- `config_run/profile_dagger.py`: opakovatelný krátký trénink se skutečným
  teacherem, oddělenými checkpointy, časováním, JSON souhrnem a volitelným
  cProfile. Benchmark neukládá nové curriculum snapshoty.

## Jak měřit na clusteru

Spusť z kořene repozitáře ve stejném Python prostředí jako běžný trénink:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python config_run/profile_dagger.py \
  --num-envs 100 --steps 101 --cprofile --verify-body-states
```

Výsledky budou v novém adresáři `config_run/output/dagger_profile/`:
`summary.json`, `profile.txt`, `training.prof`, použitá konfigurace a studentský
checkpoint. Pro porovnání škálování zopakuj s `--num-envs 50` a `--num-envs 200`.
Bez `--cprofile` získáš měření s menší režií. Volby `--render-mode FastPathTracer`
a `--spp 1` umožňují samostatně změřit vliv rendereru a kvality obrazu.

Benchmark používá kapacitu bufferu 100 kroků místo produkčních 3 000. Rozměr
obrazu zůstává 128×128, batch size 512, sběr 50 vzorků/krok, 10 aktualizací
každých 20 kroků, RefPathTracer a SPP 8. U 100/200 env už první učicí dávka
obsahuje plných 512 vzorků. Rozběh simulace a tři zahřívací kroky jsou mimo
čas tréninku; načtení teachera, první CNN aktualizace, reset a finální uložení
jsou zahrnuty. Běhy jsou krátké a s počáteční beta blízkou 1: ověřují
průchodnost a funkčnost, nikoli výslednou úspěšnost studenta. Časy vznikly na
lokální GPU, nikoli na clusteru. Paměť v JSON je pouze alokace PyTorch, nikoli
celková spotřeba Genesis a Nyx.

Raw profily pro srovnání 100 env: `config_run/output/dagger_profile/n100_baseline.prof`
a `n100.prof`; logy `n100.log` a `n100_fixed.log`. Výsledek pro 200 env je v
`config_run/output/dagger_profile/n200_fixed/`, pro 50 env v `n50_fixed/`.

## Navazující optimalizace kamer

Dávková příprava pozic a orientací kamer a odložení snímků do okamžiku,
kdy je student skutečně spotřebuje, jsou implementované v repozitáři.
Instalované soubory `gs_nyx_plugin` se na disku nemění; adaptér se zapojí
při vytváření kamery. Nativní Nyx renderování stále postupuje po prostředích,
včetně původní synchronizace a kopie snímku před opětovným použitím bufferu.

Kontrolované srovnání na stejné RTX 3060: 100 env, 45 kroků, batch 512,
RefPathTracer, 128×128, SPP 8, stejný teacher a seed:

| Varianta | Celý krátký trénink | Renderů dávky | Jednotlivých snímků |
|---|---:|---:|---:|
| Před úpravou kamer, po předchozích opravách solveru a stavů | 23,47 s | 54 | 5 400 |
| Dávkové pozice/orientace kamer | 19,76 s | 54 | 5 400 |
| Dávkové pozice + odložené vykreslení | 18,30 s | 45 | 4 500 |

Celkem další **1,28× zrychlení**, přibližně 22 % ušetřeného času.
Samotná příprava matic kamer se místo 5 400 jednotlivých volání prováděla
54 dávkovými voláními (s odložením 45). Nejde o odhad z vytížení GPU, ale
skutečně dokončené krátké DAgger tréninky. Neověřují výslednou úspěšnost
studenta ani rychlost na L40S.

`ScenarioCfg.defer_camera_rendering` je obecně vypnuté. `main_separate.py`
je zapíná pro Genesis při `train_dagger/load_and_train_dagger`, s možností
vypnutí přes `dagger_defer_camera_rendering: false`. L40 config obsahuje
explicitní `true`. PPO a evaluace nadále používají původní způsob čtení
kamer; dávkový výpočet transformací Nyx využívají také.

Při odloženém vykreslení obsahuje běžný `get_states()` fyzikální stav bez
kamer. `_student_inputs` si vyžádá `get_camera_states()`, které aktualizuje
fyziku, vykreslí a kešuje snímek do dalšího obnovení fyzikálního stavu.
Po resetu se keš invaliduje i bez změny `scene.t` a obnoví se vizuální
geometrie přes `force_render=True`. Student tak dostává obraz až po dokončení
resetů. Nezavádí se opakování starého snímku mezi různými kroky simulace.
Čas kamer se v profileru přesunul z `env_step` do `student_inputs`; pro
porovnání rychlosti je rozhodující celkový čas tréninku.

Přepínač `--eager-cameras` v `profile_dagger.py` umožňuje porovnat variantu
s vykreslováním během čtení fyzikálních stavů. Souhrny a profily jsou v
`config_run/output/dagger_profile/camera_before/`, `camera_batch_only/`
a `camera_batch_deferred/`.

Regresní test `test_nyx_camera_batch.py` porovnává transformace s původním
výpočtem pro statickou i připojenou kameru, více prostředí a sdílený výstupní
buffer. Skutečný test resetů/geometrie s lokálními assety se spouští takto:

```bash
RUN_GENESIS_CAMERA_TESTS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest metasim.test.test_genesis_deferred_camera
```

SPP zůstává 8. Snížení SPP ani skutečný nativní batch renderer nejsou součástí
této změny. Nativní batch by musel přijmout více stavů scén/kamer a vrátit
`[N,H,W,3]` bez sériové obsluhy každého prostředí z Pythonu; náklady na výpočet
všech pixelů zůstanou, ale část přípravy a spouštění práce by se mohla sdílet.

Ověření této úpravy: prošlo 8 rychlých testů a samostatně skutečný Genesis/Nyx
GPU test. Ten kontroluje, že několik resetů nevyvolá render, opakované čtení
stejného stavu využije keš a posun robota o 0,25 m při stejném `scene.t`
aktualizuje kameru i geometrické buffery rendereru. Zároveň porovnává skutečné
vstupní buffery kamery/geometrie s původní renderovací cestou Nyx. Nejde o test
bitové shody pixelů stochastického rendereru.
