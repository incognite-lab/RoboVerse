# L40 / L40S: hledání nastavení pro rekurentní DAgger 640×360

Benchmark `config_run/benchmark_dagger_scaling.py` používá aktuální ResNet-18
+ GRU, skutečných šest teacher politik a Nyx RGB **640×360 bez ořezu a resize**.
Výchozí clusterový config `chairman_separate/train_dagger_cluster_l40` je nyní
rovněž pro `train_dagger_new`, nikoli pro původní CNN 128×128. Lokální config
pro RTX 3060 se benchmarkováním nemění.

Spusť v přiděleném GPU jobu v prostředí Genesis/Nyx z kořene repozitáře:

```bash
python config_run/benchmark_dagger_scaling.py
```

Příkaz změří kandidáty, opakovaně ověří finalisty a připraví samostatný config
s příkazem ke spuštění. Dlouhý produkční trénink spustí pouze explicitní volba:

```bash
python config_run/benchmark_dagger_scaling.py --train-best
```

Každý pokus má vlastní proces a log. CUDA kontext po jeho skončení zanikne,
takže OOM jednoho kandidáta nezničí zbývající měření. Timeout má výchozí hodnotu
3600 sekund na pokus; měření větších kandidátů může trvat dlouho. Při timeoutu
se worker ukončí a kandidát se nevybere. Celé výchozí hledání může trvat hodiny.

## Co hledá

1. Počet envů 16, 32, 64, 128, 256, 512 a doplnění bodů poblíž nejrychlejších.
2. U nejlepších variant počet sekvencí v optimizer batch: 2, 4, 8, 16, 32, 64.
3. Velikost dávky obrazového encoderu: 4, 8, 16, 32, 64 a 0 = všechny snímky.
   Porovnává zapnutý i vypnutý gradient checkpointing. Vypnutí šetří výpočty,
   ale vyžaduje více VRAM; vhodné nastavení se musí skutečně změřit.
4. Počet CPU vláken 4 a 8, protože Nyx/physics mohou limitovat zásobování GPU.
5. Dva čerstvé potvrzovací běhy pro každého ze dvou finalistů. Vítězí nejvyšší
   medián potvrzených uložených vzorků/s, při zachovaném množství učení.

Zúžený průchod například:

```bash
python config_run/benchmark_dagger_scaling.py \
  --envs 32 64 128 \
  --sequence-batches 8 16 32 \
  --visual-batches 8 16 32 0 \
  --threads 4 8 --top-k 1 --confirm-runs 2
```

Parametry jsou seznamy, nikoli požadavek na souběžné procesy. Vyhledávání je
postupné a zaměřuje další měření na dosavadní nejlepší kandidáty. Neprochází
celý kartézský součin všech možností a negarantuje globální matematické optimum.
Pokud je nejlepší počet envů na hranici seznamu, rozsah lze dále rozšířit;
replay však stále musí pojmout burn-in a celou učenou sekvenci.

## Férové srovnání a paměť

- Všechny simulované envs se také ukládají do replaye. Více prostředí tedy
  nevyrábí jen další nepoužité renderované obrázky.
- Poměr učených snímků / nových uložených snímků je přesně zachovaný podle
  základního configu. Počítá se jako `updates × sequence_batch × sequence_length
  / (train_every × store_per_step)`. Burn-in je další výpočet, ne učený cíl.
- Při změně učicí dávky se upraví počet aktualizací a případně interval učení,
  aby nevznikl falešný vítěz, který dělá méně učení. Příliš dlouhé intervaly
  se vynechají (`--max-train-every`, standardně 100).
- Celkový produkční rozpočet přechodů, průběh bety a frekvence checkpointů
  se přepočítávají s počtem envů. Kvůli celočíselným vektorovým krokům může
  nastat malé zaokrouhlení. Learning rate a fyzika zůstávají stejné.
- Výchozí replay má **8 GiB na CUDA** (`--replay-gib 8 --replay-device cuda`).
  Rozpočet počítá RGB uint8, propriocepci, cíle, skrytý stav a reset masky.
  Kapacita se zaokrouhlí dolů na celé řádky envů. Profil i produkce alokují
  stejnou plnou kapacitu: žádný malý testovací buffer, který by skryl OOM.
- GPU zatížení a celkovou použitou paměť včetně Nyx/Genesis sbírá `nvidia-smi`
  jednou za sekundu. Zvlášť se ukládá PyTorch peak allocated/reserved.
  Využití celé paměti je vzorkované; velmi krátkou špičku může monitoring minout.
- Standardní `--max-memory-fraction .92` ponechává alespoň 8 % rezervy podle
  naměřené paměti. Kandidát bez této rezervy nebo bez GPU telemetrie se nevybere.
  Pro CUDA replay rozpočet 8 GiB počítá s L40/L40S 48 GB, nikoli s lokální 3060.

Výchozí clusterový kandidát má 64 envs, batch 16 sekvencí, encoderovou dávku 4,
checkpointing zapnutý a přibližně 8 GiB replaye. Je to start hledání, nikoli
naměřené optimum. Původní poměr z lokální konfigurace je 2 učené snímky na nový
uložený snímek; skutečný poměr benchmark vždy přečte ze zvoleného configu.

## Měření a výsledky

Profilování probíhá na expertově rollout (`beta=1`), aby změny krátkého rozpočtu
nepřepínaly kolektor na náhodně inicializovaného studenta. Student přesto dělá
skutečný forward, backward i optimizer kroky. Produkční beta se zachová ze
základního configu; průběh pozdějších rolloutů proto může mít jinou rychlost.

První logovací interval se vynechá. Délka dalších intervalů se zarovná na celé
cykly učení a měří se nejméně dva další plné intervaly. Výchozí minimum je
301 kroků, v případě potřeby se prodlouží. Z výběru se vyřadí běhy s chybou,
timeoutem, neplatnou loss, bez aktualizací studenta, s nesprávným rozlišením,
poměrem učení nebo změněnými nastaveními oproti produkčnímu kandidátovi.

Výběr maximalizuje rychlost celého trénovacího cyklu. Procento GPU utilization
je diagnostika: 100 % není zaručených, pokud je limitem sériové renderování,
CPU nebo přenos dat. Více práce přidané jen kvůli utilization by změnilo úlohu.
Změny optimizer batch mění dynamiku optimalizace, takže rychlostní vítěz musí
stále projít evaluací úspěšnosti studenta.

Výstupy v `config_run/output/dagger_scaling/<UTC>/`:

- `search.json`: rozsahy a limity hledání.
- `*.yaml`, `*.log`: přesná konfigurace a log každého pokusu.
- `<pokus>/config.json`, `summary.json`, TensorBoard: skutečně použitý setup,
  rychlosti sběru/učení, GPU telemetrie a inkluzivní časování jednotlivých částí.
- `results.json`: průběžné výsledky včetně neúspěšných běhů.
- `best.json`: potvrzený vítěz a cesta k výslednému YAML.

Výsledný `chairman_separate/train_dagger_cluster_l40_tuned_<UTC>.yaml` obsahuje
přesně vybrané parametry včetně CPU vláken, encoderové dávky, checkpointingu,
kapacity a zařízení replaye. Má vlastní adresář modelů a TensorBoardu.

`profile_dagger.py` automaticky rozpozná starého/nového studenta podle režimu.
Pro reprodukci konkrétního kandidáta podporuje `--config-file cesta.yaml`;
nezadané velikosti dávek, replaye a další volby převezme z tohoto souboru.
Legacy benchmark zůstává dostupný přes `--config` ukazující na původní config
s `train_dagger` a ukládáním všech envů; u něj se porovnává počet prostředí.

## Ověření bez L40

```bash
python -m unittest metasim.test.test_dagger_scaling metasim.test.test_dagger_metrics \
  config_run.dagger_recurrent.test_student config_run.dagger_recurrent.test_pipeline
```

Testy kontrolují skutečné bajty bufferu, poměr učení, škálování, celý řetězec
výběru/ověření a předání přesných parametrů do produkčního configu. Testovací
worker simuluje rychlosti, chyby a neplatná měření; tím se nezíská optimum L40.
