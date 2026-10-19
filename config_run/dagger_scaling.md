# L40S: měření maxima propustnosti DAggeru

Původních 50 env bylo konzervativní nastavení odvozené z lokálních měření,
kdy se ukládalo vždy jen 50 vzorků za krok. Nebylo to ověřené optimum L40S.
Při vyšším počtu prostředí je důležité ukládat všechna prostředí a úměrně
změnit učení. Jinak vznikají další rendery bez dalších uložených příkladů.

`train_dagger_cluster_l40.yaml` nyní používá kandidáta s 200 env. Automatické
porovnání na skutečné přidělené kartě je spolehlivější než odhad podle VRAM:

```bash
python config_run/benchmark_dagger_scaling.py --train-best
```

Tímto příkazem lze nahradit poslední Python příkaz ve Slurm skriptu. Postupně,
nikoli souběžně, změří 50, 100, 200 a 400 env, vytvoří samostatné YAML pro
nejlepší úspěšně změřenou variantu a spustí její trénink. Bez `--train-best`
jen změří kandidáty a vypíše příkaz ke spuštění. Kandidáty lze upravit pomocí
`--envs 100 200 400`. Benchmark je určen pro volnou GPU v rámci daného jobu;
souběžný jiný trénink by měření zkreslil.

## Co se škáluje

| Env | Uložených vzorků/krok | Batch | Aktualizací / 20 kroků | Kapacita bufferu | Produkční kroky |
|---:|---:|---:|---:|---:|---:|
| 50 | 50 | 1024 | 5 | 500 000 | 1 000 000 |
| 100 | 100 | 2048 | 5 | 500 000 | 500 000 |
| 200 | 200 | 4096 | 5 | 500 000 | 250 000 |
| 400 | 400 | 8192 | 5 | 500 000 | 125 000 |

Všechny varianty mají stejný produkční rozpočet **50 milionů přechodů**,
přibližně 23 GiB replay bufferu a poměr 5,12 vzorků použitých k učení na jeden
nově uložený příklad. Lineární beta nyní používá přepočtený cílový počet kroků; frekvence checkpointů se také přepočítává,
aby jejich průběh odpovídal stejnému množství nasbíraných dat. Learning rate
zůstává 0,0003. Větší batch mění dynamiku optimalizace a počet optimizer kroků;
srovnání propustnosti proto nezaručuje stejnou výslednou kvalitu studenta.

Počítadlo vektorových kroků bude s více env přibývat pomaleji, i když může
přibývat více použitelných dat za sekundu. Rozhodující ukazatel pro výběr je
`Speed/stored_samples_per_second` při odpovídajícím učení, nikoli samotné
kroky/s nebo procento vytížení GPU. Vyšší počet env může také zvýšit rozmanitost
souběžných trajektorií. Nyx nadále renderuje obrazy po jednotlivých prostředích,
takže vyšší počty nemají zaručené zrychlení.

Každý krátký benchmark má standardně 201 kroků. Výběr používá medián plných
logovacích intervalů po prvním intervalu (defaultně kroky 101–200); první
optimalizace a závěrečný krátký interval jsou z hodnocení vynechány. Krátké
benchmarky používají stejnou malou replay kapacitu mezi kandidáty, produkční
YAML vrací kapacitu 500 000 vzorků. Chybové/OOM běhy se z výběru vyřadí.
Nejde o měření konvergence ani garanci proveditelnosti všech kandidátů na všech
GPU. Objem GPU paměti v benchmarkovém JSON zahrnuje jen alokace PyTorch.

Výsledky a jednotlivé logy jsou v `config_run/output/dagger_scaling/<UTC>/`.
Výsledný YAML má název
`chairman_separate/train_dagger_cluster_l40_tuned_<UTC>.yaml` a vlastní výstupní
adresář. Benchmark výběru maxima zatím nebyl spuštěn na L40S; lokálně běžící
uživatelský trénink jsem kvůli měření nezastavoval ani nenahrazoval.

## Warning RNN vah a místní prostředí

Warning pochází z předtrénované LSTM v `config_run/policy/motion.pt`, která
řídí nohy robota. Student je CNN. TorchScript při načtení na CUDA neobnovil
kompaktní cuDNN layout vah. Při vytváření `G1MotionPolicy` se nyní jednou
provede `torch.nn.RNNBase.flatten_parameters` na uložených atributech LSTM.
Metoda není do TorchScript modelu exportovaná, proto ji volá obal z PyTorch.
Váhy, architektura, paměť resetů ani výstupy se tím nemění. Warnings se
nepotlačují. Pokud PyTorch nahlásí neslučitelnou knihovnu, zapíše se konkrétní
příčina a zůstane původní inference cesta.

Ověřeno na GPU: před opravou warning vznikl, po opravě nevznikl při opakovaných
voláních ani změnách batch size/resetu; akce se shodují s původní cestou a váhy
sdílí jeden storage. Prošlo pět testů motion policy v `roboverse_train`.

V místním `metasim_backup` je navíc skutečný nesoulad knihoven: PyTorch
2.8.0+cu126 očekává cuDNN 9.10.2, ale `.so` z tohoto prostředí vrací 9.1.0,
i když metadata balíčku uvádějí 9.10.2.21. V `roboverse_train` se načítá správná
verze 9.10.2 a oprava funguje. Prostředí běžícího procesu jsem nepřeinstalovával;
knihovny by se za jeho běhu mohly smíchat. Při dalším spuštění na PC lze použít
již funkční `roboverse_train`; pro pokračování existujícího studenta nastav
`load_and_train_dagger` a jeho checkpoint s `.training.pth` sidecarem.

Kontrola prostředí před novým během:

```bash
python -c 'import torch; print(torch.__version__); print(torch.backends.cudnn.version())'
```

Změna Python wrapperu ani aktivace jiného prostředí se do již spuštěného procesu
zpětně nepromítne. `recent=0/0` v konzoli znamená, že v posledním okně neskončil
žádný pokus; samo o sobě nejde o chybu tréninku.
