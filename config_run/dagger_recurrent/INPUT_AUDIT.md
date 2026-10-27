# Audit senzorových vstupů nového studenta — 23. 10. 2026

Čtení kloubových rychlostí a výpočet IMU příznaků prošly kontrolou zdrojového
kódu a numerickým testem skutečné dynamiky Genesis. Produkční sestavení vstupů
nebylo při původním auditu měněno. Reálný robot ani jeho telemetrie nebyly
v tomto auditu dostupné. Níže uvedená měření byla původně provedena s `pelvis`.

Následná změna přepíná zdroj na `imu_in_pelvis`, včetně jeho vlastní orientace
a úhlové rychlosti. V `g1_mygym.urdf` jej s pánví spojuje fixed joint s
`xyz="0.04525 0 -0.08339"`, `rpy="0 0 0"`. Gyro a projected gravity jsou
proto pro tuto URDF shodné; posun senzoru ovlivňuje lineární veličiny, které
student do IMU vstupu nyní nezahrnuje. Nová regrese záměrně nastavuje odlišné
stavy `pelvis` a `imu_in_pelvis` a ověřuje, že adaptér skutečně použije sensor
link. Zdroj je nově zaznamenán i v metadatech checkpointu.
Stejný fyzikální audit byl následně spuštěn znovu s `imu_in_pelvis` a prošel;
naměřené chyby níže zůstaly stejné. JSON nyní obsahuje
`"imu_body_name": "imu_in_pelvis"`.

## Cesta kloubových dat

`Genesis RigidEntity.get_dofs_position()/get_dofs_velocity()` →
`GenesisHandler._get_states()` → `RobotState.joint_pos/joint_vel` →
`StudentInputs.read()` → normalizace v `RecurrentVisionStudent.forward()`.

- Free-base G1 má 49 rychlostních DOF: 6 základny + 43 kloubů. Handler
  odstraňuje prvních šest z obou polí poloh/rychlostí. Názvy kloubů v tomto
  modelu odpovídají DOF indexům 6–48; ověřeno přímo na instanci Genesis.
- Všechny konfigurované klouby jsou rotační: q v rad, dq v rad/s. Jedná se
  o skutečné simulované rychlosti, nikoli požadované rychlosti/cíle serv.
- Adaptér používá jeden explicitní seznam názvů pro q i dq. Přeskupení je
  potřeba: simulační pořadí umisťuje levé prsty před pravou paži, konfigurace
  modelu má obě paže před prsty. Ověřena správnost všech 43 sloupců.
- Rychlosti se nenásobí ani nedělí dt/decimation; Genesis již vrací derivaci
  za sekundu. Model je teprve následně škáluje násobením 0,05.
- `BaseSimHandler.simulate()` invaliduje cache fyzikálního stavu před krokem;
  běžné resety ji rovněž invalidují. Opakované čtení v jednom kroku není nový
  časový vzorek. Derivace poloh přes hranici resetu nemá význam rychlosti.

## Cesta IMU příznaků

`GenesisHandler.get_body_states()` skládá `[pos, quat, lin_vel, ang_vel]`:

| Část pelvis body state | Význam |
|---|---|
| `3:7` | Orientace WXYZ z `get_links_quat()` |
| `10:13` | Úhlová rychlost z `get_links_ang()`, world frame, rad/s |

`StudentInputs` najde `pelvis` podle názvu a vypočte:

```text
gyro_body = R(pelvis_quaternion).T @ omega_world
gravity_body = R(pelvis_quaternion).T @ [0,0,-1]
imu = [gyro_body_x,y,z, gravity_body_x,y,z]
```

Jedná se o ideální virtuální IMU z fyzikálního stavu. Nečte se `sensors["gyro0"]`
ani `GyroSensorCfg`; nepřidává se šum, offset, zpoždění či výpadky. Vstup
neobsahuje lineární zrychlení z akcelerometru. Gravitace je bezrozměrný
jednotkový směr dolů v souřadnicích těla, nikoli velikost 9,81 m/s².
Runner nastavuje následné škálování gyra 0,25, gravitace zůstává beze změny.
Quaternion vyjadřuje orientaci linku vůči světu v současné scéně s výchozí
identity montážní rotací URDF; audit není potvrzením libovolných odlišných
konvencí jiného backendu nebo jiného montážního offsetu.

Existující `metasim/cfg/sensors/gyro.py` vrací absolutní Eulerovy úhly a používá
výřez `body_state[...,2:6]`. To neodpovídá skutečnému gyroskopu ani quaternionu
ve výše uvedeném body-state layoutu. Pro nového studenta to nemá vliv, protože
tuto třídu nepoužívá. Tento audit její implementaci nemění.

## Numerická zkouška Genesis

Použit skutečný G1 URDF, dva paralelní roboty, CPU backend a dt = 0,001 s.
Pro izolaci převodů byly vypnuté kolize, gravitace a kloubové limity; nešlo
o benchmark chůze nebo validaci parametrů dynamiky vůči reálnému robotu.
Četl se skutečný `GenesisHandler` a produkční `StudentInputs`, s umělým RGB
snímkem pouze pro splnění kamerového rozhraní; Nyx/renderování se netestovaly.
CPU inicializace byla provedena před importem handleru a jeho druhé automatické
GPU `gs.init` bylo v testovacím procesu potlačeno. Produkční handler se neměnil.

Nastaveny rozdílné polohy a znaménkové rychlosti všech 43 kloubů ve dvou envs.
Přidány i nenulové translační rychlosti základny pro ověření správného výřezu.
Po skutečném kroku se porovnalo `(q_after-q_before)/dt` s dq na konci kroku.

| Kontrola | Největší absolutní chyba |
|---|---:|
| Načtení všech nastavených q podle názvů | 0 rad |
| Načtení všech nastavených dq podle názvů | 0 rad/s |
| dq proti numerické změně q za skutečný krok | 1,46 × 10⁻⁵ rad/s |
| Gyro převod proti nezávislému SciPy výpočtu, pět orientací | 1,20 × 10⁻⁷ rad/s |
| Směr gravitace proti SciPy, pět orientací | 1,20 × 10⁻⁷ |
| World omega proti změně quaternionu za skutečný krok | 7,98 × 10⁻⁵ rad/s |
| Samostatná algebra rotace, 1000 náhodných quaternionů, float64 | 1,78 × 10⁻¹⁵ |

Průměrná chyba kloubové derivace byla 8,04 × 10⁻⁷ rad/s. Zkoušeny identity,
yaw 90°, roll 90°, pitch 90° a kombinovaná rotace [25°, −35°, 70°].
Směry gravitace odpovídaly očekávání:

| Natočení | Gravitace v těle, přibližně |
|---|---|
| Vzpřímený | `[0,0,-1]` |
| Pouze yaw 90° | `[0,0,-1]` |
| Roll +90° | `[0,-1,0]` |
| Pitch +90° | `[1,0,0]` |

Do `test_pipeline.py` byly doplněny nezávislé regresní testy rotace pomocí
SciPy a kontrola znamének a délky vektoru gravitace. Fyzikální probe a plný
JSON z tohoto spuštění jsou v `/tmp/audit_recurrent_student_inputs.py` a
`/tmp/recurrent_student_input_audit.json`; jde o dočasné artefakty.

## Co je a není potvrzeno pro sim2real

Oficiální Unitree `unitree_rl_gym/deploy/deploy_real/deploy_real.py` čte
`motor_state[i].q`, `motor_state[i].dq`, `imu_state.quaternion` (WXYZ) a
`imu_state.gyroscope`; počítá z quaternionu projected gravity. Jeho G1 config
uvádí `imu_type: pelvis`, `dof_vel_scale: 0.05`, `ang_vel_scale: 0.25`.
To odpovídá významu a škálování našich vstupů; není to ověření konkrétního
reálného robota, senzorů jeho rukou ani používaného externího klienta.

Zdroje:
- https://github.com/unitreerobotics/unitree_rl_gym/blob/main/deploy/deploy_real/deploy_real.py
- https://github.com/unitreerobotics/unitree_rl_gym/blob/main/deploy/deploy_real/configs/g1.yaml
- https://github.com/unitreerobotics/unitree_rl_gym/blob/main/deploy/deploy_real/common/rotation_helper.py

Pro nasazení je nutné ověřit všechny motor IDs/názvy a znaménka, jednotky,
IMU-to-pelvis montáž, pořadí quaternionu a jeho směr transformace a časovou
synchronizaci. Reálné gyro v body frame se nesmí podruhé převádět world→body.
Rychlosti prstů musí pocházet z odpovídajícího rozhraní rukou; jejich dostupnost
a vazbu na 14 simulačních kloubů tento audit nepotvrzuje. Stávající starý
`deploy_dagger_student_real.py` nové q/dq/IMU rozhraní dosud nepřipojuje.

Ideální signály v simulaci neprokazují shodu šumu, biasu, filtrace, zpoždění,
mechanické odezvy ani kontaktů s realitou. K tomu je potřeba reálný časovaný
log q/dq, IMU a povelů. Porovnávat rad/s a body-frame gyro před normalizací;
stejné uložené normalizační buffery pak použít při skutečné inferenci.
