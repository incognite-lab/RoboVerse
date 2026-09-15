# ChairMan Multi-PPO: úpravy tréninku 2026-09-15

Úpravy navazují na `chairman_full_training_audit.md`. Odměny a podmínky
úspěchu stage 0 zůstávají zachované. Policies nadále zadávají absolutní
cílové pozice kloubů, nikoli odchylky od inicializační polohy.

## Odměny stages 1–5

Bonus za splnění stage zůstává **+500**. Penalizace neúspěšného ukončení je
pro stages 1–5 **−250**, pro stage 0 zůstává původní globální nastavení −1.
Přechod včetně bonusu patří policy, která akci provedla, a ukončuje její GAE
sekvenci. Critic následující policy se do tohoto přechodu nezapočítává.

| Stage | Hlavní kladné váhy | Další důležité váhy |
| --- | --- | --- |
| 1 | dosahování 0,60; orientace 0,20; držení rukou 0,20; přesný cíl 0,20 | změna akce −0,03; pohyb kloubů −0,05; stabilita trupu −0,10 |
| 2 | zavírání 1,0; kontakt 0,5; držení rukou 0,5 | změna akce −0,005; pohyb kloubů, chůze, stabilita trupu po −0,01 |
| 3 | tah 0,75 | ztráta úchopu: váha +0,10 na zápornou funkci; drift rukou −0,10 |
| 4 | otevírání prstů 1,0 | nestabilita židle/robota −0,10 |
| 5 | spouštění paží 1,0 | nestabilita −0,10; opětovné zavírání prstů −0,05 |

Společné penalizace stages 3–5: změna akce −0,01; pohyb kloubů −0,02;
povely chůze −0,02; náklon trupu −0,04. Úplné hodnoty jsou v
`ChairMan_multi.py`: `STAGE1_REWARD_WEIGHTS`, `STAGE2_REWARD_WEIGHTS`,
`LATE_STAGE_REWARD_WEIGHTS` a skalární váhy specifických funkcí.

Při `gamma=0.995` se odložení bonusu +500 o krok projeví ztrátou 2,5.
Horní meze kladného průběžného rewardu stages 1–5 jsou pod touto hodnotou.
Zároveň jsou maximální průběžné penalizace menší než přínos odložení stejné
budoucí penalizace −250. Testy kontrolují i rezervu pro průběžný reward
terminálního kroku. Tím odstraňujeme konkrétní motivaci čekat před cílem
nebo urychlovat stejný nevyhnutelný neúspěch. Nejde o důkaz konvergence
ani o porovnání všech možných trajektorií; při změně gamma je nutné tyto
rozpočty přepočítat.

Stage 3 už při stání na počátku správné dráhy nedostává velkou zápornou
odměnu za nedosažení cílové rychlosti. Samotný tah dává výrazně více,
opačný pohyb je záporný a v cíli se odměňuje zastavení.

Záporný celkový průměr na začátku může být správný: neúspěšné epizody
obsahují −250. Posuzovat je potřeba zvlášť průběžné složky, úspěšnost a
důvody ukončení. Cílem není kladný reward za libovolné chování.

## Přechody, reset a učení

- Čerstvý souběžný trénink začíná stage 0. Staré snapshoty na disku
  neodemknou policies, které ještě nebyly inicializované v aktuálním běhu.
- Při prvním přechodu do nové stage se převezme actor předchozí policy.
  Critic zůstává samostatný. Převod respektuje stage indikátor v pozorování.
  Již trénovaná policy se při dalších přechodech ani resume nepřepisuje.
- Minimální směrodatná odchylka zděděných akcí je 0,10; pro prsty ve
  stages 2 a 4 je 0,20. Jde o jednotky akcí policy. Předchozí actor tak
  může předat polohu rukou, aniž by téměř zastavil průzkum zavírání/otevírání.
- Historie reward funkcí se při přechodu resetuje až po výpočtu odměny
  předchozí stage. Staré rozdíly poloh či uzavření prstů neovlivní nový úsek.
- Před zmrazením se actor během ověřování nemění; vyžadují se opakované
  úspěšné výsledky a alespoň doba jednoho timeoutu dané stage. Teprve potom
  se zmrazí. Nezmrazujeme nově aktualizovaný actor na základě výsledků
  jeho předchozí verze. Toto zmrazování se týká souběžného tréninku.

Detaily přenosu a samostatného tréninku: `chairman_actor_inheritance.md`.
Zdrojový actor se přenáší při prvním přechodu; nemusí již být plně naučený.

## Podmínky ukončení

- Stage 2/3: drift rukou se může napravit během 0,15 s; stage 3 stejně
  toleruje krátký výpadek kontaktů. Úspěch stále vyžaduje skutečný úchop
  a ruce uvnitř povoleného prostoru.
- Stage 4/5: krátké překročení limitu rychlosti má toleranci 0,20 s.
  Velký posun židle, pád a timeout nadále ukončují úsek.
- Úspěch stage 4/5 vyžaduje 0,10 s stabilního splnění. Stage 5 kontroluje
  kromě spuštěných paží také otevřené prsty.
- Kontrola klidu robota ve stages 3–5 používá vodorovnou rychlost;
  běžné svislé kolísání samo nezpůsobí neúspěch.
- Stage 4/5 mají každá 4 s. Globální horizont je 6000 kroků, tedy při
  dt=0,002 a decimation=5 celkem 60 s, nad součtem stage timeoutů 48 s.

## Diagnostika a ověření

TensorBoard nově obsahuje vážené složky `stage_N/reward_terms/Function`
a důvody selhání `stage_N/failure_masks/Reason`, průměrované přes přechody
příslušné stage. Masky selhání tedy nejsou procenta neúspěšných epizod.

V prostředí `metasim_backup` prošlo 45 testů, jeden CUDA test byl přeskočen.
Testy pokrývají přiřazení bonusu/GAE, přenos actoru, curriculum, rozpočty
odměn, progres tahu, zotavení úchopu, finální otevření prstů a zmrazování.
Plný trénink v Genesis nebyl spuštěn: fyzikální dosažitelnost a výslednou
úspěšnost musí ověřit nový běh. Známá motivace k čekání ve stage 0 zůstává
mimo tuto úpravu podle zadání.

Změny se projeví po restartu tréninku. Resume zachová již naučené i zmrazené
policies; opravy zpětně nepřepíšou jejich váhy ani jejich inicializaci.
