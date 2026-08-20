# Contest audit — `logs/contest_batch_000.txt` (studies 000–099)

Every report read in its original language (English, Spanish, Turkish, Greek, German, Bulgarian,
Dutch, Croatian, French). Twelve findings × 100 studies = **1200 cells examined**.

Rulebook applied exactly as given: `YES` asserted · `NO` denied / described normal-intact-preserved ·
`UNK` never named or explicitly uncertain. Severity `1` minimal/trace/small/mild/slight/grade 1 ·
`2` unqualified or moderate · `3` marked/large/severe/extensive/complete/full-thickness/grade 3-4.

Two policies were fixed before scoring, because the rulebook does not settle them and consistency
matters more than my preference. Both were derived from the reader's OWN dominant convention, so a
contest below is an internal inconsistency, not a difference of definition:

- **Meniscal degeneration without tear = YES sev 1.** The reader does this in 002, 039, 054, 059,
  062, 077, 085, 087, 088, 090. Cells contested only where the same wording got `NO`.
- **Contusion is a marrow-signal finding; Fracture is a structural one.** "Marrow signal normal" ⇒
  Contusion `NO`, Fracture `UNK` (the reader's own dominant behaviour: 005, 011, 013, 023, 028, 045,
  053, 056, 076, 087, 098). "Bones normal / bone integrity normal / no fracture" ⇒ Fracture `NO`
  (024, 030, 054, 059, 062, 067, 086).

## Contested cells (48)

| # | Study | uid (tail) | Finding | Current | Mine | Evidence (original language) | Reason |
|---|-------|-----------|---------|---------|------|------------------------------|--------|
| 1 | 000 | …195817260 | Contusion | NO 0 | UNK 0 | whole report: "Rotura de menisco interno. Signo de necrosis avascular subcondral en el cóndilo femoral medial. Artrosis femorotibial medial. Derrame." | Contusion/marrow never named or denied |
| 2 | 000 | …195817260 | Fracture | NO 0 | UNK 0 | same four-sentence report | fracture never named or denied |
| 3 | 007 | …738306606 | Lateral OA | NO 0 | UNK 0 | "Артикулиращият хрущял на фемура и тибията е изтънен от **медиалната** страна"; conclusion "Остеоартритни промени, **по-изразени** от медиалната страна" | only the medial side is described; "more pronounced medially" is not a lateral denial |
| 4 | 007 | …738306606 | Fracture | UNK 0 | NO 0 | "Костите, формиращи дясната коляно са със **непроменена структура**" | structural normality of bone = denial; identical Bulgarian phrasing scored NO in 085 |
| 5 | 008 | …288618513 | Fracture | NO 0 | UNK 0 | "Diz eklemini oluşturan kemik yapıların **sinyal intensiteleri** doğaldır" | signal-only statement; does not answer fracture (cf. 032, 046 where the same sentence gave UNK) |
| 6 | 010 | …965356394 | Contusion | NO 0 | UNK 0 | report mentions only "mild underlying bony change" / "minor underlying bony change" | marrow never assessed; degenerative bony change is not a contusion denial |
| 7 | 012 | …185417806 | Fracture | NO 0 | UNK 0 | no fracture sentence anywhere in the report | never named or denied |
| 8 | 014 | …132965018 | Baker's | NO 0 | UNK 0 | no popliteal/Baker sentence in the report | never named or denied |
| 9 | 015 | …156266866 | Fracture | NO 0 | UNK 0 | "οστεοχόνδρινο ξένο σώµα … έχει αποσπαστεί από την έσω µοίρα … της επιγονατίδας" | fracture never named; an avulsed osteochondral body argues against a denial |
| 10 | 017 | …850458401 | Lateral OA | UNK 0 | YES 2 | "teška gonartroza… Degenerativno stanjene **ostale** zglobne hrskavice uz mjestimične fisure" | the same span was used to score PF OA YES 2; "ostale" covers the lateral compartment too |
| 11 | 025 | …636269365 | Medial OA | UNK 0 | NO 0 | "**Eklem kıkırdakları** ve kemikler patellofemoral eklem dejenrasyonu ile uyumlu fokal kıkırdak kayıpları **dışında normal**" | cartilages declared normal except patellofemoral |
| 12 | 025 | …636269365 | Lateral OA | UNK 0 | NO 0 | same sentence | same |
| 13 | 025 | …636269365 | Contusion | UNK 0 | NO 0 | "…ve **kemikler** … dışında **normal**" | bones declared normal; identical template scored NO in 030, 067, 086 |
| 14 | 025 | …636269365 | Fracture | UNK 0 | NO 0 | same sentence | same |
| 15 | 026 | …629995449 | Contusion | UNK 0 | NO 0 | "Ostale prikazane **koštane** … strukture uredne su morfologije i **intenziteta signala**" | explicit normal bone signal |
| 16 | 029 | …594649149 | Contusion | YES 1 | NO 0 | "**No hay alteraciones de señal significativas de la médula ósea**"; the only bone wording is "mínimos **cambios óseos secundarios**" | marrow explicitly normal; secondary degenerative bone change is not a contusion |
| 17 | 029 | …594649149 | Fracture | NO 0 | UNK 0 | only the marrow sentence above | marrow-only; same Spanish template gave Fracture UNK in 053, 098 |
| 18 | 031 | …918767557 | Fracture | NO 0 | UNK 0 | no fracture or bone-integrity sentence | never named or denied |
| 19 | 032 | …845481617 | Medial Meniscus | YES 1 | YES 2 | "Medial menisküs posterior hornu ve gövde kesiminde **grade II** dejenerasyon" | grade 2 is not a severity-1 qualifier; the same reader scored grade II as 2 in 046 and 093 |
| 20 | 034 | …985753636 | Fracture | NO 0 | UNK 0 | "Alteración de señal de la médula ósea … infarto óseo. No hay **otras alteraciones de señal de la médula ósea**" | marrow-only; fracture never named |
| 21 | 036 | …839994882 | Fracture | UNK 0 | NO 0 | "Костите, формиращи дясната колянна става, са с **нормална форма**" | structural normality; identical phrasing scored NO in 085 |
| 22 | 037 | …273260339 | Lateral Meniscus | UNK 0 | YES 2 | "**Medial ve lateral** menisküste grade II dejenerasyon izlendi" | the lateral meniscus is explicitly named in the same sentence used for the medial |
| 23 | 037 | …273260339 | Baker's | NO 0 | UNK 0 | only "posterior komşuluğunda … **parameniskal kist**" | a parameniscal cyst is not a Baker's; Baker's never named |
| 24 | 037 | …273260339 | Fracture | NO 0 | UNK 0 | "kemik yapıların **sinyal intensiteleri** doğaldır" | signal-only (see #5) |
| 25 | 038 | …948754343 | Contusion | UNK 0 | NO 0 | "kemik yapıların **bütünlüğü, sinyal intensiteleri** doğaldır" | explicit normal marrow signal |
| 26 | 038 | …948754343 | Fracture | UNK 0 | NO 0 | same sentence — "**bütünlüğü**" (integrity) | integrity explicitly stated normal |
| 27 | 039 | …768015985 | Contusion | UNK 0 | NO 0 | "Ostale prikazane **koštane** … strukture uredne su morfologije i **intenziteta signala**" | same span as #15, same conclusion |
| 28 | 042 | …956171166 | Fracture | NO 0 | UNK 0 | no fracture or bone-integrity sentence (only subchondral edema/cysts) | never named or denied |
| 29 | 048 | …591573824 | Synovitis | YES 1 | UNK 0 | the only related span is "Mild **popliteus bursitis**" | bursitis is a different structure; synovitis never named — inference from a related finding |
| 30 | 050 | …005084725 | Contusion | NO 0 | UNK 0 | only "subkondral medüller **fibrokistik ve fibrovasküler değişiklikler**" | marrow never declared normal; degenerative marrow change is not a denial |
| 31 | 051 | …208780221 | Fracture | UNK 0 | NO 0 | "**Normal visualized bones** and surrounding soft tissues" | identical English sentence scored Fracture NO in 024, 059, 062 |
| 32 | 055 | …422536643 | Medial OA | NO 0 | UNK 0 | the medial paragraph reads only "complex tear in a buckethandle type at body and posterior horn…"; the "no focal chondrosis or chondral injury" clause appears in the **lateral** and patellofemoral paragraphs only | the negation sentence is absent from the medial compartment |
| 33 | 057 | …842864484 | Lateral Meniscus | NO 0 | YES 1 | "Increased signal at the posterior root attachment … likely representing **intrasubstance degeneration**"; impression "**Lateral meniscus degeneration** without definite tear" | degeneration-without-tear is the reader's own YES 1 (059, 062, 088, 090) |
| 34 | 066 | …373997197 | Fracture | NO 0 | UNK 0 | no fracture or bone-integrity sentence (only "wat botoedeem", "kleine subchondrale geode's") | never named or denied |
| 35 | 070 | …(study 070) | Fracture | NO 0 | UNK 0 | "**Marrow signal is otherwise normal**"; "No evidence of cortical collapse" | marrow-only; fracture never named (collapse ≠ fracture) |
| 36 | 075 | …(study 075) | Fracture | YES 1 | YES 2 | "Fracture d'avulsion **non déplacée** de l'épine tibiale", "Fracture non déplacée du coin postérieur du plateau tibial latéral **avec extension intra-articulaire**", "Fracture intra-articulaire non déplacée de la tête du péroné" | "non déplacée" is not a minimal/mild qualifier; three fractures stated without a severity-1 word |
| 37 | 080 | …812808052375 | Medial Meniscus | NO 0 | YES 1 | "**Cambios en la intensidad de la señal en el menisco interno sin desgarro**" | signal change without tear = degeneration; the reader's own YES 1 convention (and the near-identical Spanish wording in 090 got YES 1) |
| 38 | 081 | …307405130821 | Lateral OA | UNK 0 | YES 1 | "Manji hiperintenziteti **femoralne i tibijalne** hrskavice u smislu hodromalacije I stupnja" | compartment-unspecified femorotibial statement used for Medial OA YES 1 but not for Lateral |
| 39 | 082 | …332647762115 | Baker's | NO 0 | UNK 0 | only "**Fluid along popliteus tendon**" | Baker's never named; peritendinous fluid is not a popliteal cyst |
| 40 | 082 | …332647762115 | Fracture | YES 1 | UNK 0 | "**Suspicious for** bone fracture along lateral patellar" (repeated verbatim in the impression) | explicitly uncertain ⇒ UNK by the rulebook |
| 41 | 084 | …832488098189 | PF OA | NO 0 | UNK 0 | anterior compartment says only "ικανοποιητική **επαλληλία** της επιγονατίδας και της τροχιλίας η οποία … ελέγχεται **συγγενώς αβαθής**" | congruence/trochlear depth is morphology; patellofemoral cartilage never assessed |
| 42 | 085 | …458808329326 | Medial OA | UNK 0 | NO 0 | "**Артикулиращият хрущял на фемура и тибията е съхранен**" | femorotibial cartilage explicitly preserved |
| 43 | 085 | …458808329326 | Lateral OA | UNK 0 | NO 0 | same sentence | same |
| 44 | 085 | …458808329326 | Contusion | NO 0 | UNK 0 | the only bone statement is "Костите … са с **нормална форма**"; plus "малки субхондрални кисти" | shape/structure is not marrow signal; marrow never assessed |
| 45 | 097 | …515805908561 | Lateral OA | NO 0 | UNK 0 | "…irregular thining of intra-articular cartilage over **lateral patellofemoral joint and medial femorotibial joint**" | the lateral *femorotibial* compartment is never addressed |
| 46 | 097 | …515805908561 | Effusion | YES 1 | YES 2 | "**Joint effusion** and suprapatellar bursitis" (no qualifier; "Mild" attaches to synovitis) | stated without qualification ⇒ 2 |
| 47 | 097 | …515805908561 | Fracture | NO 0 | UNK 0 | fracture never named; "in favor of small **osteonecrosis**" | osteonecrosis is not a fracture denial |
| 48 | 098 | …556008991334542 | Medial Meniscus | NO 0 | YES 1 | "**Aumento de señal central del cuerno posterior del menisco medial** que no contacta la superficie articular, sin caracteres propios de rotura" | the same Spanish template sentence was scored YES 1 in 002 and 090 |

## Uncertain (37 cells) — flagged, not contested

| Study | Finding | Current | Doubt | Evidence |
|-------|---------|---------|-------|----------|
| 004 | Contusion | YES 1 | reactive vs traumatic marrow edema | "œdème osseux **réactionnel** [grade 4]" with "Signal de la moelle osseuse : Par ailleurs normal." |
| 006 | ACL | YES 1 | severity | "likely due to **sprain**" / "ACL sprain" — no qualifier ⇒ arguably 2 |
| 009 | Lateral OA | UNK 0 | whole-joint OA statement | "Wearing of articular cartilage and spur formation"; "**Osteoarthritis of left knee**" — used for Medial YES 2 only |
| 009 | PF OA | UNK 0 | same | same span |
| 015 | PF OA | YES 1 | osteochondral ⇒ OA | "Η βλάβη έχει αποσπαστεί από την έσω µοίρα … της **επιγονατίδας**" |
| 017 | ACL | YES 2 | severity | "**Uznapredovala** mukoidna degeneracija … bez znakova rupture" (advanced ⇒ 3?) |
| 018 | Medial OA | UNK 0 | osteochondral ⇒ OA | "Small **osteochondral lesion** at posterior portion of medial femoral condyle is **suspected**" |
| 019 | PF OA | YES 2 | severity | "Diffuse low grade and **high grade** chondrosis (… more than 50% thickness cartilage loss)" — scored 3 in 088 |
| 021 | Effusion | YES 2 | severity | "**ικανής έκτασης** ενδαρθρική συλλογή υγρού" (considerable extent ⇒ 3?) |
| 022 | Lateral OA | UNK 0 | osteochondral ⇒ OA | "chronic **osteochondral injury** at the posterolateral tibial plateau (2.3×1.3 cm)" |
| 022 | Contusion | YES 1 | severity | "There is **subchondral edema**" — unqualified ⇒ 2? |
| 026 | Fracture | UNK 0 | structural normality | "Ostale prikazane koštane … strukture uredne su **morfologije**" — "ostale" leaves scope open |
| 031 | ACL | YES 2 | severity | "**blago** uzmjenjenog intenziteta … **manja** parcijalna ruptura" ⇒ 1? |
| 033 | Medial OA | NO 0 | scope of the negation | "No focal **full-thickness** cartilage defects" denies only full-thickness loss |
| 033 | Lateral OA | NO 0 | same | same |
| 033 | PF OA | NO 0 | same | same |
| 037 | Medial Meniscus | YES 1 | severity | "grade II dejenerasyon" + "**minimal** horizontal yırtık" pull opposite ways |
| 038 | ACL | YES 1 | severity | "dejenerasyon ile uyumlu sinyal artışı" — unqualified ⇒ 2? |
| 039 | Fracture | UNK 0 | structural normality | same Croatian sentence as 026 |
| 042 | Baker's | YES 1 | severity | "Poplitealna cista veličine **20×6×5 mm**" — size given, no qualifier |
| 044 | PF OA | YES 2 | osteochondral ⇒ OA | "**Lesión osteocondral de la rótula**" |
| 045 | Effusion | YES 1 | severity | "**Leve a moderado** derrame articular" |
| 046 | ACL | YES 1 | severity | "miksoid dejenerasyona sekonder sinyal değişikliği" — unqualified |
| 049 | Lateral OA | YES 1 | severity | "her üç kompartmanda santral-marjinal **osteofit formasyonu**" — unqualified |
| 049 | PF OA | YES 1 | severity | same span |
| 052 | MCL | YES 1 | severity | "**Distenzija** medijalnog kolateralnog ligamenta" — unqualified |
| 052 | Lateral OA | UNK 0 | osteochondral ⇒ OA | "areal koštanog edema uz impresiju kortikalne kosti - **osteohondralna IIa lezija**" |
| 061 | Lateral OA | YES 2 | osteochondral ⇒ OA | "**Lesión osteocondral** en la meseta tibial lateral" |
| 066 | Medial OA | YES 1 | severity | "**beginnende** kraakbeenlijden **graad 2**" |
| 081 | MCL | YES 3 | severity | "**Veća** parcijalna ruptura medijalnog kolateralnog ligamenta" (larger, but partial) |
| 081 | Contusion | YES 1 | severity | lead sentence "Koštani edem lateralnog kondila femura" is unqualified; "Manji areali" applies to the others |
| 082 | Synovitis | YES 1 | severity | "**Synovial thickening** at anterior part of intercondylar region" — unqualified ⇒ 2? |
| 084 | Contusion | YES 2 | severity | "εστίες οστεοµυελικού οιδήµατος … σε **µεγάλη έκταση**" ⇒ 3? |
| 095 | MCL | YES 1 | severity | "Medial kollateral ligaman çevresinde **sıvı artışı**" — unqualified |
| 095 | Effusion | NO 0 | internal contradiction | "Medial ve lateral reseslerde **hafif sıvı artışı**" vs "**Eklemde sıvı artışı saptanmamıştır**" |
| 097 | Contusion | YES 1 | severity / reactive | "**Subchondral bone marrow edema/cysts** … reflecting osteoarthritis" |
| 099 | Baker's | YES 2 | is it a Baker's? | "**Quiste sinovial posterior**" — posterior synovial cyst, not named popliteal/Baker |

## Counts

- Cells examined: **1200** (100 studies × 12 findings)
- Contested: **48** (4.0%)
- Uncertain: **37** (3.1%)
- Agreed without reservation: **1115** (92.9%)

### By error type

| Type | n |
|------|---|
| NO should be UNK | 23 |
| UNK should be NO | 13 |
| UNK should be YES | 3 |
| NO should be YES | 3 |
| YES should be UNK | 2 |
| YES should be NO | 1 |
| severity wrong only | 3 |
| **total** | **48** |

### By finding

| Finding | contested | of 100 |
|---------|-----------|--------|
| Fracture | 19 | 19% |
| Contusion | 9 | 9% |
| Lateral OA | 6 | 6% |
| Medial OA | 3 | 3% |
| Medial Meniscus | 3 | 3% |
| Baker's | 3 | 3% |
| Lateral Meniscus | 2 | 2% |
| PF OA | 1 | 1% |
| Synovitis | 1 | 1% |
| Effusion | 1 | 1% |
| ACL | 0 | 0% |
| MCL | 0 | 0% |

**Fracture + Contusion = 28 of 48 contested cells (58%)** while being 2 of 12 findings (17% of cells).

### By report language

| Language | studies | cells | contested | rate |
|----------|---------|-------|-----------|------|
| Bulgarian | 4 | 48 | 6 | 12.5% |
| Turkish | 13 | 156 | 12 | 7.7% |
| Croatian | 8 | 96 | 6 | 6.3% |
| Spanish | 21 | 252 | 7 | 2.8% |
| Greek | 6 | 72 | 2 | 2.8% |
| Dutch | 3 | 36 | 1 | 2.8% |
| French | 3 | 36 | 1 | 2.8% |
| English | 40 | 480 | 13 | 2.7% |
| German | 2 | 24 | 0 | 0% |

The non-English rate (35/720 = 4.9%) is nearly double the English rate (13/480 = 2.7%), and the
excess is concentrated in the three languages whose all-normal boilerplate sentences bundle bone and
cartilage into one clause (Bulgarian, Turkish, Croatian).

## Systematic failure modes

1. **The bone axis is not governed by a stable rule (28/48 cells).** The reader has no fixed answer
   to "what does a normal-bone sentence deny?" The same span produces `NO`, `UNK`, or nothing at all
   depending on the study. Three sub-patterns:
   - *Signal-only read as structural.* Turkish "kemik yapıların **sinyal intensiteleri** doğaldır"
     ⇒ Fracture `NO` in 008 and 037, but `UNK` in 032 and 046 — same sentence, same template.
   - *Structural read as signal.* Bulgarian "Костите … с нормална форма / непроменена структура"
     ⇒ Fracture `NO` in 054 and 085 but `UNK` in 007 and 036, while in 085 it was also (wrongly)
     used to deny Contusion.
   - *A whole clause dropped.* Turkish "Eklem kıkırdakları **ve kemikler** … normal" answered four
     cells in 030/067/086 and none in 025.
2. **A finding the report never names is answered `NO` (23 cells).** Overwhelmingly Fracture,
   Contusion, and Baker's — 000, 010, 012, 014, 015, 029, 031, 037, 042, 050, 066, 070, 082, 084,
   085, 097. This is the single largest type and it is a one-way bias: the reader almost never
   invents a `YES`, it invents a *denial*. Cost: `NO` and `UNK` are different training targets, and
   a fabricated `NO` on Fracture is a confident negative on the rarest positive class in the set.
3. **Bilateral or compartment-unspecified statements answered for one side only (6 cells).**
   037 ("Medial **ve lateral** menisküste grade II" → medial only), 017 ("ostale zglobne hrskavice"
   → PF but not Lateral), 081 ("femoralne i tibijalne hrskavice" → Medial but not Lateral), 007
   ("по-изразени медиално" read as a lateral denial), 085 ("хрущял на фемура и тибията е съхранен"
   → neither side), 097 ("lateral patellofemoral **and** medial femorotibial" → lateral femorotibial
   scored NO). The rulebook's "medial and lateral are distinct" is being applied as "only the side I
   noticed first is answered".
4. **Degeneration-without-tear is scored `YES 1` in ten studies and `NO` in two (057, 080).** The
   convention itself is defensible; its application is not. 098 is the sharpest case — a Spanish
   template sentence scored `YES 1` in 002 and 090 and `NO` here.
5. **Grade→severity drift.** "grade II" → 1 in 032 but 2 in 046 and 093; "high grade
   partial-thickness" → 2 in 019 but 3 in 088. The rulebook maps grade 1→1 and grade 3-4→3 with
   grade 2 falling into the unqualified bucket (2); the reader treats grade II as "mild" about half
   the time. Only 3 cells were contested on severity alone, but 20 of the 37 uncertain cells are
   severity questions — nearly all of them "the report gave no qualifier and the reader chose 1".
   The reader's systematic lean is to *under*-score unqualified findings as 1 rather than 2.
6. **Inference from a related finding (2 cells).** 048 popliteus bursitis ⇒ Synovitis `YES`;
   029 secondary degenerative bone change ⇒ Contusion `YES` against an explicit marrow-normal
   sentence. Rare, but both are the rulebook's "do not infer a finding from a related one".
7. **Osteochondral lesion ⇒ compartment OA is genuinely unresolved, not an error.** `UNK` in 018,
   022, 052 and `YES` in 015, 044, 054, 061. Nine cells are listed as uncertain rather than
   contested because the rulebook does not say which reading is intended — but a labelling
   convention should be chosen and applied, since it is worth ~1% of all cells on its own.
8. **A negation sentence missing from one paragraph of an otherwise templated report is not
   noticed (055).** The medial compartment paragraph lacked the "no focal chondrosis" clause the
   other two carried, and the `NO` was carried over anyway.
