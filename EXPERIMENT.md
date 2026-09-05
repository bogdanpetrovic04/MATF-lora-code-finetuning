# Eksperiment
## Model i podaci

Bazni model je `Qwen/Qwen2.5-Coder-1.5B`, revizija
`df3ce67c0e24480f20468b6ef2894622d69eb73b`. Poredjeni su bazni model (B0),
custom LoRA (C1) i PEFT LoRA (P1).

Iz `nvidia/OpenCodeInstruct` izdvojeno je 20.000 trening i 2.000 validacionih
primera. Pilot od 1.000 primera je deo trening skupa (provera da li kod radi pre pokretanja punog treninga). Primenjeni su filteri
kvaliteta i duzine, deduplikacija normalizovanih promptova i provera
preklapanja sa benchmark promptovima i imenima funkcija, ukljucujuci nizove
od osam reci. Stratifikacija je radjena prema domenu, nacinu generisanja
i kvartilu duzine, seed je 42.

## Trening

- LoRA: rank 16, alpha 32, dropout 0,05, `q_proj` i `v_proj` projekcije.
- Jedna epoha; finalni checkpoint posle 625 optimizer koraka.
- AdamW: learning rate 0,0002, betas (0,9; 0,999), epsilon 1e-8, weight decay 0.
- Cosine raspored, 3% warmup, gradient clipping na normu 1.
- Micro-batch 8, akumulacija 4; efektivni batch 32.
- Najvise 1.024 tokena, completion-only loss.
- BF16 autocast, FP32 parametri, gradient checkpointing (manje GPU memorije, ali sporiji trening).
- Isti redosled podataka i podesavanja za C1 i P1; seed 42.
- Windows, RX 7900 XTX 24 GiB, ROCm 7.14.0, PyTorch 2.12.0.

## Evaluacija

Za svaki zadatak generisan je jedan temp=0 odgovor sa najvise 512 novih
tokena. B0, C1 i P1 koriste isti prompt, izdvajanje koda i evaluator.
Konacni checkpoint nije biran prema benchmark rezultatu.

EvalPlus 0.3.1, commit `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`:
HumanEval+ v0.1.10 (164 zadatka) i MBPP+ v0.2.0 (378 zadataka).
Generisani kod izvrsen je u Docker-u bez mreze. Primarne metrike su
HumanEval+ i MBPP+ pass@1; originalni testovi i validacioni loss su sekundarni.

Intervali su percentile bootstrap preko zadataka, 10.000 uzoraka, seed 42.
Razlike su prikazane kroz dobitke, gubitke i uparene bootstrap intervale.
Uradjen je po jedan trening C1 i P1.


## Dobijeni fajlovi

Finalni adapteri iz koraka 625 spakovani su u `artifacts/custom_adapter.zip`
i `artifacts/peft_adapter.zip`. Raspakovane kopije su u `adapters/custom/`
i `adapters/peft/`. Trening i validacioni podaci korisceni u eksperimentu
nalaze se u `data/train_20k.parquet` i `data/validation_2k.parquet`.
