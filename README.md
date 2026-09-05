# LoRA fine-tuning modela za generisanje koda

Poredjenje baznog Qwen2.5-Coder-1.5B modela, sopstvene PyTorch LoRA
implementacije i Hugging Face PEFT adaptera. Oba adaptera trenirana su jednu
epohu na istih 20.000 OpenCodeInstruct primera.

Autori: **Stefan Shrestha, Bogdan Petrovic**

## Rezultati

| Benchmark | Base | Custom LoRA | PEFT LoRA |
|---|---:|---:|---:|
| HumanEval | 67,07% | 77,44% | 77,44% |
| HumanEval+ | 60,37% | 70,73% | 70,12% |
| MBPP | 71,69% | 71,96% | 70,63% |
| MBPP+ | 61,11% | 62,17% | 61,64% |

Custom LoRA je na HumanEval+ benchmarku resila 116 od 164 zadatka, dok je bazni model resio 99. U direktnom poredjenju po zadacima, custom model je uspesno resio 27 zadataka na kojima je bazni model pogresio, ali je pogresio na 10 zadataka koje je bazni model resio. Na MBPP+ benchmarku razlike izmedju baznog, custom LoRA i PEFT modela bile su male.

Metrike su u [results/](results/), a postupak je opisan u
[EXPERIMENT.md](EXPERIMENT.md).

## Notebookovi

1. [Podaci](notebooks/01_analiza_podataka.ipynb) - filtriranje, podela, chat format i maskiranje tokena.
2. [LoRA i trening](notebooks/02_lora_i_trening.ipynb) - forward, custom/PEFT, akumulacija gradijenata, adapteri i loss krive.
3. [Evaluacija i demo](notebooks/03_rezultati_i_demo.ipynb) - izdvajanje koda, benchmark rezultati i opciono generisanje.

Notebookovi koriste JSON iz repozitorijuma i rade bez modelskih tezina
i Parquet fajlova. GPU je potreban samo za opcioni demo u trecem notebooku,
koji se ukljucuje postavljanjem `RUN_LIVE_DEMO = True`.
Prvi notebook koristi Qwen tokenizer iz lokalnog cache-a; ako nije dostupan,
preuzima samo tokenizer sa Hugging Face-a. Drugi koristi i mali Qwen sa
nasumicnim tezinama za prikaz pripreme custom i PEFT adaptera.

## Instalacija na Windows-u

Preporucen je Python 3.12. Iz korena repozitorijuma napraviti virtuelno
okruzenje i instalirati projekat:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,notebooks]"
.\.venv\Scripts\python.exe -m pytest -q
```

U VS Code-u kao Python interpreter i Jupyter kernel izabrati:

```text
.venv\Scripts\python.exe
```

Osnovni delovi notebookova mogu da se pokrenu bez GPU-a. Puni trening zahteva
odgovarajuce GPU okruzenje. Eksperiment je izveden na AMD Radeon RX 7900 XTX
uz ROCm 7.14, PyTorch 2.12 i Python 3.12. Na drugom racunaru instalaciju
PyTorch-a treba prilagoditi dostupnom hardveru.

## Kod i podaci

- [lora.py](src/code_lora/lora.py): LoRA sloj, ubacivanje adaptera i merge/unmerge.
- [data.py](src/code_lora/data.py): Parquet, chat format, maskiranje i padding.
- [training.py](src/code_lora/training.py): trening i validacioni loss.
- [serialization.py](src/code_lora/serialization.py): cuvanje i ucitavanje custom adaptera.
- [evaluation.py](src/code_lora/evaluation.py): ucitavanje modela i generisanje resenja.
- [tests/test_lora.py](tests/test_lora.py): pet provera LoRA implementacije.

Izvor je `nvidia/OpenCodeInstruct`. Filteri zadrzavaju test score >= 0,8,
prosecnu LLM ocenu >= 4 i najvise 1.024 formatirana tokena. Podaci su
deduplikovani i provereni prema benchmark promptovima pre podele na 20.000
trening i 2.000 validacionih primera.

Adapteri i podaci preuzimaju se kao prilozi uz GitHub Release:

- Custom adapter: [custom_adapter.zip](https://github.com/bogdanpetrovic04/MATF-lora-code-finetuning/releases/latest/download/custom_adapter.zip)
- PEFT adapter: [peft_adapter.zip](https://github.com/bogdanpetrovic04/MATF-lora-code-finetuning/releases/latest/download/peft_adapter.zip)
- Trening skup (20.000 primera): [train_20k.parquet](https://github.com/bogdanpetrovic04/MATF-lora-code-finetuning/releases/latest/download/train_20k.parquet)
- Validacioni skup (2.000 primera): [validation_2k.parquet](https://github.com/bogdanpetrovic04/MATF-lora-code-finetuning/releases/latest/download/validation_2k.parquet)

ZIP fajlove staviti u `artifacts/`, a Parquet fajlove u `data/`.
Parquet fajlovi sadrze tacne splitove koriscene u eksperimentu, na putanjama
iz [konfiguracije](configs/experiment.yaml).

ZIP arhive sadrze finalne adaptere iz koraka 625: konfiguraciju i tezine.
Raspakuju se iz korena repozitorijuma sledecim komandama:

```powershell
Expand-Archive artifacts/custom_adapter.zip -DestinationPath adapters
Expand-Archive artifacts/peft_adapter.zip -DestinationPath adapters
```

Time nastaju `adapters/custom/` i `adapters/peft/`, koje koriste validacija
i demo.

### Objavljivanje na GitHub-u

U Release dodati `custom_adapter.zip`, `peft_adapter.zip`, `train_20k.parquet`
i `validation_2k.parquet`, bez promene naziva. Gornji linkovi vode do priloga
najnovijeg Release-a. ZIP-ovi, Parquet fajlovi i raspakovani adapteri su u
`.gitignore`, pa se ne dodaju u istoriju izvornog koda.

## Komande

Iz korena repozitorijuma, u aktiviranom projektnom Python okruzenju:

```powershell
python scripts/train.py --backend custom --output-dir outputs/custom
python scripts/train.py --backend peft --output-dir outputs/peft
python scripts/validate.py --variant base
python scripts/validate.py --variant custom --adapter adapters/custom
python scripts/validate.py --variant peft --adapter adapters/peft
```

Konfiguracija koristi `cuda:1`: to je RX 7900 XTX na projektnoj masini.
Na racunaru sa jednim GPU-om podesiti `cuda:0`. Trening je koristio oko
23,88 GiB GPU memorije.

Generisanje i pokretanje benchmark testova:

```powershell
docker build -f Dockerfile.evalplus -t code-lora-evalplus:0.3.1 .
python scripts/generate_evalplus.py --benchmark humaneval --variant base --output samples/base.jsonl
powershell -File scripts/run_evalplus.ps1 -Benchmark humaneval -Samples samples/base.jsonl
```

Generisani kod izvrsava se u Docker-u bez mreze, sa ogranicenim resursima.
