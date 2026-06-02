# PP MI asistents – darba prasmes

Dash tīmekļa lietotne semantiskai prasmju meklēšanai Latvijas Bankā, izmantojot Azure OpenAI vektorizāciju un GPT-4.1.

## Ko dara

1. **Semantiskā meklēšana** — vektorizē lietotāja brīvā teksta darba aprakstu un atrod tuvākās prasmju rindas FAISS vektorindeksā.
2. **MI ranžēšana** — nosūta labākos kandidātus GPT-4.1, kas atlasa un sakārto atbilstošākās unikālās prasmes atbilstoši lietotāja vaicājumam.
3. **Eksports** — rezultātus var lejupielādēt kā `.xlsx` failu.

## Datu avoti

| Atslēga | Apraksts |
|---|---|
| `ESCO_en` | Eiropas prasmju, kompetenču, kvalifikāciju un profesiju taksonomija |
| `SkillsFuture` | Singapūras mūžizglītības prasmju taksonomija |

Abi indeksi ir iepriekš sagatavoti FAISS vektorindeksi, kas glabājas tīkla diskā (`//ezers/...`).

## Prasības

- Python 3.10+
- Piekļuve Azure OpenAI (nepieciešami divi izvietojumi — sk. vides mainīgos zemāk)
- Tīkla piekļuve FAISS indeksu koplietojumam

Atkarību instalēšana:

```bash
pip install -r requirements.txt
```

## Vides mainīgie

| Mainīgais | Lietojums |
|---|---|
| `AZURE_EMBEDDING_OPENAI_API_KEY` | `text-embedding-3-large` vektorizācijas modelis |
| `AZURE_OPENAI_API_KEY` | `gpt-4.1` čata modelis |

## Palaišana

```bash
python app.py
```

Lietotne startē uz `http://127.0.0.1:8050` pēc noklusējuma.

## Lietotāja saskarne

- **Valodas pārslēgšana** — galvenē pārslēdzies starp latviešu (LV) un angļu (EN) valodu.
- **Datu avots** — izvēlies ESCO vai SkillsFuture ar radio pogām.
- **Vaicājuma ievade** — ielīmē darba aprakstu, pienākumus vai brīva teksta prasmju profilu.
- **Prasmju skaits** — norādi, cik unikālas prasmes atgriezt (1–100).
- **Lejupielāde** — eksportē rezultātus uz `.xlsx`, tiklīdz MI asistents ir sniedzis atbildi.
