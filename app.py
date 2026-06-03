import os
import json
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import dash
from dash import dcc, html, Output, Input, State, dash_table
import dash.exceptions as de
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

# ───────────────────────────── Helpers ─────────────────────────────
def l2_to_score(dist: float) -> float:
    """Convert L2 distance to [0,1] similarity score."""
    return 1.0 / (1.0 + dist)


def _doc_to_content_meta(doc: Any) -> Tuple[str, Dict[str, Any]]:
    """Normalize FAISS-returned doc into (content, metadata)."""
    if isinstance(doc, Document):
        return doc.page_content, (doc.metadata or {})
    return str(doc), {}


def _build_row(rank: int, raw_dist: float, doc: Any, source_name: str, index_key: str = "") -> Dict[str, Any]:
    """Create a result row dict and also parse Loma/Prasme/Prasmes apraksts."""
    sim_score = l2_to_score(raw_dist)
    content, meta = _doc_to_content_meta(doc)

    row = {
        "Secība": rank,
        "Semantiskā līdzība": round(sim_score, 4),
        "Saturs": content,
        "Datu avots": source_name,
        **meta,
    }

    if index_key == "VAS_kompetences":
        row["Joma"] = meta.get("Joma", "")
        row["Tips"] = meta.get("Tips", "")
        row["Kompetence"] = meta.get("Kompetence", "")
        row["Definīcija"] = meta.get("Definīcija", "")
        row["Kompetences līmenis"] = meta.get("Kompetences līmenis", "")
        row["Rīcības rādītājs"] = meta.get("Rīcības rādītājs", "")
    else:
        parts = content.split(" - ", 2) + ["", "", ""]
        row["Loma"], row["Prasme"], row["Prasmes apraksts"] = parts[:3]
    return row


def topk_unique_prasme(query: str, target_unique: int, vector_store, source_name: str, index_key: str = "") -> List[Dict[str, Any]]:
    """
    Return the first N unique results.
    For ESCO/SkillsFuture: dedup by 'Prasme', collect 'Citas lomas'.
    For VAS: dedup by 'Rīcības rādītājs'.
    """
    query = (query or "").strip()
    target_unique = max(min(int(target_unique or 10), 1000), 1)
    is_vas = (index_key == "VAS_kompetences")

    unique_rows: List[Dict[str, Any]] = []
    seen_map: Dict[str, Dict[str, Any]] = {}

    k_current = target_unique
    k_step = max(50, target_unique)
    k_max = 5000
    fetched_len_prev = 0

    while len(seen_map) < target_unique and k_current <= k_max:
        docs_and_scores = vector_store.similarity_search_with_score(query, k=k_current)

        if len(docs_and_scores) == fetched_len_prev and k_current > fetched_len_prev:
            break

        for rank, (doc, raw_dist) in enumerate(docs_and_scores[fetched_len_prev:], start=fetched_len_prev + 1):
            row = _build_row(rank, raw_dist, doc, source_name, index_key)

            if is_vas:
                dedup_key = (row.get("Rīcības rādītājs") or "").strip()
            else:
                dedup_key = (row.get("Prasme") or "").strip()

            if not dedup_key:
                continue

            if dedup_key not in seen_map:
                idx = len(unique_rows)
                if not is_vas:
                    row["Citas lomas"] = ""
                unique_rows.append(row)
                if is_vas:
                    seen_map[dedup_key] = {"idx": idx}
                else:
                    loma_val = (row.get("Loma") or "").strip()
                    seen_map[dedup_key] = {
                        "idx": idx,
                        "primary_loma": loma_val,
                        "extra_lomas": [],
                    }
            else:
                if not is_vas:
                    info = seen_map[dedup_key]
                    loma_val = (row.get("Loma") or "").strip()
                    if loma_val and loma_val != info["primary_loma"] and loma_val not in info["extra_lomas"]:
                        info["extra_lomas"].append(loma_val)

            if len(seen_map) >= target_unique:
                break

        fetched_len_prev = len(docs_and_scores)
        if len(seen_map) >= target_unique:
            break

        k_current = min(k_current + k_step, k_max)

        if len(docs_and_scores) < fetched_len_prev:
            break

    if not is_vas:
        for p, info in seen_map.items():
            idx = info["idx"]
            extras = [e for e in info["extra_lomas"] if e]
            unique_rows[idx]["Citas lomas"] = ", ".join(extras) if extras else ""

    return unique_rows


def _extract_system_only(text: str) -> str:
    """Return only the system prompt part from the 2nd textarea (strip any appended preview)."""
    if not text:
        return DEFAULT_SYSTEM_PROMPT.strip()
    markers = [
        "\n# Lietotāja vaicājums",   # run-time marker
        "\nLietotāja meklējums:",   # live-preview marker
        "\n## Ievadītie dati:",
        "\n### Piedāvātās rindas:",
    ]
    cut_positions = [text.find(m) for m in markers if m in text]
    if cut_positions:
        return text[:min(cut_positions)].rstrip()
    return text.strip()


# ─────────────────────────── Embeddings / LLM ──────────────────────
embeddings = AzureOpenAIEmbeddings(
    deployment="text-embedding-3-large",
    model="text-embedding-3-large",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint="https://pp-mi-darba-prasmes.cognitiveservices.azure.com/",
    openai_api_version="2024-12-01-preview",
)

chat_llm = AzureChatOpenAI(
    deployment_name="gpt-5.4",
    model_name="gpt-5.4",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint="https://pp-mi-darba-prasmes.cognitiveservices.azure.com/",
    openai_api_version="2025-04-01-preview",
    temperature=0,
    reasoning_effort="none",
    seed=42,
)

class SkillRow(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    Loma: str = Field(..., description="Job role / position name")
    Prasme: str = Field(..., description="Skill name")
    Prasmes_apraksts: str = Field(..., alias="Prasmes apraksts", description="Skill description")
    Seciba: int = Field(..., alias="Secība", description="Ordering index")
    Citas_lomas: Optional[str] = Field(None, alias="Citas lomas", description="Other roles where skill applies")

class SkillRows(BaseModel):
    rows: List[SkillRow]


class VASRow(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    Joma: str = Field(..., description="Domain / field")
    Tips: str = Field(..., description="Type (functional, generic, etc.)")
    Kompetence: str = Field(..., description="Competence name")
    Definicija: str = Field(..., alias="Definīcija", description="Competence definition")
    Kompetences_limenis: str = Field(..., alias="Kompetences līmenis", description="Competence level")
    Ricibu_raditajs: str = Field(..., alias="Rīcības rādītājs", description="Behavioral indicator")
    Seciba: int = Field(..., alias="Secība", description="Ordering index")

class VASRows(BaseModel):
    rows: List[VASRow]


chat_llm_skills = chat_llm.with_structured_output(SkillRows)
chat_llm_vas = chat_llm.with_structured_output(VASRows)


VAS_SYSTEM_PROMPT = (
    """ # Sistēmas vaicājums

    Tu esi noderīgs, strukturēts kompetenču asistents - pārzini VAS kompetenču bibliotēku un atlasa tuvākos unikālos kompetenču aprakstus, balstoties uz lietotāja vaicājumu.
    Tu strādā Latvijas Bankas labā.

    ### Instructions - seko instrukciju secībai:
    1. Balsties TIKAI uz {{piedāvātās rindas}};
    2. Atlasi unikālos "rīcības rādītājus" no {{piedāvātās rindas}}, kuri ir vistuvāk {{lietotāja vaicājums}}
    3. Sakārto secībā atlikušos ierakstus un atlasi izvēlēto skaitu atbilstoši {{gala rezultātu atlase}}.
    4. Iegūstot sarakstu ar unikālajiem ierakstiem, no rindām iegūsti arī atbilstošās pārējās kolonnas (Joma, Tips, Kompetence, Definīcija, Kompetences līmenis, Rīcības rādītājs).
    5. Netulko ievietotās rindas, paturi oriģinālvalodu.
    """
)


DEFAULT_SYSTEM_PROMPT = (
    """ # Sistēmas vaicājums
    
    Tu esi noderīgs, strukturēts darba prasmju asistents - pārzini ESCO un SkillsFuture frameworks un atlasa tuvākos unikālos prasmju aprakstus, balstoties uz lietotāja vaicājumu.
    Tu strādā Latvijas Bankas labā.
    
    ### Informācija no Latvijas Bankas.
    
    #### **I.** **Stratēģiskās attīstības virzieni 2024. – 2026. gadam**

            # **1. Sabiedrībai pieejams drošs un attīstīts finanšu sektors**


            **Mērķi:**


            - Latvijā izveidota inovatīva un stabila ekosistēma sabiedrībai plaši pieejamu finanšu pakalpojumu sniegšanai, jaunu produktu ieviešanai un

            finanšu tirgus dalībnieku darbībai

            - Latvijas Banka nodrošina samērīgu finanšu sektora regulējumu un efektīvu uzraudzību

            - Latvijas Banka proaktīvi līdzdarbojas pārdomātas finanšu sektora politikas veidošanā


            |Sasniedzamais rezultāts|Sasniedzamā vērtība|Aprēķina metodika|
            |---|---|---|
            |Nemazinās finanšu sektora ieguldījums<br>tautsaimniecības attīstībā|≥ 3%|Finanšu un apdrošināšanas darbību nozares pievienotā vērtība<br>procentos pret IKP, pēdējo 3 gadu vidējais. Ikgadējs aprēķins no<br>Latvijas Bankas izmantotajiem statistiskajiem datiem|
            |Finanšu sektora risku realizēšanās nav<br>radījusi būtisku ietekmi uz sektora stabilitāti|0 incidentu ar ietekmi<br>4, un ne vairāk kā 1<br>incidents ar ietekmi 3|Ikgadējs pašvērtējums, pamatojoties uz Latvijas Bankas<br>izmantotajiem statistiskajiem datiem un risku vadības kritērijiem.|
            |Starptautiskās organizācijas un<br>starptautiskie sadarbības partneri pozitīvi<br>novērtē Latvijas Banku.|Sasniegts|Ikgadējs pašnovērtējums, ņemot vērā starptautisko organizāciju un<br>partneru (t.sk., IMF, OECD, FinCen, Treasury, Moneyval) veiktos<br>novērtējumus attiecībā uz Latvijas Banku|

            # **1a. Pret krīzēm noturīga Latvijas finanšu sistēma**

            **Mērķis:** Latvijas finanšu sistēma ir noturīga pret krīzes situācijām


            |Sasniedzamais rezultāts|Sasniedzamā vērtība|Aprēķina metodika|
            |---|---|---|
            |Latvijas finanšu sistēma ir noturīga pret<br>krīzēm, ņemot vērā aktuālos riskus un<br>apdraudējumus|a) Izpildīts<br> <br>b) 0 gadījumi gada<br>laikā|a) Ikgadējs izvērtējums, vai ir ieviesti plānotie risinājumi kritisko<br>finanšu pakalpojumu pieejamības nodrošināšanai<br> <br>b) Gadījumu skaits, kad kritiska incidenta dēļ konstatēts pārrāvums<br>kritisko finanšu pakalpojumu pieejamībā (kredīta pārvedumiem –|

            |Sasniedzamais rezultāts|Sasniedzamā vērtība|Aprēķina metodika|
            |---|---|---|
            |||ilgāk nekā 1 darbadiena; bezsaistes karšu maksājumiem – kritisko<br>tirgotāju skaits, kur nav iespējams veikt maksājumus – vairāk kā<br>10%; bankomātiem – ilgāk nekā 5h vienā bankomātā)|

            # **2. Tālredzīgu lēmumu katalizators valsts tautsaimniecības un indivīda līmenī**

            **Mērķi:**

            - Savas kompetences ietvaros Latvijas Banka proaktīvi iesaistās tālredzīgu, sabiedrības labklājībai būtisku lēmumu veidošanā

            - Uzlabojas iedzīvotāju un uzņēmumu finanšu pratība un zināšanas par ekonomiku

            |Sasniedzamais rezultāts|Sasniedzamā vērtība|Aprēķina metodika|
            |---|---|---|
            |Ekonomikas un finanšu ekspertu vērtējumā<br>Latvijas Bankas speciālisti ir Latvijas<br>viedokļu līderu pirmajā desmitniekā|<br> <br>> 3|Ekonomikas un finanšu ekspertu kvalitatīva aptauja, kuru veic<br>strukturētas intervijas veidā reizi divos gados, nosakot, cik Latvijas<br>Bankas speciālistu ir viedokļu līderu pirmajā desmitniekā|
            |Ekspertu (viedokļu līderu un lēmumu<br>pieņēmēju) un sabiedrības vairākums<br>uzskata, ka Latvijas Banka uzlabo Latvijas<br>iedzīvotāju finanšu pratību un ekonomisko<br>izglītotību|> 80%<br>|Ekonomikas un finanšu ekspertu kvalitatīva aptauja, kuru veic<br>strukturētas intervijas veidā reizi divos gados|
            |Ekspertu (viedokļu līderu un lēmumu<br>pieņēmēju) un sabiedrības vairākums<br>uzskata, ka Latvijas Banka uzlabo Latvijas<br>iedzīvotāju finanšu pratību un ekonomisko<br>izglītotību|> 50%|Sabiedrības vērtējums ikgadējā socioloģiskā aptaujā|
            |<br>Iedzīvotāji atbildes uz finanšu pratības<br>jautājumiem meklē Latvijas Bankas resursos|<br> <br>≥ 10%|<br>Izmantojot_Google Analytics_ datus, tiek novērtēts tīmekļvietnes<br>"Naudas skola" unikālo lietotāju skaita pieaugums gada laikā|

            # **3. Finanšu sistēmas un tautsaimniecības ilgtspējība**

            **Mērķi:**


                - Latvijas Banka virza ilgtspējības ieviešanu finanšu nozarē un tautsaimniecībā, t. sk., piedaloties ilgtspējību veicinošas regulatīvās


            vides veidošanā

                - Latvijas Banka rīkojas ilgtspējīgi ikdienas darbā, t.sk., veicinot darbinieku izpratni par ilgtspējību

            |Sasniedzamais rezultāts|Sasniedzamā vērtība|Aprēķina metodika|
            |---|---|---|
            |Izstrādāts finanšu sektora ilgtspējības jomas<br>datu apmaiņas risinājums|Izpildīts|Iekšējais izvērtējums, vai finanšu sektorā ir īstenotas aktivitātes<br>ilgtspējības jomas datu apmaiņas, apstrādes, savietojamības un<br>savienojamības nodrošināšanai|
            |Samazinās Latvijas Bankas oglekļa pēda|2024 = - 45%<br>2025 = - 60%<br>2026 = - 65%|Ikgadējais Finanšu pārvaldes veiktais_Scope 1_ un_Scope 2_ CO2<br>emisiju aprēķins (tCO2), bāzes gads = 2022. gads|
            |Aug "zaļo" iepirkumu skaits|+ 5%|Brīvprātīgo “zaļo” iepirkumu skaita pieaugums pret iepriekšējo gadu|

            # **4. Efektīva un inovatīva centrālā banka**

            **Mērķi:**


            - Latvijas Banka prasmīgi pārvalda datus, gūstot no tiem maksimālu vērtību

            Latvijas Banka kāpina efektivitāti ar tehnoloģiju palīdzību

            - Latvijas Bankā ir attīstīta inovāciju kultūra

            |Sasniedzamais rezultāts|Sasniedzamā vērtība|Aprēķina metodika|
            |---|---|---|
            |Darbinieki pozitīvi vērtē savu lietotāja<br>pieredzi bankas IT ekosistēmā|> 4 (5 ballu skalā +<br>obligāts komentārs<br>neapmierinātības<br>gadījumā)|Darbinieku ikgadēja aptauja|

            Palielinās darbinieku laika ietaupījums Reizi gadā darbinieku fokusa grupās novērtēts vidējais laika patēriņa
            Lejupejoša tendence
            standartizētu procesu nodrošināšanai ietaupījums standartizētu procesu nodrošināšanai

            # **5. Jēgpilna darbavieta profesionāļiem**


            **Mērķi:**


            - Latvijas Bankā ir iekļaujoša un elastīga darba vide, kas nodrošina darbinieku iesaisti un labbūtību

            - Latvijas Banka atbalsta darbinieku profesionālo un personīgo attīstību un talantu piesaisti

            - Latvijas Bankas darbība un tās darbinieku rīcība atbilst Latvijas Bankas vērtībām

            |Sasniedzamais rezultāts|Sasniedzamā vērtība|Aprēķina metodika|
            |---|---|---|
            |Augošs  darbinieku rekomendācijas<br>rādītājs (eNPS)|≤ 20%|Neapmierināto darbinieku (vērtējums 0-6) īpatsvara noteikšana<br>darbinieku aptaujas ietvaros (atbilstoši eNPS metodoloģijai)|
            |Vienlīdzīgs atalgojums sievietēm un<br>vīriešiem par vienādu darbu (_adjusted pay_<br>_gap_)|+/- 1% atšķirība<br>2026. gadā|Novērtējums atbilstoši pilnveidotai metodoloģijai, izmantojot<br>Personāla pārvaldes datus|


    #### **Latvijas Bankas darbinieku rīcības – vērtību diskusiju rezultāts**

        77 % darbinieku (381) 19 vērtību diskusijās (2023. gada jūlijs–novembris).


        **Drosmīgi izaicinājumos**


        Mēs esam **elastīgi pārmaiņās**, īstenojam tās apzināti un mērķtiecīgi. Apkārt notiekošais mūs

        **iedvesmo meklēt jaunus risinājumus** un **mainīties** uz labu. Mēs **iedrošinām** cits citu un arī

        apkārtējos būt **drosmīgiem**, **pieņemt izaicinājumus**, **paust savu viedokli** un **atzīt, ja**

        **kļūdāmies**, jo tikai tā varam virzīties uz priekšu.


        1. Es izrādu iniciatīvu un uzņemos jaunus, izaicinošus uzdevumus, lai veicinātu

        pozitīvas pārmaiņas.

        2. Es paužu savu viedokli pat tad, ja tas ir atšķirīgas no citu domām, veicinot

        daudzpusīgu diskusiju un ideju jaunradi.

        3. Es uzņemos riskus un esmu atvērts pārmaiņām, iesaistoties jaunās un inovatīvās

        aktivitātēs un pieņemot nezināmo.

        4. Es atzīstu savas kļūdas un mācos no tām, uzlūkojot kļūdīšanos kā iespēju augt

        un attīstīties, nevis kā šķērsli vai kaunpilnu notikumu.

        5. Esmu gatavs pieņemt sarežģītus lēmumus, aktīvi veicinu to izpildi, uzņemos

        atbildību par to rezultātiem, kā arī pielāgojos un rīkojos apņēmīgi, ja viss

        nenotiek tā, kā plānots.

        6. Es regulāri izmēģinu jaunas lietas un esmu gatavs izkāpt no komforta zonas, lai

        attīstītos gan personīgajā, gan profesionālajā jomā.


        **Vienoti cilvēcībā**


        Mēs zinām, ka spēks ir **komandā** un labākais rezultāts top **sadarbojoties** . Mūsu komandu

        vieno **cilvēcība** . Mēs **veidojam pozitīvu vidi** sev apkārt, **darot savu darbu ar prieku** . Mēs

        esam atbildīgi un **pildām savus solījumus** . Mums ir būtiska sava un kolēģu **labbūtība**, tāpēc

        esam cilvēcīgi un **rūpējamies**, lai ikviens **justos pieņemts un iekļauts** .


        1. Es izturos pret citiem ar cieņu, respektējot viņu laiku un personiskās robežas.

        2. Esmu atsaucīgs un palīdzu kolēģiem risināt jautājumus pēc būtības, izvairoties no

        birokrātijas.

        3. Es uzklausu citu viedokļus un aktīvi atbalstu viedokļu dažādību, cenšoties

        pieņemt un izprast dažādus skatpunktus bez aizspriedumiem un pārmetumiem.

        4. Man ir pozitīva attieksme, es veicinu labvēlīgu gaisotni darba vidē un esmu

        atbildīgs par to, ko un kā saku.

        5. Es iedziļinos un izturos ar sapratni, allaž gatavs atbalstīt kolēģus grūtā brīdī.

        6. Es veicinu komandas darbu, uzklausot dažādus viedokļus, idejas un vajadzības,

        kur katrs dalībnieks ir līdzvērtīgs.


        **Jaudīgi attīstībā**


        Ikviens no mums ir profesionālis ar savu unikālo pieredzi, ko visveiksmīgāk varam likt lietā

        tad, ja **dalāmies tajā ar citiem** . Mēs apzināmies – lai dotu, mums vispirms **jāapgūst pašiem** .

        Tāpēc strādājam atvērtām acīm un it visur **saskatām iespēju mācīties** . Mēs **pilnveidojamies**

        **paši un iedvesmojam to darīt citus**, lai kopā sekmētu mūsu valsts attīstību.


        1. Esmu atvērts inovācijām un jaunām darba praksēm, uzdrošinos izmēģināt tās, lai

        uzlabotu esošos procesus.

        2. Es nepārtraukti pilnveidoju savas prasmes un zināšanas, apgūstot jaunu

        informāciju un iemaņas un izmantojot tās savā ikdienas darbā, mērķtiecīgi

        plānojot laiku tam.

        3. Es aktīvi dalos savā pieredzē un zināšanās, veicinot to nodošanu kolēģiem.

        4. Es izrādu iniciatīvu un ieviešu jaunus risinājumus, kas veicina efektīvāku darba

        organizāciju un attīstību.

        5. Es mācos no saviem un kolēģu veiksmes stāstiem un neizdošanās gadījumiem,

        pielāgoju savu rīcību.

        6. Es regulāri lūdzu atgriezenisko saiti un analizēju sava darba rezultātus, turpmāk

        izmantojot gūtās atziņas sava darba snieguma uzlabošanā.


        # **Latvijas Banka vadītāju rīcības, balstoties uz jaunajam vērtībām.**

        03.03.2023 apstiprināts padomes diskusijā


        **Drosmīgi izaicinājumos**


        Mēs esam **elastīgi pārmaiņās**, īstenojam tās apzināti un mērķtiecīgi. Apkārt notiekošais mūs

        **iedvesmo meklēt jaunus risinājumus** un **mainīties** uz labu. Mēs **iedrošinām** cits citu un arī

        apkārtējos būt **drosmīgiem**, **pieņemt izaicinājumus**, **paust savu viedokli** un **atzīt, ja**

        **kļūdāmies**, jo tikai tā varam virzīties uz priekšu.


        1. Es spēju pielāgoties sarežģītām situācijām un tās izmantot kā iespēju pārmaiņām.

        2. Nepazīstamās situācijās es domāju racionāli un saglabāju pozitīvu nostāju,


        kontrolējot savas emocijas.

        3. Meklējot jaunus risinājumus, es piedāvāju inovatīvas idejas un ļauju darbiniekiem


        eksperimentēt.

        4. Es uzklausu darbinieku un kolēģu idejas ar atvērtu prātu, nemeklējot neizdošanās


        gadījumus pagātnē.

        5. Es nodrošinu nepieciešamos resursus jaunu un inovatīvu ideju radīšanai

        6. Es analizēju jaunas idejas kopā ar citiem, sniedzot konstruktīvu atgriezenisko saiti.

        7. Es lūdzu kolēģiem un darbiniekiem atgriezenisko saiti un to uzņemu racionāli un


        pozitīvi.

        8. Es atklāti atzīstu savas kļūdas; nemeklēju vainīgos.

        9. Es aizstāvu darbinieku idejas un uzņemos līdzatbildību par tām. 


        **Vienoti cilvēcībā**


        Mēs zinām, ka spēks ir **komandā** un labākais rezultāts top **sadarbojoties** . Mūsu komandu

        vieno **cilvēcība** . Mēs **veidojam pozitīvu vidi** sev apkārt, **darot savu darbu ar prieku** . Mēs

        esam atbildīgi un **pildām savus solījumus** . Mums ir būtiska sava un kolēģu **labbūtība**, tāpēc

        esam cilvēcīgi un **rūpējamies**, lai ikviens **justos pieņemts un iekļauts** .


        1. Es uzturu līdzsvaru starp darbu un privāto dzīvi.

        2. Saspīlētas situācijas risinu, izmantojot veselīgu un atbilstošu humoru.

        3. Es veidoju konstruktīvas attiecības, izturos ar cieņu un līdzvērtīgi pret visiem


        cilvēkiem bankā un ārpus tās.

        4. Man ir neformālas sarunas ar darbiniekiem un es izrādu interesi par viņu


        profesionālo sniegumu un dzīvi ārpus darba.

        5. Plānojot darba uzdevumus, es ar izpratni izturos pret darbinieku individuālajām


        vajadzībām.

        6. Es praktizēju uzmanīgu un aktīvu klausīšanos.

        7. Es radu vidi, kurā darbinieki labprāt runā par saviem pārdzīvojumiem un mentālo


        labsajūtu.

        8. Es definēju skaidrus "spēles noteikumus" un pildu savus solījumus.

        9. Es pamanu darbinieku panākumus un atzinīgi tos novērtēju. 


        **Jaudīgi attīstībā**


        Ikviens no mums ir profesionālis ar savu unikālo pieredzi, ko visveiksmīgāk varam likt lietā

        tad, ja **dalāmies tajā ar citiem** . Mēs apzināmies – lai dotu, mums vispirms **jāapgūst pašiem** .

        Tāpēc strādājam atvērtām acīm un it visur **saskatām iespēju mācīties** . Mēs **pilnveidojamies**

        **paši un iedvesmojam to darīt citus**, lai kopā sekmētu mūsu valsts attīstību.


        1. Es identificēju savas stiprās puses un attīstības jomas.

        2. Es uzņemos personisku atbildību par jaunu zināšanu apguvi un atvēlu laiku sevis un


        citu pilnveidei.

        3. Es mācos no saviem un kolēģu veiksmes stāstiem un neizdošanās gadījumiem,


        pielāgoju savu rīcību

        4. Es pārzinu savu darbinieku profesionālos izaicinājumus, palīdzu izvirzīt attīstības


        mērķus un izstrādāt aktivitātes, lai tos sasniegtu.

        5. Es iesaistos un iesaistu darbiniekus projektos ārpus ierastā ikdienas darba.

        6. Es regulāri dalos ar savu pieredzi un zināšanām ar darbiniekiem un kolēģiem. 


    ### Instructions - seko instrukciju secībai;:
    1. Balsties TIKAI uz {{piedāvātās rindas}};
    2. Atlasi unikālos "prasmju aprakstus" no {{piedāvātās rindas}}, kuras ir vistuvāk {{lietotāja vaicājums}}
    3. Sakārto secībā atlikušos "prasmju aprakstus" un atlasi izvēlēto skaitu tās atbilstoši {{gala rezultātu atlase}}.
    4. Iegūstot sarakstu ar unikālajiem "prasmju aprakstiem", no rindām iegūsti arī atbilstošās pārējās kolonnas, balstoties uz tuvāko "lomu" pēc {{lietotāja vaicājums}} 
    5. Ja kādam prasmes aprakstam ir vairāk kā viena loma, tad uzskaiti vēl trīs populārākās lomas katrai prasmes apraksta rindai "Loma" kolonnā, bet neveido jaunas rindas, tikai papildini katras rindas "Lomas" kolonnu.
    6. Netulko ievietotās prasmju rindas, paturi oriģinālvalodu
    
    """
)

# ───────────────────────── Translations ───────────────────────────
TRANSLATIONS = {
    "lv": {
        "app_title": "MI asistents darba prasmēm",
        "app_subtitle": "Semantiskā meklēšana ESCO un SkillsFuture darba prasmju datubāzēs.",
        "lang_btn": "EN",
        "source_label": "Datu avots",
        "source_hint": "Izvēlies prasmju taksonomiju, kurā meklēt atbilstošās prasmes.",
        "query_label": "Ievadīt datus par sevi (lietotāja vaicājums)",
        "query_placeholder": "Apraksti savu amatu (vari iekopēt pienākumus), lomu, pieredzi, prasmes, intereses u.tml.",
        "query_hint": "Jo detalizētāks apraksts, jo precīzākas prasmes.",
        "topk_label": "Atbilstošāko darba prasmju skaits",
        "topk_hint": "No 1 līdz 100 unikālām prasmēm.",
        "topk_suffix": "prasmes",
        "run_btn": "Sākt MI asistenta darbību",
        "run_hint": "Tiks veikta semantiskā meklēšana un MI atlase no izvēlētās datubāzes.",
        "results_title": "MI asistenta apstrādātā gala atbilde",
        "results_subtitle": "Zemāk redzami atlasītie prasmju ieraksti tabulas formātā.",
        "download_btn": "Lejupielādēt rezultātus xlsx formātā",
        "download_hint": "Lejupielāde kļūs aktīva pēc MI asistenta veiksmīgas atbildes.",
    },
    "en": {
        "app_title": "AI Skills Assistant",
        "app_subtitle": "Semantic search across ESCO and SkillsFuture skills databases.",
        "lang_btn": "LV",
        "source_label": "Data source",
        "source_hint": "Choose the skills taxonomy to search for matching skills.",
        "query_label": "Enter your information (user query)",
        "query_placeholder": "Describe your job title (you can paste your responsibilities), role, experience, skills, interests, etc.",
        "query_hint": "The more detailed the description, the more precise the skills.",
        "topk_label": "Number of most relevant skills",
        "topk_hint": "From 1 to 100 unique skills.",
        "topk_suffix": "skills",
        "run_btn": "Start AI Assistant",
        "run_hint": "Semantic search and AI selection will be performed from the chosen database.",
        "results_title": "AI Assistant final response",
        "results_subtitle": "The selected skill records are displayed in table format below.",
        "download_btn": "Download results as xlsx",
        "download_hint": "Download becomes active after a successful AI assistant response.",
    },
}

INDEX_CONFIG_LABELS = {
    "lv": {
        "ESCO_en": "ESCO (Eiropas prasmju un kvalifikāciju datubāze)",
        "SkillsFuture": "SkillsFuture (Singapūras mūžizglītības prasmju datubāze)",
        "VAS_kompetences": "VAS kompetenču bibliotēka",
    },
    "en": {
        "ESCO_en": "ESCO (European Skills and Qualifications database)",
        "SkillsFuture": "SkillsFuture (Singapore lifelong learning skills database)",
        "VAS_kompetences": "VAS Competence Library",
    },
}

# ──────────────────────── FAISS index config/loader ─────────────────
APP_DIR = Path(__file__).resolve().parent


INDEX_CONFIG = {
    "ESCO_en": {
        "label": "ESCO (Eiropas prasmju un kvalifikāciju datubāze)",
        "path": Path("knowledge_base_sq8/faiss_esco_en"),
    },
    "SkillsFuture": {
        "label": "SkillsFuture (Singapūras mūžizglītības prasmju datubāze)",
        "path": Path("knowledge_base_sq8/faiss_skillsfuture_idx"),
    },
    "VAS_kompetences": {
        "label": "VAS kompetenču bibliotēka",
        "path": Path("knowledge_base_sq8/faiss_vas_kompetences"),
    },
}


def _reassemble_if_needed(index_dir: Path) -> None:
    """If index.faiss is split into .partNNN chunks, reassemble it."""
    target = index_dir / "index.faiss"
    parts = sorted(index_dir.glob("index.faiss.part*"))
    if not parts:
        return  # no split files – original must exist
    if target.exists() and target.stat().st_size > 0:
        return  # already reassembled
    with open(target, "wb") as out:
        for p in parts:
            out.write(p.read_bytes())


@lru_cache(maxsize=8)
def load_faiss_store(choice_key: str):
    cfg = INDEX_CONFIG.get(choice_key)
    if not cfg:
        raise ValueError(f"Unknown index key: {choice_key}")
    index_path = APP_DIR / cfg["path"]
    _reassemble_if_needed(index_path)
    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ───────────────────────────── Dash App ────────────────────────────
app = dash.Dash(__name__)
app.title = "PP MI asistents - darba prasmes"

app.layout = html.Div(
    style={
        "minHeight": "100vh",
        "backgroundColor": "#f3f4f6",
        "padding": "2rem",
        "boxSizing": "border-box",
        "fontFamily": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    },
    children=[
        html.Div(
            style={
                "maxWidth": "1100px",
                "margin": "0 auto",
            },
            children=[
                # Header card
                html.Div(
                    style={
                        "backgroundColor": "#111827",
                        "color": "white",
                        "borderRadius": "0.75rem",
                        "padding": "1.5rem 1.75rem",
                        "marginBottom": "1.5rem",
                        "boxShadow": "0 10px 25px rgba(0,0,0,0.15)",
                    },
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "gap": "1rem",
                                "flexWrap": "wrap",
                            },
                            children=[
                                html.Div(
                                    children=[
                                        html.H1(
                                            id="app-title",
                                            children="MI asistents darba prasmēm",
                                            style={
                                                "margin": 0,
                                                "fontSize": "1.75rem",
                                                "fontWeight": 700,
                                            },
                                        ),
                                        html.P(
                                            id="app-subtitle",
                                            children="Semantiskā meklēšana ESCO un SkillsFuture darba prasmju datubāzēs.",
                                            style={
                                                "margin": "0.4rem 0 0",
                                                "fontSize": "0.95rem",
                                                "opacity": 0.85,
                                            },
                                        ),
                                    ]
                                ),
                                html.Div(
                                    id="lang-toggle",
                                    style={
                                        "display": "flex",
                                        "alignItems": "center",
                                        "border": "2px solid rgba(255,255,255,0.5)",
                                        "borderRadius": "999px",
                                        "overflow": "hidden",
                                        "flexShrink": 0,
                                    },
                                    children=[
                                        html.Button(
                                            "LV",
                                            id="lang-btn-lv",
                                            n_clicks=0,
                                            style={
                                                "padding": "0.35rem 0.9rem",
                                                "border": "none",
                                                "backgroundColor": "white",
                                                "color": "#111827",
                                                "fontWeight": 700,
                                                "fontSize": "0.85rem",
                                                "cursor": "default",
                                                "letterSpacing": "0.05em",
                                            },
                                        ),
                                        html.Button(
                                            "EN",
                                            id="lang-btn-en",
                                            n_clicks=0,
                                            style={
                                                "padding": "0.35rem 0.9rem",
                                                "border": "none",
                                                "backgroundColor": "transparent",
                                                "color": "rgba(255,255,255,0.7)",
                                                "fontWeight": 700,
                                                "fontSize": "0.85rem",
                                                "cursor": "pointer",
                                                "letterSpacing": "0.05em",
                                            },
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),

                # Main card for inputs
                html.Div(
                    style={
                        "backgroundColor": "white",
                        "borderRadius": "0.75rem",
                        "padding": "1.5rem 1.75rem",
                        "boxShadow": "0 10px 25px rgba(15,23,42,0.08)",
                        "marginBottom": "1.5rem",
                    },
                    children=[
                        # Source choice row
                        html.Div(
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "gap": "1rem",
                                "marginBottom": "1.25rem",
                                "flexWrap": "wrap",
                            },
                            children=[
                                html.Div(
                                    children=[
                                        html.Label(
                                            id="source-label",
                                            children="Datu avots",
                                            style={
                                                "fontWeight": 600,
                                                "fontSize": "0.9rem",
                                                "display": "block",
                                            },
                                        ),
                                        dcc.RadioItems(
                                            id="faiss-choice",
                                            options=[
                                                {"label": v["label"], "value": k}
                                                for k, v in INDEX_CONFIG.items()
                                            ],
                                            value="ESCO_en",
                                            inline=True,
                                            style={"marginTop": "0.4rem"},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    id="source-hint",
                                    style={
                                        "fontSize": "0.8rem",
                                        "color": "#6b7280",
                                        "textAlign": "right",
                                    },
                                    children="Izvēlies prasmju taksonomiju, kurā meklēt atbilstošās prasmes.",
                                ),
                            ],
                        ),

                        html.Hr(style={"borderColor": "#e5e7eb", "margin": "1rem 0 1.25rem"}),

                        # User query
                        html.Div(
                            style={"marginBottom": "1.25rem"},
                            children=[
                                html.Label(
                                    id="query-label",
                                    children="Ievadīt datus par sevi (lietotāja vaicājums)",
                                    htmlFor="query-input",
                                    style={
                                        "fontWeight": 600,
                                        "fontSize": "0.9rem",
                                        "display": "block",
                                        "marginBottom": "0.35rem",
                                    },
                                ),
                                dcc.Textarea(
                                    id="query-input",
                                    placeholder="Apraksti savu amatu (vari iekopēt pienākumus), lomu, pieredzi, prasmes, intereses u.tml.",
                                    style={
                                        "width": "100%",
                                        "minHeight": "140px",
                                        "resize": "vertical",
                                        "padding": "0.75rem 0.85rem",
                                        "borderRadius": "0.6rem",
                                        "border": "1px solid #d1d5db",
                                        "fontSize": "0.95rem",
                                        "boxSizing": "border-box",
                                        "backgroundColor": "#f9fafb",
                                    },
                                ),
                                html.Div(
                                    id="query-hint",
                                    children="Jo detalizētāks apraksts, jo precīzākas prasmes.",
                                    style={
                                        "marginTop": "0.35rem",
                                        "fontSize": "0.8rem",
                                        "color": "#6b7280",
                                    },
                                ),
                            ],
                        ),

                        # Top-K + run button row
                        html.Div(
                            style={
                                "display": "flex",
                                "alignItems": "flex-end",
                                "gap": "1.25rem",
                                "flexWrap": "wrap",
                            },
                            children=[
                                html.Div(
                                    style={"flex": "0 0 240px"},
                                    children=[
                                        html.Label(
                                            id="topk-label",
                                            children="Atbilstošāko darba prasmju skaits",
                                            style={
                                                "fontWeight": 600,
                                                "fontSize": "0.9rem",
                                                "display": "block",
                                                "marginBottom": "0.35rem",
                                            },
                                        ),
                                        html.Div(
                                            style={
                                                "display": "flex",
                                                "alignItems": "center",
                                                "gap": "0.5rem",
                                            },
                                            children=[
                                                dcc.Input(
                                                    id="llm-topk-input",
                                                    type="number",
                                                    min=1,
                                                    max=100,
                                                    step=1,
                                                    value=10,
                                                    style={
                                                        "width": "5rem",
                                                        "padding": "0.45rem 0.5rem",
                                                        "borderRadius": "0.5rem",
                                                        "border": "1px solid #d1d5db",
                                                        "fontSize": "0.95rem",
                                                    },
                                                ),
                                                html.Span(
                                                    id="topk-suffix",
                                                    children="prasmes",
                                                    style={
                                                        "fontSize": "0.9rem",
                                                        "color": "#4b5563",
                                                    },
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            id="topk-hint",
                                            children="No 1 līdz 100 unikālām prasmēm.",
                                            style={
                                                "marginTop": "0.35rem",
                                                "fontSize": "0.8rem",
                                                "color": "#6b7280",
                                            },
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style={"flex": "1"},
                                    children=[
                                        html.Button(
                                            id="run-llm-btn",
                                            children="Sākt MI asistenta darbību",
                                            n_clicks=0,
                                            style={
                                                "padding": "0.6rem 1.4rem",
                                                "borderRadius": "999px",
                                                "border": "none",
                                                "backgroundColor": "#2563eb",
                                                "color": "white",
                                                "fontWeight": 600,
                                                "fontSize": "0.95rem",
                                                "cursor": "pointer",
                                                "boxShadow": "0 8px 16px rgba(37,99,235,0.35)",
                                            },
                                        ),
                                        html.Div(
                                            id="run-hint",
                                            children="Tiks veikta semantiskā meklēšana un MI atlase no izvēlētās datubāzes.",
                                            style={
                                                "marginTop": "0.4rem",
                                                "fontSize": "0.8rem",
                                                "color": "#6b7280",
                                            },
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),

                # Results card
                html.Div(
                    style={
                        "backgroundColor": "white",
                        "borderRadius": "0.75rem",
                        "padding": "1.5rem 1.75rem",
                        "boxShadow": "0 10px 25px rgba(15,23,42,0.08)",
                        "marginBottom": "1rem",
                    },
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "gap": "1rem",
                                "marginBottom": "0.75rem",
                                "flexWrap": "wrap",
                            },
                            children=[
                                html.Div(
                                    children=[
                                        html.H3(
                                            id="results-title",
                                            children="MI asistenta apstrādātā gala atbilde",
                                            style={
                                                "margin": 0,
                                                "fontSize": "1.1rem",
                                                "fontWeight": 700,
                                                "color": "#111827",
                                            },
                                        ),
                                        html.P(
                                            id="results-subtitle",
                                            children="Zemāk redzami atlasītie prasmju ieraksti tabulas formātā.",
                                            style={
                                                "margin": "0.25rem 0 0",
                                                "fontSize": "0.85rem",
                                                "color": "#6b7280",
                                            },
                                        ),
                                    ]
                                ),
                            ],
                        ),
                        dcc.Loading(
                            id="loading-llm-table",
                            type="circle",
                            children=[
                                dash_table.DataTable(
                                    id="llm-table",
                                    data=[],
                                    columns=[],
                                    page_size=25,
                                    style_as_list_view=True,
                                    style_table={
                                        "maxHeight": "60vh",
                                        "overflowY": "auto",
                                        "overflowX": "auto",
                                        "borderRadius": "0.5rem",
                                        "border": "1px solid #e5e7eb",
                                    },
                                    style_header={
                                        "fontWeight": "600",
                                        "backgroundColor": "#111827",
                                        "color": "white",
                                        "border": "none",
                                        "padding": "0.6rem 0.75rem",
                                    },
                                    style_cell={
                                        "textAlign": "left",
                                        "minWidth": "120px",
                                        "whiteSpace": "pre-line",
                                        "padding": "0.5rem 0.75rem",
                                        "border": "none",
                                        "fontSize": "0.9rem",
                                    },
                                    style_data={
                                        "borderBottom": "1px solid #e5e7eb",
                                    },
                                    style_data_conditional=[
                                        {
                                            "if": {"row_index": "odd"},
                                            "backgroundColor": "#f9fafb",
                                        },
                                        {
                                            "if": {"state": "active"},
                                            "backgroundColor": "#dbeafe",
                                            "border": "1px solid #93c5fd",
                                        },
                                    ],
                                )
                            ],
                        ),
                    ],
                ),

                # Download row
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "gap": "1rem",
                        "flexWrap": "wrap",
                    },
                    children=[
                        html.Div(
                            children=[
                                html.Button(
                                    id="btn-download",
                                    children="Lejupielādēt rezultātus xlsx formātā",
                                    n_clicks=0,
                                    disabled=True,
                                    style={
                                        "padding": "0.6rem 1.4rem",
                                        "borderRadius": "999px",
                                        "border": "none",
                                        "backgroundColor": "#059669",
                                        "color": "white",
                                        "fontWeight": 600,
                                        "fontSize": "0.95rem",
                                        "cursor": "pointer",
                                        "boxShadow": "0 8px 16px rgba(5,150,105,0.35)",
                                        "opacity": 0.7,
                                    },
                                ),
                            ]
                        ),
                        html.Div(
                            id="download-hint",
                            children="Lejupielāde kļūs aktīva pēc MI asistenta veiksmīgas atbildes.",
                            style={
                                "fontSize": "0.8rem",
                                "color": "#6b7280",
                            },
                        ),
                    ],
                ),

                # Hidden store + download component
                dcc.Store(id="llm-output-store"),
                dcc.Store(id="lang-store", data="en"),
                dcc.Download(id="download-results"),
            ],
        )
    ],
)


# ─────────────────────────── Callbacks ─────────────────────────────
@app.callback(
    Output("llm-output-store", "data"),
    Output("llm-table", "data"),
    Output("llm-table", "columns"),
    Input("run-llm-btn", "n_clicks"),
    State("query-input", "value"),
    State("llm-topk-input", "value"),
    State("faiss-choice", "value"),
    State("lang-store", "data"),
    prevent_initial_call=True,
)
def run_full_pipeline(n_clicks, user_text, llm_topk, faiss_choice, lang):
    """
    Single entrypoint:
    - FAISS meklēšana fonā (k = 10 * topk) izvēlētajā indeksā (ESCO_en / SkillsFuture)
    - Top unikālās prasmes (topk_unique_prasme)
    - LLM vaicājums ar DEFAULT_SYSTEM_PROMPT
    - JSON atbilde tiek saglabāta llm-output-store, un paralēli tiek atjaunota tabula.

    dcc.Loading ap llm-table parāda progress ring visā šī callback izpildes laikā.
    """
    _COL_NAMES_EN = {
        "Loma": "Role",
        "Prasme": "Skill",
        "Prasmes apraksts": "Skill description",
        "Citas lomas": "Other roles",
    }

    def _make_columns(df_cols, lang):
        if (lang or "en") == "en":
            return [{"name": _COL_NAMES_EN.get(col, col), "id": col} for col in df_cols]
        return [{"name": col, "id": col} for col in df_cols]

    if not n_clicks:
        raise de.PreventUpdate

    user_text = (user_text or "").strip()
    if not user_text:
        data = [{"Kļūda": "Lūdzu, ievadīt vaicājumu."}]
        df = pd.DataFrame(data)
        # (no "Secība" here, but we keep logic consistent)
        df_display = df.drop(columns=["Secība"], errors="ignore")
        columns = _make_columns(df_display.columns, lang)
        return {"text": "Lūdzu, ievadīt vaicājumu."}, df_display.to_dict("records"), columns

    # Only changeable numeric parameter: topk
    try:
        llm_topk = int(llm_topk or 10)
    except ValueError:
        llm_topk = 10
    llm_topk = max(1, min(llm_topk, 100))

    # FAISS search in background; k = 10 * topk
    target_unique = llm_topk * 10
    index_key = faiss_choice or "ESCO_en"

    try:
        vector_store = load_faiss_store(index_key)
        source_label = INDEX_CONFIG[index_key]["label"]
        rows = topk_unique_prasme(user_text, target_unique, vector_store, source_label, index_key)
    except Exception as e:
        err = f"Embedding/FAISS kļūda: {e}"
        data = [{"Kļūda": err}]
        df = pd.DataFrame(data)
        df_display = df.drop(columns=["Secība"], errors="ignore")
        columns = _make_columns(df_display.columns, lang)
        return {"text": err}, df_display.to_dict("records"), columns

    # Prepare LLM prompt
    is_vas = (index_key == "VAS_kompetences")
    gala_sel = llm_topk
    entity_name = "kompetenču" if is_vas else "prasmju"
    human_prompt = (
        f"# Lietotāja vaicājums: \n\n"
        f"### Gala rezultātu atlase. Obligāti apkopo no piedāvātājām {entity_name} tabulas rindām šādu atbilstošako rindu skaitu: {gala_sel}):\n\n"
        f"## Ievadītie dati:\n\n"
        f"### Lietotāja vaicājums: {user_text}\n\n"
        f"### Piedāvātās rindas: {rows}"
    )

    system_prompt = VAS_SYSTEM_PROMPT.strip() if is_vas else DEFAULT_SYSTEM_PROMPT.strip()
    active_llm = chat_llm_vas if is_vas else chat_llm_skills

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt.strip()},
    ]

    try:
        response = active_llm.invoke(messages)

        # response is a SkillRows / VASRows object
        parsed = response.model_dump(by_alias=True)["rows"]

        summary = json.dumps(parsed, ensure_ascii=False, indent=2)
        store_value = summary
    except Exception as exc:
        error_msg = f"LLM error: {exc}"
        data = [{"Kļūda": error_msg}]
        df = pd.DataFrame(data)
        df_display = df.drop(columns=["Secība"], errors="ignore")
        columns = _make_columns(df_display.columns, lang)
        return {"text": error_msg}, df_display.to_dict("records"), columns

    # Convert JSON to rows/columns for the DataTable
    try:
        parsed = json.loads(summary)
    except Exception:
        parsed = [{"Atbilde": summary}]

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        parsed = [{"Atbilde": str(parsed)}]

    df = pd.DataFrame(parsed)

    # ── Remove "Secība" column from the displayed table ─────────────
    df_display = df.drop(columns=["Secība"], errors="ignore")

    columns = _make_columns(df_display.columns, lang)

    return store_value, df_display.to_dict("records"), columns


@app.callback(
    Output("btn-download", "disabled"),
    Input("llm-output-store", "data"),
)
def toggle_download_button(llm_data):
    return not bool(llm_data)


@app.callback(
    Output("download-results", "data"),
    Input("btn-download", "n_clicks"),
    State("llm-output-store", "data"),
    prevent_initial_call=True,
)
def generate_xlsx(_, llm_data):
    if not llm_data:
        raise de.PreventUpdate

    data = llm_data
    if isinstance(llm_data, str):
        try:
            data = json.loads(llm_data)
        except Exception:
            # if it's just a JSON string from the LLM, wrap it
            data = [{"Atbilde": llm_data}]

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        # nothing sensible to export
        raise de.PreventUpdate

    df = pd.DataFrame(data)

    # ── Remove "Secība" column from export ─────────────────────────
    if "Secība" in df.columns:
        df = df.drop(columns=["Secība"])

    try:
        return dcc.send_data_frame(
            df.to_excel,
            "llm_summary.xlsx",
            sheet_name="LLM_Summary",
            index=False,
            engine="openpyxl",
        )
    except Exception as e:
        # Fallback: send a simple text file with the error so it's visible in the browser
        error_text = f"Neizdevās ģenerēt XLSX failu.\n\nKļūda:\n{repr(e)}"
        return dict(
            content=error_text,
            filename="download_error.txt",
            type="text/plain",
        )

# ─────────────────────── Language callbacks ────────────────────────
@app.callback(
    Output("lang-store", "data"),
    Input("lang-btn-lv", "n_clicks"),
    Input("lang-btn-en", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_language(n_lv, n_en):
    from dash import ctx
    return "lv" if ctx.triggered_id == "lang-btn-lv" else "en"


_ACTIVE_STYLE = {
    "padding": "0.35rem 0.9rem",
    "border": "none",
    "backgroundColor": "white",
    "color": "#111827",
    "fontWeight": 700,
    "fontSize": "0.85rem",
    "cursor": "default",
    "letterSpacing": "0.05em",
}
_INACTIVE_STYLE = {
    "padding": "0.35rem 0.9rem",
    "border": "none",
    "backgroundColor": "transparent",
    "color": "rgba(255,255,255,0.7)",
    "fontWeight": 700,
    "fontSize": "0.85rem",
    "cursor": "pointer",
    "letterSpacing": "0.05em",
}


@app.callback(
    Output("lang-btn-lv", "style"),
    Output("lang-btn-en", "style"),
    Output("app-title", "children"),
    Output("app-subtitle", "children"),
    Output("source-label", "children"),
    Output("source-hint", "children"),
    Output("faiss-choice", "options"),
    Output("query-label", "children"),
    Output("query-input", "placeholder"),
    Output("query-hint", "children"),
    Output("topk-label", "children"),
    Output("topk-suffix", "children"),
    Output("topk-hint", "children"),
    Output("run-llm-btn", "children"),
    Output("run-hint", "children"),
    Output("results-title", "children"),
    Output("results-subtitle", "children"),
    Output("btn-download", "children"),
    Output("download-hint", "children"),
    Input("lang-store", "data"),
)
def update_ui_language(lang):
    lang = lang or "lv"
    t = TRANSLATIONS[lang]
    labels = INDEX_CONFIG_LABELS[lang]
    radio_options = [{"label": labels[k], "value": k} for k in INDEX_CONFIG]
    lv_style = _ACTIVE_STYLE if lang == "lv" else _INACTIVE_STYLE
    en_style = _ACTIVE_STYLE if lang == "en" else _INACTIVE_STYLE
    return (
        lv_style,
        en_style,
        t["app_title"],
        t["app_subtitle"],
        t["source_label"],
        t["source_hint"],
        radio_options,
        t["query_label"],
        t["query_placeholder"],
        t["query_hint"],
        t["topk_label"],
        t["topk_suffix"],
        t["topk_hint"],
        t["run_btn"],
        t["run_hint"],
        t["results_title"],
        t["results_subtitle"],
        t["download_btn"],
        t["download_hint"],
    )


# ─────────────────────────── Run server ────────────────────────────
if __name__ == "__main__":
    app.run()
