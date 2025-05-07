# Zonneaccu Analyse Tool

## Overzicht
De Zonneaccu Analyse Tool is een Streamlit-applicatie die gebruikers helpt bij het analyseren van de rendabiliteit van verschillende opslagmethoden voor overtollige zonne-energie. Het doel is om inzicht te bieden in de kosten en besparingen van warmwaterboilers en accu's als opslagmethoden voor zonnepaneeloverschotten.

## Functionaliteiten
- Importeren en analyseren van historische energieproductie- en verbruiksdata
- Berekenen van potentiële besparingen door warmwaterboileropslagsystemen
- Berekenen van potentiële besparingen door accuopslagsystemen
- Vergelijken van verschillende opslagmethoden op basis van kosten en rendement
- Visualiseren van energiestromen en besparingsprofielen

## Installatie

```bash
# Clone de repository
git clone https://github.com/Fbeunder/zonneaccu.git
cd zonneaccu

# Maak een virtuele omgeving aan (optioneel maar aanbevolen)
python -m venv venv
source venv/bin/activate  # Op Windows: venv\Scripts\activate

# Installeer de vereiste packages
pip install -r requirements.txt
```

## Gebruik

```bash
streamlit run app.py
```

Open vervolgens een webbrowser en ga naar de URL die door Streamlit wordt getoond (meestal http://localhost:8501).

## Projectstructuur

```
zonneaccu/
├── app.py                     # Hoofdapplicatie
├── data_processing/           # Modules voor dataverwerking
│   ├── __init__.py
│   ├── data_loader.py         # Functionaliteit voor het laden van data
├── storage_modules/           # Opslagmodules
│   ├── __init__.py
├── ui_components/            # UI-componenten
│   ├── __init__.py
├── utils/                    # Hulpfuncties
│   ├── __init__.py
│   ├── config_manager.py     # Beheer van configuratie
└── requirements.txt          # Projectafhankelijkheden
```

## Vereisten
- Python 3.8 of hoger
- Zie requirements.txt voor alle packagevereisten

## Bijdragen
Bijdragen aan dit project zijn welkom! Voel je vrij om issues aan te maken of pull requests in te dienen.

## Licentie
Dit project is beschikbaar onder de MIT-licentie.
