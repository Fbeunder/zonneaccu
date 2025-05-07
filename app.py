"""Hoofdapplicatie voor de Zonneaccu Analyse Tool.

Deze Streamlit applicatie stelt gebruikers in staat om de potentiële besparingen
te analyseren van verschillende energieopslagmethoden voor zonnepaneeloverschotten.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import packages
from data_processing import data_loader, data_analysis
from ui_components import data_import_page, data_analysis_page, boiler_analysis_page
from storage_modules import boiler_module
from utils import config_manager


def main():
    """Hoofdfunctie die de Streamlit applicatie initialiseert en uitvoert."""
    # Configureer de pagina
    st.set_page_config(
        page_title="Zonneaccu Analyse",
        page_icon="☀️",
        layout="wide",
    )
    
    # Initialiseer sessiestate variabelen als ze nog niet bestaan
    if 'data_loaded' not in st.session_state:
        st.session_state['data_loaded'] = False
    if 'energy_data' not in st.session_state:
        st.session_state['energy_data'] = None
    
    # Toon hoofdtitel
    st.title("Zonneaccu Analyse Tool")
    st.markdown("""Deze tool helpt bij het analyseren van de rendabiliteit van 
                verschillende opslagmethoden voor overschotten van zonnepanelen.""")
    
    # Sidebar voor navigatie en configutatie
    with st.sidebar:
        st.subheader("Navigatie")
        page = st.radio("Ga naar:", [
            "Home",
            "Data Import",
            "Data Analyse",
            "Boiler Analyse",
            "Accu Analyse",
            "Vergelijking"
        ])
        
        st.subheader("Configuratie")
        # Placeholder voor toekomstige configuratieopties
        
        # Toon data status
        if st.session_state['data_loaded']:
            st.success("✅ Data geladen")
        else:
            st.warning("❌ Geen data geladen")
    
    # Toon de geselecteerde pagina
    if page == "Home":
        st.subheader("Welkom!")
        st.write("""
        Upload uw energiedata om te beginnen met de analyse.
        Navigeer via het menu links naar de verschillende secties.
        """)
        
        # Toon een korte uitleg over de applicatie
        st.subheader("Over deze applicatie")
        st.markdown("""
        Deze applicatie helpt bij het analyseren van de potentiële besparingen door overproductie 
        van zonnepanelen op te slaan. U kunt verschillende opslagmethoden evalueren, zoals:
        
        - **Warmwaterboiler**: Opslag van energie in de vorm van warm water
        - **Accu**: Opslag van elektrische energie in een batterij
        
        Begin door uw energiedata te uploaden via de 'Data Import' pagina.
        Analyseer vervolgens de patronen in uw data via de 'Data Analyse' pagina.
        Evalueer daarna verschillende opslagmethoden via de 'Boiler Analyse' en 'Accu Analyse' pagina's.
        """)
        
    elif page == "Data Import":
        # Gebruik de data import component
        data_import_page.render_data_import_page()
        
    elif page == "Data Analyse":
        # Gebruik de data analyse component
        data_analysis_page.render_data_analysis_page()
        
    elif page == "Boiler Analyse":
        # Gebruik de boiler analyse component
        boiler_analysis_page.render_boiler_analysis_page()
        
    elif page == "Accu Analyse":
        st.subheader("Accu Analyse")
        
        # Controleer of er data is geladen
        if not st.session_state.get('data_loaded', False):
            st.warning("Laad eerst energiedata via de 'Data Import' pagina.")
            st.stop()
            
        # Placeholder voor toekomstige accu analyse
        st.info("De accu analyse functionaliteit wordt binnenkort geïmplementeerd.")
        
    elif page == "Vergelijking":
        st.subheader("Vergelijking Opslagmethoden")
        
        # Controleer of er data is geladen
        if not st.session_state.get('data_loaded', False):
            st.warning("Laad eerst energiedata via de 'Data Import' pagina.")
            st.stop()
            
        # Placeholder voor toekomstige vergelijkingsfunctionaliteit
        st.info("De vergelijkingsfunctionaliteit wordt binnenkort geïmplementeerd.")


if __name__ == "__main__":
    main()