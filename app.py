"""Hoofdapplicatie voor de Zonneaccu Analyse Tool.

Deze Streamlit applicatie stelt gebruikers in staat om de potentiële besparingen
te analyseren van verschillende energieopslagmethoden voor zonnepaneeloverschotten.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import packages
from data_processing import data_loader
from utils import config_manager


def main():
    """Hoofdfunctie die de Streamlit applicatie initialiseert en uitvoert."""
    # Configureer de pagina
    st.set_page_config(
        page_title="Zonneaccu Analyse",
        page_icon="☀️",
        layout="wide",
    )
    
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
            "Boiler Analyse",
            "Accu Analyse",
            "Vergelijking"
        ])
        
        st.subheader("Configuratie")
        # Placeholder voor toekomstige configuratieopties
        
    # Toon de geselecteerde pagina (placeholder functionaliteit)
    if page == "Home":
        st.subheader("Welkom!")
        st.write("""
        Upload uw energiedata om te beginnen met de analyse.
        Navigeer via het menu links naar de verschillende secties.
        """)
        
    elif page == "Data Import":
        st.subheader("Data Importeren")
        upload_file = st.file_uploader("Upload CSV met energiedata", type=['csv'])
        # Placeholder voor toekomstige data import functionaliteit
        
    elif page == "Boiler Analyse":
        st.subheader("Warmwaterboiler Analyse")
        # Placeholder voor toekomstige boiler analyse
        
    elif page == "Accu Analyse":
        st.subheader("Accu Analyse")
        # Placeholder voor toekomstige accu analyse
        
    elif page == "Vergelijking":
        st.subheader("Vergelijking Opslagmethoden")
        # Placeholder voor toekomstige vergelijkingsfunctionaliteit


if __name__ == "__main__":
    main()
