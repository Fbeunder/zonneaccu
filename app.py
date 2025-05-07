"""Hoofdapplicatie voor de Zonneaccu Analyse Tool.

Deze Streamlit applicatie stelt gebruikers in staat om de potentiële besparingen
te analyseren van verschillende energieopslagmethoden voor zonnepaneeloverschotten.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Import packages
from data_processing import data_loader, data_analysis
from ui_components import data_import_page, data_analysis_page, boiler_analysis_page, battery_analysis_page, config_page
from storage_modules import boiler_module, battery_module
from utils import config_manager


def initialize_session_state():
    """Initialiseer de sessiestate variabelen als ze nog niet bestaan."""
    if 'data_loaded' not in st.session_state:
        st.session_state['data_loaded'] = False
    if 'energy_data' not in st.session_state:
        st.session_state['energy_data'] = None
    if 'current_config' not in st.session_state:
        # Laad standaard configuratie
        st.session_state['current_config'] = config_manager.DEFAULT_CONFIG.copy()
    if 'active_config_profile' not in st.session_state:
        st.session_state['active_config_profile'] = None


def apply_configuration():
    """Pas de huidige configuratie toe op de sessiestate."""
    # Hier kunnen we configuratiewaarden toepassen op andere delen van de applicatie
    # Bijvoorbeeld door ze in de sessiestate te zetten
    pass


def main():
    """Hoofdfunctie die de Streamlit applicatie initialiseert en uitvoert."""
    # Configureer de pagina
    st.set_page_config(
        page_title="Zonneaccu Analyse",
        page_icon="☀️",
        layout="wide",
    )
    
    # Initialiseer sessiestate
    initialize_session_state()
    
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
            "Vergelijking",
            "Configuratie"
        ])
        
        st.subheader("Status")
        # Toon data status
        if st.session_state['data_loaded']:
            st.success("✅ Data geladen")
        else:
            st.warning("❌ Geen data geladen")
        
        # Toon actieve configuratie
        active_profile = st.session_state.get('active_config_profile')
        if active_profile:
            st.success(f"✅ Configuratie: {active_profile}")
        else:
            st.info("ℹ️ Standaard configuratie actief")
        
        # Snelle configuratie laden
        st.subheader("Snelle Configuratie")
        config_mgr = config_manager.get_config_manager()
        config_profiles = config_mgr.get_config_list()
        
        if config_profiles:
            selected_profile = st.selectbox(
                "Laad configuratie:",
                options=["Standaard"] + config_profiles,
                index=0
            )
            
            if st.button("Toepassen"):
                if selected_profile == "Standaard":
                    st.session_state['current_config'] = config_manager.DEFAULT_CONFIG.copy()
                    st.session_state['active_config_profile'] = None
                    st.success("Standaard configuratie geladen!")
                else:
                    loaded_config = config_mgr.load_config(selected_profile)
                    st.session_state['current_config'] = loaded_config
                    st.session_state['active_config_profile'] = selected_profile
                    st.success(f"Configuratie '{selected_profile}' geladen!")
                
                # Pas de configuratie toe
                apply_configuration()
                st.rerun()
    
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
        
        U kunt uw configuratie-instellingen beheren via de 'Configuratie' pagina.
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
        # Gebruik de accu analyse component
        battery_analysis_page.render_battery_analysis_page()
        
    elif page == "Configuratie":
        # Gebruik de configuratie component
        config_page.render_config_page()
        
    elif page == "Vergelijking":
        st.subheader("Vergelijking Opslagmethoden")
        
        # Controleer of er data is geladen
        if not st.session_state.get('data_loaded', False):
            st.warning("Laad eerst energiedata via de 'Data Import' pagina.")
            st.stop()
            
        # Controleer of er simulatieresultaten zijn voor beide opslagmethoden
        if 'boiler_simulation_results' not in st.session_state or 'battery_simulation_results' not in st.session_state:
            st.warning("Voer eerst zowel de boiler als de accu simulatie uit via de respectievelijke analyse pagina's.")
            st.stop()
        
        # Haal de simulatieresultaten op
        boiler_results = st.session_state['boiler_simulation_results']
        battery_results = st.session_state['battery_simulation_results']
        
        # Toon vergelijking van de belangrijkste resultaten
        st.subheader("Vergelijking van Opslagmethoden")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Warmwaterboiler")
            st.metric(
                "Opgeslagen Energie",
                f"{boiler_results['total_energy_to_boiler']/1000:.1f} kWh"
            )
            st.metric(
                "Nuttig Gebruikte Energie",
                f"{boiler_results['total_energy_from_boiler']/1000:.1f} kWh"
            )
            st.metric(
                "Efficiëntie",
                f"{boiler_results['boiler_efficiency']:.1f}%"
            )
            st.metric(
                "Benutting Energieoverschot",
                f"{boiler_results['surplus_utilization']:.1f}%"
            )
        
        with col2:
            st.markdown("### Accu")
            st.metric(
                "Opgeslagen Energie",
                f"{battery_results['total_energy_to_battery']/1000:.1f} kWh"
            )
            st.metric(
                "Nuttig Gebruikte Energie",
                f"{battery_results['total_energy_from_battery']/1000:.1f} kWh"
            )
            st.metric(
                "Round-trip Efficiëntie",
                f"{battery_results['round_trip_efficiency']:.1f}%"
            )
            st.metric(
                "Toename Zelfconsumptie",
                f"{battery_results['self_consumption_increase']:.1f}%"
            )
        
        # Toon vergelijking van financiële resultaten
        st.subheader("Financiële Vergelijking")
        
        # Bereken jaarlijkse besparingen
        boiler_daily_savings = boiler_results['savings'] / boiler_results.get('simulation_days', 1)
        boiler_annual_savings = boiler_daily_savings * 365
        
        battery_daily_savings = battery_results['savings'] / battery_results.get('simulation_days', 1)
        battery_annual_savings = battery_daily_savings * 365
        
        # Maak een vergelijkingsgrafiek
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Warmwaterboiler', 'Accu'],
            y=[boiler_annual_savings, battery_annual_savings],
            name='Jaarlijkse Besparingen',
            marker_color=['#1f77b4', '#ff7f0e']
        ))
        
        fig.update_layout(
            title="Vergelijking Jaarlijkse Besparingen",
            xaxis_title="Opslagmethode",
            yaxis_title="Jaarlijkse Besparingen (€)",
            yaxis=dict(tickformat='€,.2f')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Toon conclusie
        st.subheader("Conclusie")
        
        if boiler_annual_savings > battery_annual_savings:
            st.success(f"""De **warmwaterboiler** lijkt de meest rendabele optie te zijn met jaarlijkse besparingen van 
                      €{boiler_annual_savings:.2f} tegenover €{battery_annual_savings:.2f} voor de accu.""")
        elif battery_annual_savings > boiler_annual_savings:
            st.success(f"""De **accu** lijkt de meest rendabele optie te zijn met jaarlijkse besparingen van 
                      €{battery_annual_savings:.2f} tegenover €{boiler_annual_savings:.2f} voor de warmwaterboiler.""")
        else:
            st.info("Beide opslagmethoden lijken ongeveer even rendabel te zijn.")
        
        st.markdown("""
        **Overweeg bij uw keuze ook:**
        - Initiële investeringskosten
        - Levensduur van de systemen
        - Ruimtevereisten
        - Installatiegemak
        - Uw specifieke energiebehoeften
        """)
        
        # Voeg een knop toe om de huidige configuratie op te slaan
        if st.button("Huidige configuratie opslaan"):
            st.session_state['_redirect_to_config'] = True
            st.rerun()
    
    # Controleer of we moeten doorverwijzen naar de configuratiepagina
    if st.session_state.get('_redirect_to_config', False):
        st.session_state['_redirect_to_config'] = False
        st.switch_page("Configuratie")


if __name__ == "__main__":
    main()