"""UI-component voor het beheren van configuraties.

Deze module biedt een gebruikersinterface voor het beheren van configuratieprofielen,
waarmee gebruikers hun instellingen kunnen opslaan, laden en delen.
"""

import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

from utils.config_manager import get_config_manager, DEFAULT_CONFIG


def render_config_page():
    """Render de configuratiepagina."""
    st.subheader("Configuratiebeheer")
    
    # Initialiseer de config manager
    config_manager = get_config_manager()
    
    # Haal beschikbare configuratieprofielen op
    config_profiles = config_manager.get_config_list()
    
    # Toon uitleg
    st.markdown("""
    Op deze pagina kunt u uw configuratie-instellingen beheren. U kunt:
    - Configuratieprofielen opslaan en laden
    - Verschillende profielen maken voor verschillende scenario's
    - Configuraties exporteren en importeren om te delen
    """)
    
    # Toon huidige configuratie
    st.subheader("Huidige Configuratie")
    
    # Haal huidige configuratie op uit sessie state of gebruik standaard
    current_config = st.session_state.get('current_config', DEFAULT_CONFIG.copy())
    
    # Toon configuratie in tabs
    tabs = st.tabs(["Boiler", "Accu", "Energie"])
    
    with tabs[0]:
        st.subheader("Boiler Instellingen")
        boiler_config = current_config.get('boiler', {})
        
        col1, col2 = st.columns(2)
        with col1:
            boiler_capacity = st.number_input(
                "Boiler Capaciteit (liters)",
                min_value=50.0,
                max_value=1000.0,
                value=float(boiler_config.get('capacity_liters', 200)),
                step=10.0
            )
            
            boiler_temp_min = st.number_input(
                "Minimum Temperatuur (°C)",
                min_value=20.0,
                max_value=60.0,
                value=float(boiler_config.get('temperature_min', 40)),
                step=1.0
            )
        
        with col2:
            boiler_temp_max = st.number_input(
                "Maximum Temperatuur (°C)",
                min_value=40.0,
                max_value=95.0,
                value=float(boiler_config.get('temperature_max', 80)),
                step=1.0
            )
            
            boiler_efficiency = st.number_input(
                "Efficiëntie",
                min_value=0.5,
                max_value=1.0,
                value=float(boiler_config.get('efficiency', 0.95)),
                step=0.01,
                format="%.2f"
            )
        
        # Update boiler configuratie
        current_config['boiler'] = {
            'capacity_liters': boiler_capacity,
            'temperature_min': boiler_temp_min,
            'temperature_max': boiler_temp_max,
            'efficiency': boiler_efficiency
        }
    
    with tabs[1]:
        st.subheader("Accu Instellingen")
        battery_config = current_config.get('battery', {})
        
        col1, col2 = st.columns(2)
        with col1:
            battery_capacity = st.number_input(
                "Accu Capaciteit (kWh)",
                min_value=0.5,
                max_value=100.0,
                value=float(battery_config.get('capacity_kwh', 5)),
                step=0.5
            )
            
            battery_power = st.number_input(
                "Maximum Vermogen (kW)",
                min_value=0.5,
                max_value=20.0,
                value=float(battery_config.get('max_power_kw', 2.5)),
                step=0.1
            )
            
            battery_dod = st.number_input(
                "Depth of Discharge",
                min_value=0.1,
                max_value=1.0,
                value=float(battery_config.get('depth_of_discharge', 0.8)),
                step=0.05,
                format="%.2f"
            )
        
        with col2:
            battery_charge_eff = st.number_input(
                "Laad Efficiëntie",
                min_value=0.5,
                max_value=1.0,
                value=float(battery_config.get('efficiency_charge', 0.92)),
                step=0.01,
                format="%.2f"
            )
            
            battery_discharge_eff = st.number_input(
                "Ontlaad Efficiëntie",
                min_value=0.5,
                max_value=1.0,
                value=float(battery_config.get('efficiency_discharge', 0.92)),
                step=0.01,
                format="%.2f"
            )
        
        # Update accu configuratie
        current_config['battery'] = {
            'capacity_kwh': battery_capacity,
            'max_power_kw': battery_power,
            'efficiency_charge': battery_charge_eff,
            'efficiency_discharge': battery_discharge_eff,
            'depth_of_discharge': battery_dod
        }
    
    with tabs[2]:
        st.subheader("Energie Instellingen")
        energy_config = current_config.get('energy', {})
        
        col1, col2 = st.columns(2)
        with col1:
            electricity_price = st.number_input(
                "Elektriciteitsprijs (€/kWh)",
                min_value=0.01,
                max_value=1.0,
                value=float(energy_config.get('electricity_price', 0.28)),
                step=0.01,
                format="%.2f"
            )
        
        with col2:
            gas_price = st.number_input(
                "Gasprijs (€/m³)",
                min_value=0.1,
                max_value=5.0,
                value=float(energy_config.get('gas_price', 1.10)),
                step=0.05,
                format="%.2f"
            )
            
            feed_in_tariff = st.number_input(
                "Terugleververgoeding (€/kWh)",
                min_value=0.0,
                max_value=1.0,
                value=float(energy_config.get('feed_in_tariff', 0.08)),
                step=0.01,
                format="%.2f"
            )
        
        # Update energie configuratie
        current_config['energy'] = {
            'electricity_price': electricity_price,
            'gas_price': gas_price,
            'feed_in_tariff': feed_in_tariff
        }
    
    # Sla de huidige configuratie op in de sessie state
    st.session_state['current_config'] = current_config
    
    # Toon configuratiebeheer opties
    st.subheader("Configuratieprofielen")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Profiel opslaan
        st.markdown("### Profiel Opslaan")
        profile_name = st.text_input("Profielnaam", key="save_profile_name")
        
        if st.button("Configuratie Opslaan", key="save_config_btn"):
            if profile_name:
                success = config_manager.save_config(current_config, profile_name)
                if success:
                    st.success(f"Configuratie '{profile_name}' succesvol opgeslagen!")
                    # Ververs de lijst met profielen
                    config_profiles = config_manager.get_config_list()
                else:
                    st.error("Fout bij het opslaan van de configuratie.")
            else:
                st.warning("Voer een profielnaam in om de configuratie op te slaan.")
    
    with col2:
        # Profiel laden
        st.markdown("### Profiel Laden")
        
        if config_profiles:
            selected_profile = st.selectbox(
                "Selecteer een profiel",
                options=config_profiles,
                key="load_profile_select"
            )
            
            if st.button("Configuratie Laden", key="load_config_btn"):
                loaded_config = config_manager.load_config(selected_profile)
                st.session_state['current_config'] = loaded_config
                st.success(f"Configuratie '{selected_profile}' geladen! Ververs de pagina om de wijzigingen te zien.")
                st.rerun()
            
            if st.button("Profiel Verwijderen", key="delete_profile_btn"):
                if st.session_state.get('confirm_delete', False):
                    success = config_manager.delete_config(selected_profile)
                    if success:
                        st.success(f"Profiel '{selected_profile}' verwijderd!")
                        # Ververs de lijst met profielen
                        config_profiles = config_manager.get_config_list()
                        st.session_state['confirm_delete'] = False
                        st.rerun()
                    else:
                        st.error("Fout bij het verwijderen van het profiel.")
                else:
                    st.session_state['confirm_delete'] = True
                    st.warning(f"Weet u zeker dat u profiel '{selected_profile}' wilt verwijderen? Klik nogmaals om te bevestigen.")
        else:
            st.info("Geen opgeslagen profielen gevonden. Sla eerst een configuratie op.")
    
    # Exporteren en importeren
    st.subheader("Configuratie Exporteren/Importeren")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Exporteren
        st.markdown("### Exporteren")
        
        if config_profiles:
            export_profile = st.selectbox(
                "Selecteer een profiel om te exporteren",
                options=config_profiles,
                key="export_profile_select"
            )
            
            if st.button("Configuratie Exporteren", key="export_config_btn"):
                config_to_export = config_manager.load_config(export_profile)
                # Converteer naar JSON string
                json_str = json.dumps(config_to_export, indent=2)
                # Bied download aan
                st.download_button(
                    label="Download Configuratie",
                    data=json_str,
                    file_name=f"{export_profile}_config.json",
                    mime="application/json"
                )
        else:
            st.info("Geen opgeslagen profielen gevonden om te exporteren.")
    
    with col2:
        # Importeren
        st.markdown("### Importeren")
        
        uploaded_file = st.file_uploader("Upload een configuratiebestand", type=["json"])
        import_name = st.text_input("Profielnaam voor import (optioneel)", key="import_profile_name")
        
        if uploaded_file is not None:
            if st.button("Configuratie Importeren", key="import_config_btn"):
                try:
                    # Lees de geüploade configuratie
                    config_data = json.loads(uploaded_file.getvalue().decode("utf-8"))
                    
                    # Bepaal de profielnaam
                    profile_name = import_name if import_name else uploaded_file.name.split(".")[0]
                    
                    # Sla de configuratie op
                    success = config_manager.save_config(config_data, profile_name)
                    
                    if success:
                        st.success(f"Configuratie succesvol geïmporteerd als '{profile_name}'!")
                        # Ververs de lijst met profielen
                        config_profiles = config_manager.get_config_list()
                    else:
                        st.error("Fout bij het importeren van de configuratie.")
                except json.JSONDecodeError:
                    st.error("Ongeldig JSON-bestand. Controleer het bestand en probeer opnieuw.")
                except Exception as e:
                    st.error(f"Fout bij het importeren: {str(e)}")
    
    # Toon huidige configuratie als JSON
    with st.expander("Toon huidige configuratie als JSON"):
        st.json(current_config)