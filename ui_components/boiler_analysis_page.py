"""Component voor de warmwaterboiler analyse pagina in de Streamlit interface."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List

from data_processing import data_loader, data_analysis
from storage_modules import boiler_module


def render_boiler_analysis_page():
    """Render de warmwaterboiler analyse pagina in de Streamlit interface."""
    st.subheader("Warmwaterboiler Analyse")
    
    # Controleer of er data is geladen
    if not st.session_state.get('data_loaded', False):
        st.warning("Laad eerst energiedata via de 'Data Import' pagina.")
        st.stop()
    
    # Haal de geladen data op
    data = st.session_state['energy_data']
    
    # Toon tabs voor verschillende soorten analyses
    tabs = st.tabs(["Boiler Configuratie", "Simulatie Resultaten", "Energiestromen", 
                    "Financiële Analyse", "Optimalisatie"])
    
    # Tab 1: Boiler Configuratie
    with tabs[0]:
        boiler_config = render_boiler_config_tab(data)
    
    # Tab 2: Simulatie Resultaten
    with tabs[1]:
        render_simulation_results_tab(data, boiler_config)
    
    # Tab 3: Energiestromen
    with tabs[2]:
        render_energy_flows_tab(data, boiler_config)
    
    # Tab 4: Financiële Analyse
    with tabs[3]:
        render_financial_analysis_tab(data, boiler_config)
    
    # Tab 5: Optimalisatie
    with tabs[4]:
        render_optimization_tab(data)


def render_boiler_config_tab(data: pd.DataFrame) -> Dict[str, Any]:
    """Render de boiler configuratie tab en geef de configuratie terug.
    
    Args:
        data: DataFrame met energiedata
        
    Returns:
        Dict[str, Any]: Boiler configuratie parameters
    """
    st.subheader("Boiler Configuratie")
    
    # Uitleg
    st.write("""
    Configureer de parameters van de warmwaterboiler voor de simulatie. 
    De boiler wordt gebruikt om overtollige zonne-energie op te slaan in de vorm van warm water.
    """)
    
    # Maak kolommen voor de configuratie
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Boiler Eigenschappen")
        
        volume = st.slider(
            "Boilervolume (liter):",
            min_value=50,
            max_value=500,
            value=200,
            step=10,
            help="Het watervolume van de boiler in liters"
        )
        
        min_temp = st.slider(
            "Minimale temperatuur (°C):",
            min_value=30,
            max_value=60,
            value=40,
            step=1,
            help="De minimale temperatuur waarbij het water nog bruikbaar is"
        )
        
        max_temp = st.slider(
            "Maximale temperatuur (°C):",
            min_value=60,
            max_value=95,
            value=85,
            step=1,
            help="De maximale temperatuur tot waar het water verwarmd kan worden"
        )
        
        heating_power = st.slider(
            "Verwarmingsvermogen (W):",
            min_value=1000,
            max_value=5000,
            value=2000,
            step=100,
            help="Het elektrische vermogen van het verwarmingselement"
        )
        
        efficiency = st.slider(
            "Efficiëntie (%):",
            min_value=80,
            max_value=100,
            value=98,
            step=1,
            help="De efficiëntie waarmee elektrische energie wordt omgezet in warmte"
        ) / 100  # Converteer naar fractie
    
    with col2:
        st.subheader("Omgevings- en Economische Parameters")
        
        ambient_temp = st.slider(
            "Omgevingstemperatuur (°C):",
            min_value=10,
            max_value=30,
            value=20,
            step=1,
            help="De gemiddelde temperatuur van de ruimte waarin de boiler staat"
        )
        
        heat_loss_coefficient = st.slider(
            "Warmteverliescoëfficiënt (W/(m²·K)):",
            min_value=0.5,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Hoe snel de boiler warmte verliest aan de omgeving"
        )
        
        gas_price = st.number_input(
            "Gasprijs (€/m³):",
            min_value=0.5,
            max_value=3.0,
            value=1.20,
            step=0.01,
            help="De huidige gasprijs per kubieke meter"
        )
        
        electricity_price = st.number_input(
            "Elektriciteitsprijs (€/kWh):",
            min_value=0.1,
            max_value=0.5,
            value=0.30,
            step=0.01,
            help="De huidige elektriciteitsprijs per kilowattuur"
        )
        
        gas_efficiency = st.slider(
            "Gasketel efficiëntie (%):",
            min_value=70,
            max_value=98,
            value=90,
            step=1,
            help="De efficiëntie van een gasketel voor warmwaterbereiding"
        ) / 100  # Converteer naar fractie
    
    # Gebruiksprofiel configuratie
    st.subheader("Warmwatergebruik Profiel")
    
    # Voorgedefinieerde profielen
    profile_options = {
        "Laag gebruik": {7: 30, 19: 30},
        "Gemiddeld gebruik": {7: 50, 8: 20, 19: 40, 22: 20},
        "Hoog gebruik": {7: 60, 8: 30, 13: 20, 19: 50, 22: 40},
        "Aangepast": {}
    }
    
    selected_profile = st.selectbox(
        "Selecteer een gebruiksprofiel:",
        options=list(profile_options.keys()),
        index=1,  # Standaard op gemiddeld gebruik
        help="Voorgedefinieerde patronen van warmwatergebruik gedurende de dag"
    )
    
    usage_profile = profile_options[selected_profile].copy()
    
    # Als aangepast profiel is geselecteerd, toon invoervelden
    if selected_profile == "Aangepast":
        st.write("Configureer uw eigen warmwatergebruik per uur (in liters):")
        
        # Maak een grid van invoervelden voor elk uur
        hours_per_row = 6
        for i in range(0, 24, hours_per_row):
            cols = st.columns(hours_per_row)
            for j, col in enumerate(cols):
                hour = i + j
                if hour < 24:  # Controleer of we nog binnen 24 uur zijn
                    usage = col.number_input(
                        f"{hour}:00",
                        min_value=0,
                        max_value=100,
                        value=0,
                        step=5
                    )
                    if usage > 0:
                        usage_profile[hour] = usage
    else:
        # Toon het geselecteerde profiel
        if usage_profile:
            profile_df = pd.DataFrame({
                'Uur': list(usage_profile.keys()),
                'Watergebruik (L)': list(usage_profile.values())
            })
            
            fig = px.bar(
                profile_df,
                x='Uur',
                y='Watergebruik (L)',
                title=f"Warmwatergebruik Profiel: {selected_profile}",
                labels={'Uur': 'Uur van de dag', 'Watergebruik (L)': 'Watergebruik (liter)'},
                height=400
            )
            
            fig.update_layout(
                xaxis=dict(tickmode='linear', tick0=0, dtick=1),
                bargap=0.2
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Bereken en toon de opslagcapaciteit
    boiler = boiler_module.BoilerModel(
        volume=volume,
        min_temp=min_temp,
        max_temp=max_temp,
        ambient_temp=ambient_temp,
        heat_loss_coefficient=heat_loss_coefficient,
        heating_power=heating_power,
        efficiency=efficiency,
        gas_price=gas_price,
        electricity_price=electricity_price,
        gas_efficiency=gas_efficiency
    )
    
    storage_capacity = boiler.calculate_storage_capacity()
    
    st.info(f"""
    **Boiler Opslagcapaciteit**: {storage_capacity:.1f} Wh ({storage_capacity/1000:.2f} kWh)  
    Dit is de maximale hoeveelheid energie die in de boiler kan worden opgeslagen als het water wordt verwarmd van {min_temp}°C naar {max_temp}°C.
    """)
    
    # Sla de configuratie op in de sessie state
    boiler_config = {
        'boiler': boiler,
        'usage_profile': usage_profile
    }
    
    if 'boiler_config' not in st.session_state:
        st.session_state['boiler_config'] = boiler_config
    else:
        st.session_state['boiler_config'] = boiler_config
    
    return boiler_config


def render_simulation_results_tab(data: pd.DataFrame, boiler_config: Dict[str, Any]):
    """Render de simulatie resultaten tab.
    
    Args:
        data: DataFrame met energiedata
        boiler_config: Boiler configuratie parameters
    """
    st.subheader("Simulatie Resultaten")
    
    # Haal de boiler configuratie op
    boiler = boiler_config['boiler']
    usage_profile = boiler_config['usage_profile']
    
    # Voer de simulatie uit
    try:
        with st.spinner("Simulatie wordt uitgevoerd..."):
            simulation_results = boiler_module.simulate_boiler_storage(
                data=data,
                boiler=boiler,
                usage_profile=usage_profile
            )
        
        # Sla de resultaten op in de sessie state
        st.session_state['boiler_simulation_results'] = simulation_results
        
        # Toon de belangrijkste resultaten
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Totaal Opgeslagen Energie",
                f"{simulation_results['total_energy_to_boiler']/1000:.1f} kWh"
            )
            st.metric(
                "Benutting Energieoverschot",
                f"{simulation_results['surplus_utilization']:.1f}%",
                help="Percentage van het totale energieoverschot dat is benut voor de boiler"
            )
        
        with col2:
            st.metric(
                "Totaal Gebruikte Energie",
                f"{simulation_results['total_energy_from_boiler']/1000:.1f} kWh"
            )
            st.metric(
                "Totaal Warmteverlies",
                f"{simulation_results['total_heat_loss']/1000:.1f} kWh"
            )
        
        with col3:
            st.metric(
                "Boiler Efficiëntie",
                f"{simulation_results['boiler_efficiency']:.1f}%",
                help="Percentage van de opgeslagen energie dat nuttig is gebruikt"
            )
            st.metric(
                "Totaal Warmwatergebruik",
                f"{simulation_results['total_hot_water_usage']:.0f} L"
            )
        
        # Toon temperatuur en energieinhoud plot
        st.subheader("Boilertemperatuur en Energieinhoud")
        
        figures = boiler_module.create_boiler_simulation_plots(simulation_results)
        st.plotly_chart(figures['temperature_energy'], use_container_width=True)
        
        # Toon optie om de simulatiedata te downloaden
        if st.checkbox("Toon simulatiedata"):
            st.dataframe(simulation_results['simulation_data'], use_container_width=True)
            
            csv = simulation_results['simulation_data'].to_csv(index=True)
            st.download_button(
                label="Download simulatiedata als CSV",
                data=csv,
                file_name="boiler_simulation_data.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"Fout bij het uitvoeren van de simulatie: {str(e)}")


def render_energy_flows_tab(data: pd.DataFrame, boiler_config: Dict[str, Any]):
    """Render de energiestromen tab.
    
    Args:
        data: DataFrame met energiedata
        boiler_config: Boiler configuratie parameters
    """
    st.subheader("Energiestromen")
    
    # Controleer of er simulatieresultaten zijn
    if 'boiler_simulation_results' not in st.session_state:
        st.warning("Ga eerst naar de 'Simulatie Resultaten' tab om de simulatie uit te voeren.")
        return
    
    simulation_results = st.session_state['boiler_simulation_results']
    
    # Maak en toon de plots
    figures = boiler_module.create_boiler_simulation_plots(simulation_results)
    
    # Energiestromen plot
    st.subheader("Energiestromen in de Boiler")
    st.plotly_chart(figures['energy_flows'], use_container_width=True)
    
    # Warmwatergebruik plot
    st.subheader("Warmwatergebruik")
    st.plotly_chart(figures['water_usage'], use_container_width=True)
    
    # Benutting van energieoverschot plot
    st.subheader("Benutting van Energieoverschot")
    st.plotly_chart(figures['surplus_utilization'], use_container_width=True)
    
    # Toon een samenvatting van de energiestromen
    st.subheader("Samenvatting Energiestromen")
    
    # Bereken percentages
    total_energy_in = simulation_results['total_energy_to_boiler']
    total_energy_out = simulation_results['total_energy_from_boiler'] + simulation_results['total_heat_loss']
    
    if total_energy_in > 0:
        useful_percentage = simulation_results['total_energy_from_boiler'] / total_energy_in * 100
        loss_percentage = simulation_results['total_heat_loss'] / total_energy_in * 100
    else:
        useful_percentage = 0
        loss_percentage = 0
    
    # Maak een Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["Energieoverschot", "Boiler", "Nuttig Gebruikt", "Warmteverlies"],
            color=["green", "blue", "orange", "red"]
        ),
        link=dict(
            source=[0, 1, 1],  # indices verwijzen naar de nodes
            target=[1, 2, 3],
            value=[total_energy_in, simulation_results['total_energy_from_boiler'], simulation_results['total_heat_loss']],
            color=["rgba(0,255,0,0.4)", "rgba(255,165,0,0.4)", "rgba(255,0,0,0.4)"]
        )
    )])
    
    fig.update_layout(
        title_text="Energiestromen Sankey Diagram",
        font_size=12
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Toon percentages
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Nuttig Gebruikte Energie",
            f"{simulation_results['total_energy_from_boiler']/1000:.1f} kWh",
            f"{useful_percentage:.1f}% van input"
        )
    
    with col2:
        st.metric(
            "Warmteverlies",
            f"{simulation_results['total_heat_loss']/1000:.1f} kWh",
            f"{loss_percentage:.1f}% van input"
        )


def render_financial_analysis_tab(data: pd.DataFrame, boiler_config: Dict[str, Any]):
    """Render de financiële analyse tab.
    
    Args:
        data: DataFrame met energiedata
        boiler_config: Boiler configuratie parameters
    """
    st.subheader("Financiële Analyse")
    
    # Controleer of er simulatieresultaten zijn
    if 'boiler_simulation_results' not in st.session_state:
        st.warning("Ga eerst naar de 'Simulatie Resultaten' tab om de simulatie uit te voeren.")
        return
    
    simulation_results = st.session_state['boiler_simulation_results']
    
    # Investeringskosten invoer
    st.subheader("Investeringskosten")
    
    col1, col2 = st.columns(2)
    
    with col1:
        boiler_cost = st.number_input(
            "Aanschafkosten boiler (€):",
            min_value=0,
            max_value=5000,
            value=int(500 + 2 * boiler_config['boiler'].volume),  # Schatting op basis van volume
            step=50
        )
        
        installation_cost = st.number_input(
            "Installatiekosten (€):",
            min_value=0,
            max_value=2000,
            value=300,
            step=50
        )
    
    with col2:
        annual_maintenance = st.number_input(
            "Jaarlijkse onderhoudskosten (€):",
            min_value=0,
            max_value=500,
            value=50,
            step=10
        )
        
        analysis_years = st.slider(
            "Aantal jaren voor analyse:",
            min_value=5,
            max_value=25,
            value=15,
            step=1
        )
    
    # Bereken financiële analyse
    try:
        with st.spinner("Financiële analyse wordt berekend..."):
            financial_results = boiler_module.calculate_payback_period(
                simulation_results=simulation_results,
                boiler_cost=boiler_cost,
                installation_cost=installation_cost,
                annual_maintenance=annual_maintenance,
                years=analysis_years
            )
        
        # Toon de belangrijkste financiële resultaten
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Jaarlijkse Besparingen",
                f"€{financial_results['annual_savings']:.2f}"
            )
            st.metric(
                "Totale Investering",
                f"€{financial_results['total_investment']:.2f}"
            )
        
        with col2:
            if financial_results['payback_period'] == float('inf'):
                st.metric(
                    "Terugverdientijd",
                    "Nooit",
                    help="De investering wordt nooit terugverdiend omdat de jaarlijkse besparingen lager zijn dan de onderhoudskosten"
                )
            else:
                st.metric(
                    "Terugverdientijd",
                    f"{financial_results['payback_period']:.1f} jaar"
                )
            
            st.metric(
                "Return on Investment (ROI)",
                f"{financial_results['roi']:.1f}%",
                help=f"ROI na {analysis_years} jaar"
            )
        
        with col3:
            st.metric(
                "Netto Contante Waarde (NCW)",
                f"€{financial_results['npv']:.2f}",
                help="NCW met een discontovoet van 3%"
            )
            st.metric(
                "Simulatieperiode",
                f"{financial_results['simulation_days']:.1f} dagen"
            )
        
        # Toon de financiële analyse plot
        st.subheader("Financiële Analyse over Tijd")
        
        fig = boiler_module.create_financial_analysis_plot(financial_results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Toon de kasstromen tabel
        if st.checkbox("Toon kasstromen tabel"):
            cash_flows_df = pd.DataFrame(financial_results['cash_flows'])
            st.dataframe(cash_flows_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Fout bij het uitvoeren van de financiële analyse: {str(e)}")


def render_optimization_tab(data: pd.DataFrame):
    """Render de optimalisatie tab.
    
    Args:
        data: DataFrame met energiedata
    """
    st.subheader("Boiler Optimalisatie")
    
    st.write("""
    Deze analyse helpt bij het bepalen van de optimale boilergrootte en configuratie 
    voor uw specifieke situatie. Verschillende boilervolumes en gebruiksprofielen 
    worden geëvalueerd om de meest kosteneffectieve oplossing te vinden.
    """)
    
    # Configuratie voor de optimalisatie
    st.subheader("Optimalisatie Configuratie")
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_volume = st.number_input(
            "Minimum boilervolume (L):",
            min_value=50,
            max_value=400,
            value=100,
            step=10
        )
        
        max_volume = st.number_input(
            "Maximum boilervolume (L):",
            min_value=min_volume + 50,
            max_value=500,
            value=300,
            step=10
        )
        
        step_volume = st.number_input(
            "Volume stapgrootte (L):",
            min_value=10,
            max_value=100,
            value=50,
            step=10
        )
    
    with col2:
        include_low_usage = st.checkbox("Laag gebruik profiel", value=True)
        include_medium_usage = st.checkbox("Gemiddeld gebruik profiel", value=True)
        include_high_usage = st.checkbox("Hoog gebruik profiel", value=True)
    
    # Maak de volume range
    volume_range = list(range(min_volume, max_volume + 1, step_volume))
    
    # Maak de gebruiksprofielen dictionary
    usage_profiles = {}
    
    if include_low_usage:
        usage_profiles["Laag gebruik"] = {7: 30, 19: 30}
    
    if include_medium_usage:
        usage_profiles["Gemiddeld gebruik"] = {7: 50, 8: 20, 19: 40, 22: 20}
    
    if include_high_usage:
        usage_profiles["Hoog gebruik"] = {7: 60, 8: 30, 13: 20, 19: 50, 22: 40}
    
    if not usage_profiles:
        st.warning("Selecteer ten minste één gebruiksprofiel voor de optimalisatie.")
        return
    
    # Voer de optimalisatie uit
    if st.button("Start Optimalisatie"):
        try:
            with st.spinner("Optimalisatie wordt uitgevoerd... Dit kan even duren."):
                optimization_results = boiler_module.optimize_boiler_parameters(
                    data=data,
                    volume_range=volume_range,
                    usage_profiles=usage_profiles
                )
            
            # Toon de optimale configuratie
            st.subheader("Optimale Configuratie")
            
            optimal = optimization_results['optimal_config']
            
            st.success(f"""
            **Optimale boilergrootte**: {optimal['volume']:.0f} liter  
            **Gebruiksprofiel**: {optimal['profile_name']}  
            **Jaarlijkse besparingen**: €{optimal['annual_savings']:.2f}  
            **Terugverdientijd**: {optimal['payback_period']:.1f} jaar  
            **Boilerefficiëntie**: {optimal['boiler_efficiency']:.1f}%  
            **Benutting energieoverschot**: {optimal['surplus_utilization']:.1f}%
            """)
            
            # Toon de optimalisatie plots
            st.subheader("Optimalisatie Resultaten")
            
            figures = boiler_module.create_optimization_plots(optimization_results)
            
            st.plotly_chart(figures['payback_comparison'], use_container_width=True)
            st.plotly_chart(figures['savings_comparison'], use_container_width=True)
            st.plotly_chart(figures['efficiency_comparison'], use_container_width=True)
            
            # Toon de volledige resultaten tabel
            if st.checkbox("Toon alle resultaten"):
                st.dataframe(optimization_results['all_results'], use_container_width=True)
        
        except Exception as e:
            st.error(f"Fout bij het uitvoeren van de optimalisatie: {str(e)}")