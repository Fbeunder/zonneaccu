"""Component voor de accu analyse pagina in de Streamlit interface."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List

from data_processing import data_loader, data_analysis
from storage_modules import battery_module


def render_battery_analysis_page():
    """Render de accu analyse pagina in de Streamlit interface."""
    st.subheader("Accu Analyse")
    
    # Controleer of er data is geladen
    if not st.session_state.get('data_loaded', False):
        st.warning("Laad eerst energiedata via de 'Data Import' pagina.")
        st.stop()
    
    # Haal de geladen data op
    data = st.session_state['energy_data']
    
    # Toon tabs voor verschillende soorten analyses
    tabs = st.tabs(["Accu Configuratie", "Simulatie Resultaten", "Energiestromen", 
                    "Financiële Analyse", "Optimalisatie"])
    
    # Tab 1: Accu Configuratie
    with tabs[0]:
        battery_config = render_battery_config_tab(data)
    
    # Tab 2: Simulatie Resultaten
    with tabs[1]:
        render_simulation_results_tab(data, battery_config)
    
    # Tab 3: Energiestromen
    with tabs[2]:
        render_energy_flows_tab(data, battery_config)
    
    # Tab 4: Financiële Analyse
    with tabs[3]:
        render_financial_analysis_tab(data, battery_config)
    
    # Tab 5: Optimalisatie
    with tabs[4]:
        render_optimization_tab(data)


def render_battery_config_tab(data: pd.DataFrame) -> Dict[str, Any]:
    """Render de accu configuratie tab en geef de configuratie terug.
    
    Args:
        data: DataFrame met energiedata
        
    Returns:
        Dict[str, Any]: Accu configuratie parameters
    """
    st.subheader("Accu Configuratie")
    
    # Uitleg
    st.write("""
    Configureer de parameters van de accu voor de simulatie. 
    De accu wordt gebruikt om overtollige zonne-energie op te slaan en later te gebruiken wanneer er een energietekort is.
    """)
    
    # Maak kolommen voor de configuratie
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Accu Eigenschappen")
        
        capacity = st.slider(
            "Accucapaciteit (kWh):",
            min_value=1.0,
            max_value=20.0,
            value=10.0,
            step=0.5,
            help="De totale energiecapaciteit van de accu in kilowattuur"
        )
        
        max_charge_rate = st.slider(
            "Maximale laadsnelheid (kW):",
            min_value=1.0,
            max_value=10.0,
            value=3.7,
            step=0.1,
            help="De maximale snelheid waarmee de accu kan worden opgeladen"
        )
        
        max_discharge_rate = st.slider(
            "Maximale ontlaadsnelheid (kW):",
            min_value=1.0,
            max_value=10.0,
            value=3.7,
            step=0.1,
            help="De maximale snelheid waarmee de accu kan worden ontladen"
        )
        
        charge_efficiency = st.slider(
            "Laadefficiëntie (%):",
            min_value=80,
            max_value=100,
            value=95,
            step=1,
            help="De efficiëntie waarmee elektrische energie wordt opgeslagen in de accu"
        ) / 100  # Converteer naar fractie
        
        discharge_efficiency = st.slider(
            "Ontlaadefficiëntie (%):",
            min_value=80,
            max_value=100,
            value=95,
            step=1,
            help="De efficiëntie waarmee opgeslagen energie wordt omgezet in elektrische energie"
        ) / 100  # Converteer naar fractie
    
    with col2:
        st.subheader("Operationele Parameters")
        
        min_soc = st.slider(
            "Minimale State of Charge (%):",
            min_value=0,
            max_value=50,
            value=10,
            step=5,
            help="De minimale laadtoestand van de accu die wordt aangehouden"
        ) / 100  # Converteer naar fractie
        
        max_soc = st.slider(
            "Maximale State of Charge (%):",
            min_value=50,
            max_value=100,
            value=90,
            step=5,
            help="De maximale laadtoestand van de accu die wordt aangehouden"
        ) / 100  # Converteer naar fractie
        
        self_discharge_rate = st.slider(
            "Zelfontlading per dag (%):",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            help="Het percentage energie dat de accu per dag verliest door zelfontlading"
        ) / 100  # Converteer naar fractie
        
        electricity_buy_price = st.number_input(
            "Elektriciteitsprijs inkoop (€/kWh):",
            min_value=0.1,
            max_value=0.5,
            value=0.30,
            step=0.01,
            help="De prijs die je betaalt voor elektriciteit van het net"
        )
        
        electricity_sell_price = st.number_input(
            "Elektriciteitsprijs verkoop (€/kWh):",
            min_value=0.0,
            max_value=0.5,
            value=0.15,
            step=0.01,
            help="De prijs die je ontvangt voor elektriciteit die je teruglevert aan het net"
        )
    
    # Besturingsstrategie selectie
    st.subheader("Besturingsstrategie")
    
    strategy_options = {
        "maximize_self_consumption": "Maximaliseer eigen verbruik",
        "time_of_use": "Tijd-gebaseerde optimalisatie",
        "peak_shaving": "Piekafvlakking"
    }
    
    selected_strategy = st.selectbox(
        "Selecteer een besturingsstrategie:",
        options=list(strategy_options.keys()),
        format_func=lambda x: strategy_options[x],
        index=0,  # Standaard op maximaliseer eigen verbruik
        help="De strategie die wordt gebruikt om te bepalen wanneer de accu wordt geladen en ontladen"
    )
    
    # Toon uitleg over de geselecteerde strategie
    if selected_strategy == "maximize_self_consumption":
        st.info("""
        **Maximaliseer eigen verbruik**: Deze strategie laadt de accu op wanneer er een energieoverschot is 
        en ontlaadt de accu wanneer er een energietekort is. Het doel is om zoveel mogelijk van je eigen 
        opgewekte energie te gebruiken en zo min mogelijk energie te kopen van het net.
        """)
    elif selected_strategy == "time_of_use":
        st.info("""
        **Tijd-gebaseerde optimalisatie**: Deze strategie laadt de accu op tijdens daluren (wanneer de 
        elektriciteitsprijs laag is) en ontlaadt de accu tijdens piekuren (wanneer de elektriciteitsprijs 
        hoog is). In deze simulatie worden daluren gedefinieerd als 00:00-07:00 en piekuren als 17:00-22:00.
        """)
    elif selected_strategy == "peak_shaving":
        st.info("""
        **Piekafvlakking**: Deze strategie probeert pieken in energieverbruik en -productie af te vlakken. 
        De accu wordt opgeladen wanneer de energieproductie boven een bepaalde drempel komt en ontladen 
        wanneer het energieverbruik boven een bepaalde drempel komt. Dit kan helpen om netcongestie te 
        verminderen en capaciteitstarieven te optimaliseren.
        """)
    
    # Degradatie parameters
    with st.expander("Degradatie Parameters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            cycle_degradation_rate = st.slider(
                "Degradatie per cyclus (%):",
                min_value=0.001,
                max_value=0.1,
                value=0.05,
                step=0.001,
                format="%.3f",
                help="Het percentage capaciteitsverlies per volledige laad/ontlaad-cyclus"
            ) / 100  # Converteer naar fractie
            
            calendar_degradation_rate = st.slider(
                "Degradatie per jaar (%):",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.5,
                help="Het percentage capaciteitsverlies per jaar, ongeacht gebruik"
            ) / 100  # Converteer naar fractie
        
        with col2:
            max_cycles = st.number_input(
                "Maximaal aantal cycli:",
                min_value=1000,
                max_value=10000,
                value=5000,
                step=500,
                help="Het maximale aantal volledige laad/ontlaad-cycli dat de accu kan doorstaan"
            )
    
    # Bereken en toon de opslagcapaciteit
    battery = battery_module.BatteryModel(
        capacity=capacity,
        max_charge_rate=max_charge_rate,
        max_discharge_rate=max_discharge_rate,
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency,
        self_discharge_rate=self_discharge_rate,
        min_soc=min_soc,
        max_soc=max_soc,
        cycle_degradation_rate=cycle_degradation_rate,
        calendar_degradation_rate=calendar_degradation_rate,
        max_cycles=max_cycles,
        electricity_buy_price=electricity_buy_price,
        electricity_sell_price=electricity_sell_price
    )
    
    usable_capacity = battery.usable_capacity
    storage_capacity = battery.calculate_storage_capacity()
    round_trip_efficiency = charge_efficiency * discharge_efficiency * 100
    
    # Toon informatie over de accu
    st.info(f"""
    **Accu Specificaties**:
    - **Totale capaciteit**: {capacity:.1f} kWh
    - **Bruikbare capaciteit**: {usable_capacity:.1f} kWh ({usable_capacity/capacity*100:.0f}% van totaal)
    - **Bruikbare opslagcapaciteit**: {storage_capacity/1000:.2f} kWh
    - **Round-trip efficiëntie**: {round_trip_efficiency:.1f}%
    - **Maximale laadtijd**: {usable_capacity/max_charge_rate:.1f} uur (van min naar max SoC)
    - **Maximale ontlaadtijd**: {usable_capacity/max_discharge_rate:.1f} uur (van max naar min SoC)
    """)
    
    # Sla de configuratie op in de sessie state
    battery_config = {
        'battery': battery,
        'control_strategy': selected_strategy
    }
    
    if 'battery_config' not in st.session_state:
        st.session_state['battery_config'] = battery_config
    else:
        st.session_state['battery_config'] = battery_config
    
    return battery_config


def render_simulation_results_tab(data: pd.DataFrame, battery_config: Dict[str, Any]):
    """Render de simulatie resultaten tab.
    
    Args:
        data: DataFrame met energiedata
        battery_config: Accu configuratie parameters
    """
    st.subheader("Simulatie Resultaten")
    
    # Haal de accu configuratie op
    battery = battery_config['battery']
    control_strategy = battery_config['control_strategy']
    
    # Voer de simulatie uit
    try:
        with st.spinner("Simulatie wordt uitgevoerd..."):
            simulation_results = battery_module.simulate_battery_storage(
                data=data,
                battery=battery,
                control_strategy=control_strategy
            )
        
        # Sla de resultaten op in de sessie state
        st.session_state['battery_simulation_results'] = simulation_results
        
        # Toon de belangrijkste resultaten
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Totaal Opgeslagen Energie",
                f"{simulation_results['total_energy_to_battery']/1000:.1f} kWh"
            )
            st.metric(
                "Zelfconsumptie",
                f"{simulation_results['new_self_consumption']:.1f}%",
                f"{simulation_results['self_consumption_increase']:.1f}%",
                help="Percentage van de opgewekte energie dat zelf wordt gebruikt (toename t.o.v. zonder accu)"
            )
        
        with col2:
            st.metric(
                "Totaal Gebruikte Energie",
                f"{simulation_results['total_energy_from_battery']/1000:.1f} kWh"
            )
            st.metric(
                "Round-trip Efficiëntie",
                f"{simulation_results['round_trip_efficiency']:.1f}%",
                help="Percentage van de opgeslagen energie dat nuttig is gebruikt"
            )
        
        with col3:
            st.metric(
                "Totaal Zelfontlading",
                f"{simulation_results['total_self_discharge']/1000:.2f} kWh"
            )
            st.metric(
                "Geschatte Levensduur",
                f"{simulation_results['remaining_years']:.1f} jaar" if simulation_results['remaining_years'] < 100 else "100+ jaar",
                help="Geschatte resterende levensduur op basis van het huidige gebruikspatroon"
            )
        
        # Toon State of Charge en energieinhoud plot
        st.subheader("Accu State of Charge en Energieinhoud")
        
        figures = battery_module.create_battery_simulation_plots(simulation_results)
        st.plotly_chart(figures['soc_energy'], use_container_width=True)
        
        # Toon dagelijks patroon
        st.subheader("Gemiddeld Dagelijks Patroon")
        st.plotly_chart(figures['daily_pattern'], use_container_width=True)
        
        # Toon optie om de simulatiedata te downloaden
        if st.checkbox("Toon simulatiedata"):
            st.dataframe(simulation_results['simulation_data'], use_container_width=True)
            
            csv = simulation_results['simulation_data'].to_csv(index=True)
            st.download_button(
                label="Download simulatiedata als CSV",
                data=csv,
                file_name="battery_simulation_data.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"Fout bij het uitvoeren van de simulatie: {str(e)}")


def render_energy_flows_tab(data: pd.DataFrame, battery_config: Dict[str, Any]):
    """Render de energiestromen tab.
    
    Args:
        data: DataFrame met energiedata
        battery_config: Accu configuratie parameters
    """
    st.subheader("Energiestromen")
    
    # Controleer of er simulatieresultaten zijn
    if 'battery_simulation_results' not in st.session_state:
        st.warning("Ga eerst naar de 'Simulatie Resultaten' tab om de simulatie uit te voeren.")
        return
    
    simulation_results = st.session_state['battery_simulation_results']
    
    # Maak en toon de plots
    figures = battery_module.create_battery_simulation_plots(simulation_results)
    
    # Energiestromen plot
    st.subheader("Energiestromen in de Accu")
    st.plotly_chart(figures['energy_flows'], use_container_width=True)
    
    # Grid interactie plot
    st.subheader("Interactie met Elektriciteitsnet")
    st.plotly_chart(figures['grid_interaction'], use_container_width=True)
    
    # Benutting van energieoverschot plot
    st.subheader("Benutting van Energieoverschot")
    st.plotly_chart(figures['surplus_utilization'], use_container_width=True)
    
    # Toon een samenvatting van de energiestromen
    st.subheader("Samenvatting Energiestromen")
    
    # Bereken percentages
    total_energy_in = simulation_results['total_energy_to_battery']
    total_energy_out = simulation_results['total_energy_from_battery']
    total_self_discharge = simulation_results['total_self_discharge']
    
    if total_energy_in > 0:
        useful_percentage = total_energy_out / total_energy_in * 100
        loss_percentage = total_self_discharge / total_energy_in * 100
    else:
        useful_percentage = 0
        loss_percentage = 0
    
    # Maak een Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["Energieoverschot", "Accu", "Eigen Verbruik", "Zelfontlading"],
            color=["green", "blue", "orange", "red"]
        ),
        link=dict(
            source=[0, 1, 1],  # indices verwijzen naar de nodes
            target=[1, 2, 3],
            value=[total_energy_in, total_energy_out, total_self_discharge],
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
            f"{total_energy_out/1000:.1f} kWh",
            f"{useful_percentage:.1f}% van input"
        )
    
    with col2:
        st.metric(
            "Zelfontlading",
            f"{total_self_discharge/1000:.1f} kWh",
            f"{loss_percentage:.1f}% van input"
        )
    
    # Toon vergelijking met en zonder accu
    st.subheader("Vergelijking Met en Zonder Accu")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Import Zonder Accu",
            f"€{simulation_results['original_import_cost']:.2f}"
        )
        st.metric(
            "Export Zonder Accu",
            f"€{simulation_results['original_export_revenue']:.2f}"
        )
        st.metric(
            "Netto Kosten Zonder Accu",
            f"€{simulation_results['original_net_cost']:.2f}"
        )
    
    with col2:
        st.metric(
            "Import Met Accu",
            f"€{simulation_results['new_import_cost']:.2f}",
            f"{(simulation_results['new_import_cost'] - simulation_results['original_import_cost'])/simulation_results['original_import_cost']*100:.1f}%" if simulation_results['original_import_cost'] > 0 else "N/A"
        )
        st.metric(
            "Export Met Accu",
            f"€{simulation_results['new_export_revenue']:.2f}",
            f"{(simulation_results['new_export_revenue'] - simulation_results['original_export_revenue'])/simulation_results['original_export_revenue']*100:.1f}%" if simulation_results['original_export_revenue'] > 0 else "N/A"
        )
        st.metric(
            "Netto Kosten Met Accu",
            f"€{simulation_results['new_net_cost']:.2f}",
            f"{(simulation_results['new_net_cost'] - simulation_results['original_net_cost'])/abs(simulation_results['original_net_cost'])*100:.1f}%" if simulation_results['original_net_cost'] != 0 else "N/A"
        )


def render_financial_analysis_tab(data: pd.DataFrame, battery_config: Dict[str, Any]):
    """Render de financiële analyse tab.
    
    Args:
        data: DataFrame met energiedata
        battery_config: Accu configuratie parameters
    """
    st.subheader("Financiële Analyse")
    
    # Controleer of er simulatieresultaten zijn
    if 'battery_simulation_results' not in st.session_state:
        st.warning("Ga eerst naar de 'Simulatie Resultaten' tab om de simulatie uit te voeren.")
        return
    
    simulation_results = st.session_state['battery_simulation_results']
    
    # Investeringskosten invoer
    st.subheader("Investeringskosten")
    
    col1, col2 = st.columns(2)
    
    with col1:
        battery_cost = st.number_input(
            "Aanschafkosten accu (€):",
            min_value=0,
            max_value=20000,
            value=int(500 * battery_config['battery'].capacity),  # Schatting op basis van capaciteit
            step=100
        )
        
        installation_cost = st.number_input(
            "Installatiekosten (€):",
            min_value=0,
            max_value=5000,
            value=1000,
            step=100
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
    
    # Vervangingskosten
    replacement_enabled = st.checkbox("Accu vervanging meenemen in analyse", value=True)
    
    if replacement_enabled:
        col1, col2 = st.columns(2)
        
        with col1:
            replacement_years = st.slider(
                "Vervanging na (jaren):",
                min_value=5,
                max_value=15,
                value=10,
                step=1
            )
        
        with col2:
            replacement_cost_factor = st.slider(
                "Vervangingskosten (% van origineel):",
                min_value=50,
                max_value=100,
                value=70,
                step=5
            ) / 100  # Converteer naar fractie
    else:
        replacement_years = None
        replacement_cost_factor = 0.7  # Standaardwaarde, wordt niet gebruikt
    
    # Bereken financiële analyse
    try:
        with st.spinner("Financiële analyse wordt berekend..."):
            financial_results = battery_module.calculate_payback_period(
                simulation_results=simulation_results,
                battery_cost=battery_cost,
                installation_cost=installation_cost,
                annual_maintenance=annual_maintenance,
                replacement_years=replacement_years if replacement_enabled else None,
                replacement_cost_factor=replacement_cost_factor,
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
            
            if financial_results['irr'] is not None:
                st.metric(
                    "Interne Rentevoet (IRR)",
                    f"{financial_results['irr']*100:.1f}%",
                    help="Het rendement op de investering"
                )
            else:
                st.metric(
                    "Interne Rentevoet (IRR)",
                    "Niet berekenbaar",
                    help="IRR kan niet worden berekend voor deze kasstromen"
                )
        
        # Toon de financiële analyse plot
        st.subheader("Financiële Analyse over Tijd")
        
        fig = battery_module.create_financial_analysis_plot(financial_results)
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
    st.subheader("Accu Optimalisatie")
    
    st.write("""
    Deze analyse helpt bij het bepalen van de optimale accugrootte en besturingsstrategie 
    voor uw specifieke situatie. Verschillende accucapaciteiten en besturingsstrategieën 
    worden geëvalueerd om de meest kosteneffectieve oplossing te vinden.
    """)
    
    # Configuratie voor de optimalisatie
    st.subheader("Optimalisatie Configuratie")
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_capacity = st.number_input(
            "Minimum accucapaciteit (kWh):",
            min_value=1.0,
            max_value=15.0,
            value=5.0,
            step=1.0
        )
        
        max_capacity = st.number_input(
            "Maximum accucapaciteit (kWh):",
            min_value=min_capacity + 1.0,
            max_value=20.0,
            value=15.0,
            step=1.0
        )
        
        step_capacity = st.number_input(
            "Capaciteit stapgrootte (kWh):",
            min_value=1.0,
            max_value=5.0,
            value=2.5,
            step=0.5
        )
    
    with col2:
        include_self_consumption = st.checkbox("Maximaliseer eigen verbruik", value=True)
        include_time_of_use = st.checkbox("Tijd-gebaseerde optimalisatie", value=True)
        include_peak_shaving = st.checkbox("Piekafvlakking", value=True)
    
    # Maak de capaciteit range
    capacity_range = []
    current_capacity = min_capacity
    while current_capacity <= max_capacity:
        capacity_range.append(current_capacity)
        current_capacity += step_capacity
    
    # Maak de besturingsstrategieën lijst
    control_strategies = []
    
    if include_self_consumption:
        control_strategies.append("maximize_self_consumption")
    
    if include_time_of_use:
        control_strategies.append("time_of_use")
    
    if include_peak_shaving:
        control_strategies.append("peak_shaving")
    
    if not control_strategies:
        st.warning("Selecteer ten minste één besturingsstrategie voor de optimalisatie.")
        return
    
    # Voer de optimalisatie uit
    if st.button("Start Optimalisatie"):
        try:
            with st.spinner("Optimalisatie wordt uitgevoerd... Dit kan even duren."):
                optimization_results = battery_module.optimize_battery_parameters(
                    data=data,
                    capacity_range=capacity_range,
                    control_strategies=control_strategies
                )
            
            # Toon de optimale configuratie
            st.subheader("Optimale Configuratie")
            
            optimal = optimization_results['optimal_config']
            
            strategy_names = {
                "maximize_self_consumption": "Maximaliseer eigen verbruik",
                "time_of_use": "Tijd-gebaseerde optimalisatie",
                "peak_shaving": "Piekafvlakking"
            }
            
            st.success(f"""
            **Optimale accugrootte**: {optimal['capacity']:.1f} kWh  
            **Besturingsstrategie**: {strategy_names[optimal['strategy']]}  
            **Jaarlijkse besparingen**: €{optimal['annual_savings']:.2f}  
            **Terugverdientijd**: {optimal['payback_period']:.1f} jaar  
            **Round-trip efficiëntie**: {optimal['round_trip_efficiency']:.1f}%  
            **Toename zelfconsumptie**: {optimal['self_consumption_increase']:.1f}%
            """)
            
            # Toon de optimalisatie plots
            st.subheader("Optimalisatie Resultaten")
            
            figures = battery_module.create_optimization_plots(optimization_results)
            
            st.plotly_chart(figures['payback_comparison'], use_container_width=True)
            st.plotly_chart(figures['savings_comparison'], use_container_width=True)
            st.plotly_chart(figures['npv_comparison'], use_container_width=True)
            st.plotly_chart(figures['efficiency_comparison'], use_container_width=True)
            
            # Toon de volledige resultaten tabel
            if st.checkbox("Toon alle resultaten"):
                # Vervang strategie codes door leesbare namen
                results_df = optimization_results['all_results'].copy()
                results_df['strategy'] = results_df['strategy'].map(strategy_names)
                
                st.dataframe(results_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Fout bij het uitvoeren van de optimalisatie: {str(e)}")