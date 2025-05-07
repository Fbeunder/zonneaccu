"""Component voor de data analyse pagina in de Streamlit interface."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List

from data_processing import data_loader, data_analysis


def render_data_analysis_page():
    """Render de data analyse pagina in de Streamlit interface."""
    st.subheader("Data Analyse")
    
    # Controleer of er data is geladen
    if not st.session_state.get('data_loaded', False):
        st.warning("Laad eerst energiedata via de 'Data Import' pagina.")
        st.stop()
    
    # Haal de geladen data op
    data = st.session_state['energy_data']
    
    # Toon tabs voor verschillende soorten analyses
    tabs = st.tabs(["Tijdreeks Visualisatie", "Dagelijkse Patronen", "Wekelijkse Patronen", 
                    "Seizoenspatronen", "Energiebalans", "Opslagpotentieel", "Outlier Analyse"])
    
    # Tab 1: Tijdreeks Visualisatie
    with tabs[0]:
        render_time_series_tab(data)
    
    # Tab 2: Dagelijkse Patronen
    with tabs[1]:
        render_daily_patterns_tab(data)
    
    # Tab 3: Wekelijkse Patronen
    with tabs[2]:
        render_weekly_patterns_tab(data)
    
    # Tab 4: Seizoenspatronen
    with tabs[3]:
        render_seasonal_patterns_tab(data)
    
    # Tab 5: Energiebalans
    with tabs[4]:
        render_energy_balance_tab(data)
    
    # Tab 6: Opslagpotentieel
    with tabs[5]:
        render_storage_potential_tab(data)
    
    # Tab 7: Outlier Analyse
    with tabs[6]:
        render_outlier_analysis_tab(data)


def render_time_series_tab(data: pd.DataFrame):
    """Render de tijdreeks visualisatie tab.
    
    Args:
        data (pd.DataFrame): De energiedata.
    """
    st.subheader("Tijdreeks Visualisatie")
    
    # Selecteer kolommen om te visualiseren
    available_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    default_columns = [col for col in ['Energy Produced (Wh)', 'Energy Consumed (Wh)', 'Net Energy (Wh)'] 
                      if col in available_columns]
    
    selected_columns = st.multiselect(
        "Selecteer kolommen om te visualiseren:",
        options=available_columns,
        default=default_columns
    )
    
    if not selected_columns:
        st.warning("Selecteer ten minste één kolom om te visualiseren.")
        return
    
    # Optie voor resampling
    resample_options = {
        "Geen resampling": None,
        "Uurlijks": "H",
        "Dagelijks": "D",
        "Wekelijks": "W",
        "Maandelijks": "M"
    }
    
    resample_interval = st.selectbox(
        "Resample data voor visualisatie:",
        options=list(resample_options.keys()),
        index=0
    )
    
    # Creeër en toon de tijdreeksplot
    try:
        fig = data_analysis.create_time_series_plot(
            data=data,
            columns=selected_columns,
            title="Energiedata Tijdreeks",
            resample=resample_options[resample_interval]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Optie om de data te downloaden
        if st.checkbox("Toon data"):
            display_df = data[selected_columns].copy()
            if resample_options[resample_interval] is not None:
                display_df = display_df.resample(resample_options[resample_interval]).mean()
            
            # Reset de index voor weergave
            display_df = display_df.reset_index()
            st.dataframe(display_df, use_container_width=True)
            
    except Exception as e:
        st.error(f"Fout bij het maken van de tijdreeksplot: {str(e)}")


def render_daily_patterns_tab(data: pd.DataFrame):
    """Render de dagelijkse patronen analyse tab.
    
    Args:
        data (pd.DataFrame): De energiedata.
    """
    st.subheader("Dagelijkse Patronen Analyse")
    
    try:
        # Analyseer dagelijkse patronen
        daily_patterns = data_analysis.identify_daily_patterns(data)
        
        # Toon dagelijks profiel plot
        st.subheader("Gemiddeld Dagelijks Profiel")
        
        profile_group_by = st.radio(
            "Groepeer per:",
            options=["Alle data", "Weekdag vs. Weekend", "Seizoen"],
            horizontal=True
        )
        
        group_by_map = {
            "Alle data": "all",
            "Weekdag vs. Weekend": "weekday_weekend",
            "Seizoen": "season"
        }
        
        try:
            fig = data_analysis.create_daily_profile_plot(
                data=data,
                group_by=group_by_map[profile_group_by]
            )
            st.plotly_chart(fig, use_container_width=True)
        except ValueError as e:
            if "seizoensanalyse" in str(e).lower():
                st.warning("Onvoldoende data voor seizoensanalyse (minimaal 2 maanden nodig).")
            else:
                st.error(f"Fout bij het maken van het dagelijkse profiel: {str(e)}")
        
        # Toon energiebalans heatmap
        st.subheader("Energiebalans Heatmap")
        st.write("Deze heatmap toont de netto energiebalans per uur van de dag en per datum.")
        
        try:
            heatmap_fig = data_analysis.create_energy_balance_heatmap(data)
            st.plotly_chart(heatmap_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Fout bij het maken van de heatmap: {str(e)}")
        
        # Toon belangrijke inzichten
        st.subheader("Belangrijke Inzichten")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Piek Productie Uur", f"{daily_patterns['production_peak_hour']}:00")
            st.metric("Totaal Dagelijks Overschot", f"{daily_patterns['total_daily_surplus']:.1f} Wh")
            st.metric("Uren met Energieoverschot", len(daily_patterns['surplus_hours']))
        
        with col2:
            st.metric("Piek Verbruik Uur", f"{daily_patterns['consumption_peak_hour']}:00")
            st.metric("Totaal Dagelijks Tekort", f"{daily_patterns['total_daily_deficit']:.1f} Wh")
            st.metric("Uren met Energietekort", len(daily_patterns['deficit_hours']))
        
        # Toon opslagpotentieel
        st.metric("Dagelijks Opslagpotentieel", f"{daily_patterns['storage_potential']:.1f} Wh", 
                 help="De hoeveelheid energie die theoretisch opgeslagen en later gebruikt kan worden")
        
        # Toon de uurlijkse patronen data
        if st.checkbox("Toon uurlijkse patronen data"):
            hourly_patterns = daily_patterns['hourly_patterns'].reset_index()
            st.dataframe(hourly_patterns, use_container_width=True)
        
    except Exception as e:
        st.error(f"Fout bij het analyseren van dagelijkse patronen: {str(e)}")


def render_weekly_patterns_tab(data: pd.DataFrame):
    """Render de wekelijkse patronen analyse tab.
    
    Args:
        data (pd.DataFrame): De energiedata.
    """
    st.subheader("Wekelijkse Patronen Analyse")
    
    try:
        # Analyseer wekelijkse patronen
        weekly_patterns = data_analysis.identify_weekly_patterns(data)
        
        # Toon wekelijks profiel
        st.subheader("Gemiddeld Wekelijks Profiel")
        
        # Converteer de daily_patterns DataFrame naar een formaat geschikt voor plotting
        daily_df = weekly_patterns['daily_patterns'].reset_index()
        daily_df['day_name'] = daily_df['day_of_week'].map({
            0: "Maandag", 1: "Dinsdag", 2: "Woensdag", 3: "Donderdag", 
            4: "Vrijdag", 5: "Zaterdag", 6: "Zondag"
        })
        
        # Creeër een bar chart voor productie en verbruik per dag
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=daily_df['day_name'],
            y=daily_df['Energy Produced (Wh)'],
            name='Productie',
            marker_color='rgba(55, 83, 109, 0.7)'
        ))
        
        fig.add_trace(go.Bar(
            x=daily_df['day_name'],
            y=daily_df['Energy Consumed (Wh)'],
            name='Verbruik',
            marker_color='rgba(219, 64, 82, 0.7)'
        ))
        
        fig.update_layout(
            title="Gemiddelde Energieproductie en -verbruik per Dag van de Week",
            xaxis_title="Dag van de Week",
            yaxis_title="Energie (Wh)",
            barmode='group',
            xaxis={'categoryorder':'array', 'categoryarray':[
                "Maandag", "Dinsdag", "Woensdag", "Donderdag", "Vrijdag", "Zaterdag", "Zondag"
            ]}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Toon weekdag vs. weekend vergelijking
        st.subheader("Weekdag vs. Weekend Vergelijking")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Productieverschil Weekend vs. Weekdag", 
                f"{weekly_patterns['weekday_weekend_diff']['production']:.1f} Wh",
                delta=f"{(weekly_patterns['weekday_weekend_diff']['production'] / weekly_patterns['weekday_avg']['Energy Produced (Wh)'] * 100):.1f}%" 
                if weekly_patterns['weekday_avg']['Energy Produced (Wh)'] > 0 else None
            )
        
        with col2:
            st.metric(
                "Verbruiksverschil Weekend vs. Weekdag", 
                f"{weekly_patterns['weekday_weekend_diff']['consumption']:.1f} Wh",
                delta=f"{(weekly_patterns['weekday_weekend_diff']['consumption'] / weekly_patterns['weekday_avg']['Energy Consumed (Wh)'] * 100):.1f}%"
            )
        
        # Toon belangrijke inzichten
        st.subheader("Belangrijke Inzichten")
        
        col1, col2 = st.columns(2)
        
        with col1:
            peak_production_day = daily_df.loc[weekly_patterns['production_peak_day'], 'day_name']
            st.metric("Dag met Hoogste Productie", peak_production_day)
        
        with col2:
            peak_consumption_day = daily_df.loc[weekly_patterns['consumption_peak_day'], 'day_name']
            st.metric("Dag met Hoogste Verbruik", peak_consumption_day)
        
        # Toon de dagelijkse patronen data
        if st.checkbox("Toon dagelijkse patronen data"):
            st.dataframe(daily_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Fout bij het analyseren van wekelijkse patronen: {str(e)}")


def render_seasonal_patterns_tab(data: pd.DataFrame):
    """Render de seizoenspatronen analyse tab.
    
    Args:
        data (pd.DataFrame): De energiedata.
    """
    st.subheader("Seizoenspatronen Analyse")
    
    try:
        # Analyseer seizoenspatronen
        seasonal_patterns = data_analysis.identify_seasonal_patterns(data)
        
        if seasonal_patterns is None:
            st.warning("Onvoldoende data voor seizoensanalyse. Er zijn minimaal 2 maanden aan data nodig.")
            return
        
        # Toon maandelijks profiel
        st.subheader("Gemiddeld Maandelijks Profiel")
        
        # Converteer de monthly_patterns DataFrame naar een formaat geschikt voor plotting
        monthly_df = seasonal_patterns['monthly_patterns'].reset_index()
        monthly_df['month_name'] = monthly_df['month'].map({
            1: "Jan", 2: "Feb", 3: "Mrt", 4: "Apr", 5: "Mei", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Dec"
        })
        
        # Creeër een bar chart voor productie en verbruik per maand
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=monthly_df['month_name'],
            y=monthly_df['Energy Produced (Wh)'],
            name='Productie',
            marker_color='rgba(55, 83, 109, 0.7)'
        ))
        
        fig.add_trace(go.Bar(
            x=monthly_df['month_name'],
            y=monthly_df['Energy Consumed (Wh)'],
            name='Verbruik',
            marker_color='rgba(219, 64, 82, 0.7)'
        ))
        
        fig.update_layout(
            title="Gemiddelde Energieproductie en -verbruik per Maand",
            xaxis_title="Maand",
            yaxis_title="Energie (Wh)",
            barmode='group',
            xaxis={'categoryorder':'array', 'categoryarray':[
                "Jan", "Feb", "Mrt", "Apr", "Mei", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dec"
            ]}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Toon seizoensprofiel als er voldoende data is
        if len(seasonal_patterns['seasonal_patterns']) > 1:
            st.subheader("Gemiddeld Seizoensprofiel")
            
            # Converteer de seasonal_patterns DataFrame naar een formaat geschikt voor plotting
            seasonal_df = seasonal_patterns['seasonal_patterns'].reset_index()
            
            # Creeër een bar chart voor productie en verbruik per seizoen
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=seasonal_df['season'],
                y=seasonal_df['Energy Produced (Wh)'],
                name='Productie',
                marker_color='rgba(55, 83, 109, 0.7)'
            ))
            
            fig.add_trace(go.Bar(
                x=seasonal_df['season'],
                y=seasonal_df['Energy Consumed (Wh)'],
                name='Verbruik',
                marker_color='rgba(219, 64, 82, 0.7)'
            ))
            
            fig.update_layout(
                title="Gemiddelde Energieproductie en -verbruik per Seizoen",
                xaxis_title="Seizoen",
                yaxis_title="Energie (Wh)",
                barmode='group',
                xaxis={'categoryorder':'array', 'categoryarray':["Winter", "Spring", "Summer", "Fall"]}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Toon belangrijke inzichten
            st.subheader("Belangrijke Inzichten")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if seasonal_patterns['production_peak_season']:
                    st.metric("Seizoen met Hoogste Productie", seasonal_patterns['production_peak_season'])
            
            with col2:
                if seasonal_patterns['consumption_peak_season']:
                    st.metric("Seizoen met Hoogste Verbruik", seasonal_patterns['consumption_peak_season'])
        
        # Toon de maandelijkse patronen data
        if st.checkbox("Toon maandelijkse patronen data"):
            st.dataframe(monthly_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Fout bij het analyseren van seizoenspatronen: {str(e)}")


def render_energy_balance_tab(data: pd.DataFrame):
    """Render de energiebalans analyse tab.
    
    Args:
        data (pd.DataFrame): De energiedata.
    """
    st.subheader("Energiebalans Analyse")
    
    try:
        # Analyseer energiebalans
        balance_analysis = data_analysis.analyze_energy_balance(data)
        
        # Toon overzicht van energiebalans
        st.subheader("Energiebalans Overzicht")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Totaal Energieoverschot", 
                f"{balance_analysis['total_surplus'] / 1000:.1f} kWh"
            )
            st.metric(
                "Percentage Tijd met Overschot", 
                f"{balance_analysis['surplus_percentage']:.1f}%"
            )
        
        with col2:
            st.metric(
                "Totaal Energietekort", 
                f"{balance_analysis['total_deficit'] / 1000:.1f} kWh"
            )
            st.metric(
                "Percentage Tijd met Tekort", 
                f"{balance_analysis['deficit_percentage']:.1f}%"
            )
        
        with col3:
            st.metric(
                "Maximale Opslagcapaciteit Nodig", 
                f"{balance_analysis['max_storage_needed'] / 1000:.1f} kWh",
                help="De maximale opslagcapaciteit die nodig zou zijn om alle overschotten op te slaan en later te gebruiken"
            )
        
        # Toon zelfvoorzienendheid analyse
        st.subheader("Zelfvoorzienendheid Analyse")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Huidige Zelfvoorzienendheid", 
                f"{balance_analysis['current_self_sufficiency']:.1f}%",
                help="Percentage van het verbruik dat momenteel wordt gedekt door eigen productie"
            )
        
        with col2:
            st.metric(
                "Theoretisch Maximum Zelfvoorzienendheid", 
                f"{balance_analysis['theoretical_max_self_sufficiency']:.1f}%",
                delta=f"+{balance_analysis['storage_improvement_potential']:.1f}%",
                help="Maximaal haalbare zelfvoorzienendheid met perfecte energieopslag"
            )
        
        # Visualiseer de cumulatieve energiebalans
        st.subheader("Cumulatieve Energiebalans")
        st.write("Deze grafiek toont de cumulatieve som van de netto energie over tijd, wat inzicht geeft in de benodigde opslagcapaciteit.")
        
        # Bereken cumulatieve energiebalans
        cumulative_energy = data['Net Energy (Wh)'].cumsum()
        
        # Creeër de figuur
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=cumulative_energy.index,
            y=cumulative_energy,
            mode='lines',
            name='Cumulatieve Energiebalans',
            line=dict(color='royalblue')
        ))
        
        # Voeg horizontale lijnen toe voor minimum en maximum
        fig.add_shape(
            type="line",
            x0=cumulative_energy.index[0],
            y0=cumulative_energy.min(),
            x1=cumulative_energy.index[-1],
            y1=cumulative_energy.min(),
            line=dict(color="red", width=2, dash="dash"),
        )
        
        fig.add_shape(
            type="line",
            x0=cumulative_energy.index[0],
            y0=cumulative_energy.max(),
            x1=cumulative_energy.index[-1],
            y1=cumulative_energy.max(),
            line=dict(color="green", width=2, dash="dash"),
        )
        
        # Voeg annotaties toe
        fig.add_annotation(
            x=cumulative_energy.idxmin(),
            y=cumulative_energy.min(),
            text=f"Min: {cumulative_energy.min():.1f} Wh",
            showarrow=True,
            arrowhead=1
        )
        
        fig.add_annotation(
            x=cumulative_energy.idxmax(),
            y=cumulative_energy.max(),
            text=f"Max: {cumulative_energy.max():.1f} Wh",
            showarrow=True,
            arrowhead=1
        )
        
        fig.update_layout(
            title="Cumulatieve Energiebalans",
            xaxis_title="Datum/Tijd",
            yaxis_title="Cumulatieve Energie (Wh)",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Fout bij het analyseren van de energiebalans: {str(e)}")


def render_storage_potential_tab(data: pd.DataFrame):
    """Render de opslagpotentieel analyse tab.
    
    Args:
        data (pd.DataFrame): De energiedata.
    """
    st.subheader("Opslagpotentieel Analyse")
    
    try:
        # Laat de gebruiker een opslagcapaciteit kiezen
        max_theoretical_capacity = data_analysis.analyze_energy_balance(data)['max_storage_needed']
        
        st.write(f"De maximale theoretische opslagcapaciteit die nodig zou zijn is {max_theoretical_capacity/1000:.1f} kWh.")
        
        storage_capacity = st.slider(
            "Selecteer opslagcapaciteit (kWh):",
            min_value=0.5,
            max_value=max(20.0, max_theoretical_capacity/1000 * 1.1),  # Iets hoger dan theoretisch maximum
            value=min(5.0, max_theoretical_capacity/1000),  # Default waarde
            step=0.5
        )
        
        # Converteer van kWh naar Wh voor de berekeningen
        storage_capacity_wh = storage_capacity * 1000
        
        # Bereken opslagpotentieel
        storage_results = data_analysis.calculate_storage_potential(data, capacity=storage_capacity_wh)
        
        # Toon opslagresultaten
        st.subheader("Opslagresultaten")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Totaal Opgeslagen Energie", 
                f"{storage_results['total_energy_stored'] / 1000:.1f} kWh"
            )
            st.metric(
                "Totaal Gebruikte Energie uit Opslag", 
                f"{storage_results['total_energy_used_from_storage'] / 1000:.1f} kWh"
            )
        
        with col2:
            st.metric(
                "Opslagefficiëntie", 
                f"{storage_results['storage_efficiency']:.1f}%",
                help="Percentage van de opgeslagen energie dat daadwerkelijk wordt gebruikt"
            )
            st.metric(
                "Maximaal Bereikte Opslaglevel", 
                f"{storage_results['max_storage_level'] / 1000:.1f} kWh",
                delta=f"{storage_results['max_storage_level'] / storage_capacity_wh * 100:.1f}% van capaciteit"
            )
        
        # Toon zelfvoorzienendheid verbetering
        st.subheader("Zelfvoorzienendheid Verbetering")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Oorspronkelijke Zelfvoorzienendheid", 
                f"{storage_results['original_self_sufficiency']:.1f}%"
            )
        
        with col2:
            st.metric(
                "Verbeterde Zelfvoorzienendheid", 
                f"{storage_results['improved_self_sufficiency']:.1f}%",
                delta=f"+{storage_results['self_sufficiency_improvement']:.1f}%"
            )
        
        # Visualiseer het opslaglevel over tijd
        st.subheader("Opslaglevel over Tijd")
        
        # Creeër de figuur
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=storage_results['simulation_data'].index,
            y=storage_results['simulation_data']['Storage Level (Wh)'],
            mode='lines',
            name='Opslaglevel',
            line=dict(color='royalblue')
        ))
        
        # Voeg een horizontale lijn toe voor de maximale capaciteit
        fig.add_shape(
            type="line",
            x0=storage_results['simulation_data'].index[0],
            y0=storage_capacity_wh,
            x1=storage_results['simulation_data'].index[-1],
            y1=storage_capacity_wh,
            line=dict(color="red", width=2, dash="dash"),
        )
        
        fig.add_annotation(
            x=storage_results['simulation_data'].index[0],
            y=storage_capacity_wh,
            text=f"Max Capaciteit: {storage_capacity} kWh",
            showarrow=False,
            yshift=10
        )
        
        fig.update_layout(
            title="Opslaglevel over Tijd",
            xaxis_title="Datum/Tijd",
            yaxis_title="Opslaglevel (Wh)",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Toon overflow analyse
        if storage_results['total_storage_overflow'] > 0:
            st.subheader("Overflow Analyse")
            st.write(f"Er was {storage_results['total_storage_overflow']/1000:.1f} kWh aan energie die niet kon worden opgeslagen vanwege capaciteitsbeperkingen.")
            
            # Visualiseer de overflow
            overflow_fig = go.Figure()
            
            overflow_fig.add_trace(go.Scatter(
                x=storage_results['simulation_data'].index,
                y=storage_results['simulation_data']['Storage Overflow (Wh)'],
                mode='lines',
                name='Overflow',
                line=dict(color='orange')
            ))
            
            overflow_fig.update_layout(
                title="Energie Overflow (Niet Opgeslagen)",
                xaxis_title="Datum/Tijd",
                yaxis_title="Overflow (Wh)",
                hovermode="x unified"
            )
            
            st.plotly_chart(overflow_fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Fout bij het analyseren van het opslagpotentieel: {str(e)}")


def render_outlier_analysis_tab(data: pd.DataFrame):
    """Render de outlier analyse tab.
    
    Args:
        data (pd.DataFrame): De energiedata.
    """
    st.subheader("Outlier Analyse")
    
    try:
        # Selecteer kolommen voor outlier detectie
        available_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        default_columns = [col for col in ['Energy Produced (Wh)', 'Energy Consumed (Wh)'] 
                          if col in available_columns]
        
        selected_columns = st.multiselect(
            "Selecteer kolommen voor outlier detectie:",
            options=available_columns,
            default=default_columns
        )
        
        if not selected_columns:
            st.warning("Selecteer ten minste één kolom voor outlier detectie.")
            return
        
        # Selecteer methode voor outlier detectie
        method = st.radio(
            "Selecteer outlier detectie methode:",
            options=["IQR (Interquartile Range)", "Z-score"],
            horizontal=True
        )
        
        method_map = {
            "IQR (Interquartile Range)": "iqr",
            "Z-score": "zscore"
        }
        
        # Stel drempelwaarde in
        if method_map[method] == "iqr":
            threshold = st.slider(
                "IQR Drempelwaarde (standaard: 1.5):",
                min_value=1.0,
                max_value=3.0,
                value=1.5,
                step=0.1
            )
        else:  # z-score
            threshold = st.slider(
                "Z-score Drempelwaarde (standaard: 3.0):",
                min_value=2.0,
                max_value=5.0,
                value=3.0,
                step=0.1
            )
        
        # Detecteer outliers
        outliers = data_analysis.detect_outliers(
            data=data,
            columns=selected_columns,
            method=method_map[method],
            threshold=threshold
        )
        
        # Toon resultaten per kolom
        for col in selected_columns:
            st.subheader(f"Outliers in {col}")
            
            if col in outliers and not outliers[col].empty:
                # Toon aantal outliers en percentage
                outlier_count = len(outliers[col])
                outlier_percentage = (outlier_count / len(data)) * 100
                
                st.write(f"Gevonden: {outlier_count} outliers ({outlier_percentage:.2f}% van de data)")
                
                # Visualiseer de outliers in een box plot
                fig = go.Figure()
                
                fig.add_trace(go.Box(
                    y=data[col],
                    name=col,
                    boxpoints='outliers',
                    jitter=0.3,
                    pointpos=-1.8,
                    marker=dict(color='blue', size=3),
                    line=dict(color='blue')
                ))
                
                fig.update_layout(
                    title=f"Box Plot voor {col}",
                    yaxis_title=col,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Toon de outlier data
                if st.checkbox(f"Toon outlier data voor {col}"):
                    # Reset de index voor weergave als Date/Time een index is
                    display_df = outliers[col].copy()
                    if isinstance(display_df.index, pd.DatetimeIndex):
                        display_df = display_df.reset_index()
                    
                    st.dataframe(display_df, use_container_width=True)
                
            else:
                st.write("Geen outliers gevonden met de huidige instellingen.")
        
    except Exception as e:
        st.error(f"Fout bij het uitvoeren van outlier analyse: {str(e)}")