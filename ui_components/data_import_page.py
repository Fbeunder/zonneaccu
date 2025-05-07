"""Component voor de data import pagina in de Streamlit interface."""

import os
import streamlit as st
import pandas as pd
import tempfile
from pathlib import Path

from data_processing import data_loader


def render_data_import_page():
    """Render de data import pagina in de Streamlit interface."""
    st.subheader("Data Importeren")
    
    # Uitleg over de verwachte dataformaten
    with st.expander("Uitleg over dataformaten"):
        st.markdown("""
        ### Verwacht dataformaat
        
        Upload een CSV-bestand met de volgende kolommen:
        - **Date/Time**: Tijdstip van de meting (bijv. 01/01/2024 00:00)
        - **Energy Produced (Wh)**: Geproduceerde energie in Watt-uur
        - **Energy Consumed (Wh)**: Verbruikte energie in Watt-uur
        - **Exported to Grid (Wh)**: Naar het net geëxporteerde energie in Watt-uur
        - **Imported from Grid (Wh)**: Van het net geïmporteerde energie in Watt-uur
        
        Voorbeeld van de eerste regels:
        ```
        Date/Time,Energy Produced (Wh),Energy Consumed (Wh),Exported to Grid (Wh),Imported from Grid (Wh)
        01/01/2024 00:00,0,81,0,81
        01/01/2024 00:15,0,68,0,68
        01/01/2024 00:30,0,66,0,66
        ```
        """)
    
    # Optie om voorbeelddata te gebruiken
    use_sample_data = st.checkbox("Gebruik voorbeelddata", value=False)
    
    if use_sample_data:
        # Genereer voorbeelddata als deze nog niet bestaat
        sample_data_path = "data/sample_energy_data.csv"
        if not os.path.exists(sample_data_path):
            os.makedirs(os.path.dirname(sample_data_path), exist_ok=True)
            data_loader.create_sample_data(sample_data_path, days=1)
        
        # Laad de voorbeelddata
        try:
            df = data_loader.load_csv_data(sample_data_path)
            st.session_state['energy_data'] = df
            st.session_state['data_loaded'] = True
            st.success(f"Voorbeelddata succesvol geladen met {len(df)} metingen.")
            
            # Toon een preview van de data
            show_data_preview(df)
            
        except Exception as e:
            st.error(f"Fout bij het laden van voorbeelddata: {str(e)}")
    
    else:
        # Bestandsuploader voor eigen data
        upload_file = st.file_uploader("Upload CSV met energiedata", type=['csv'])
        
        if upload_file is not None:
            try:
                # Sla het geüploade bestand tijdelijk op
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    tmp_file.write(upload_file.getvalue())
                    temp_path = tmp_file.name
                
                # Laad de data met onze data_loader
                df = data_loader.load_csv_data(temp_path)
                
                # Valideer de data
                is_valid, errors = data_loader.validate_data(df)
                
                if not is_valid:
                    st.warning("De geüploade data bevat mogelijke problemen:")
                    for error in errors:
                        st.warning(f"- {error}")
                    
                    # Vraag of de gebruiker toch door wil gaan
                    if not st.checkbox("Toch doorgaan met deze data", value=False):
                        st.stop()
                
                # Sla de data op in de sessie
                st.session_state['energy_data'] = df
                st.session_state['data_loaded'] = True
                st.success(f"Data succesvol geladen met {len(df)} metingen.")
                
                # Toon een preview van de data
                show_data_preview(df)
                
                # Verwijder het tijdelijke bestand
                os.unlink(temp_path)
                
            except Exception as e:
                st.error(f"Fout bij het verwerken van het bestand: {str(e)}")
    
    # Toon opties voor data preprocessing als er data is geladen
    if st.session_state.get('data_loaded', False):
        st.subheader("Data Preprocessing")
        
        # Optie voor resampling
        resample_interval = st.selectbox(
            "Resample data naar interval:",
            options=["Geen resampling", "15 minuten", "30 minuten", "1 uur", "1 dag"],
            index=0
        )
        
        # Voer resampling uit indien geselecteerd
        if resample_interval != "Geen resampling":
            interval_map = {
                "15 minuten": "15min",
                "30 minuten": "30min",
                "1 uur": "1H",
                "1 dag": "1D"
            }
            
            try:
                resampled_df = data_loader.resample_data(
                    st.session_state['energy_data'], 
                    interval=interval_map[resample_interval]
                )
                
                st.session_state['energy_data'] = resampled_df
                st.success(f"Data succesvol geresampeld naar {resample_interval} intervallen.")
                
                # Toon een preview van de geresamplede data
                show_data_preview(resampled_df)
                
            except Exception as e:
                st.error(f"Fout bij het resampling van de data: {str(e)}")
        
        # Bereken afgeleide waarden
        if st.checkbox("Bereken afgeleide waarden", value=True):
            try:
                derived_df = data_loader.calculate_derived_values(st.session_state['energy_data'])
                st.session_state['energy_data'] = derived_df
                
                # Toon een preview van de data met afgeleide waarden
                show_data_preview(derived_df)
                
            except Exception as e:
                st.error(f"Fout bij het berekenen van afgeleide waarden: {str(e)}")
        
        # Toon basisstatistieken
        if st.checkbox("Toon basisstatistieken", value=True):
            try:
                stats = data_loader.calculate_basic_statistics(st.session_state['energy_data'])
                
                # Toon de statistieken in een nette tabel
                st.subheader("Basisstatistieken")
                
                # Maak een DataFrame van de statistieken voor betere weergave
                stats_df = pd.DataFrame(list(stats.items()), columns=['Statistiek', 'Waarde'])
                st.dataframe(stats_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Fout bij het berekenen van statistieken: {str(e)}")


def show_data_preview(df, rows=5):
    """Toon een preview van de data.
    
    Args:
        df (pd.DataFrame): De te tonen DataFrame.
        rows (int): Aantal rijen om te tonen.
    """
    st.subheader("Data Preview")
    
    # Reset de index voor weergave als Date/Time een index is
    preview_df = df.copy()
    if isinstance(preview_df.index, pd.DatetimeIndex):
        preview_df = preview_df.reset_index()
        preview_df.rename(columns={'index': 'Date/Time'}, inplace=True)
    
    # Toon de eerste paar rijen
    st.dataframe(preview_df.head(rows), use_container_width=True)