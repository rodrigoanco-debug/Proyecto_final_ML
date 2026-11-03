"""Construcción de los conjuntos de características A y B usados en los notebooks 02-04.

Solo se utilizan vuelos y clima de enero a noviembre de 2013. La publicación
meteorológica se estima como hora de observación + 2 horas; no está verificada.
"""

import numpy as np
import pandas as pd
from data import load_raw, TARGET_COL

FEATURES_A = ['carrier', 'origin', 'dest', 'distance', 'dia_semana',
              'salida_min', 'llegada_min', 'mes_sin', 'mes_cos']
CATEGORICAS = ['carrier', 'origin', 'dest', 'dia_semana']
NUMERICAS_A = ['distance', 'salida_min', 'llegada_min', 'mes_sin', 'mes_cos']
CLIMA = ['temp', 'dewp', 'humid', 'wind_speed', 'precip', 'pressure', 'visib']
HISTORICAS = ['hist_' + c for c in CLIMA]
EXTRA_CLIMA = (HISTORICAS + ['edad_clima_h', 'sin_clima', 'hist_viento_sospechoso']
               + [c + '_faltante' for c in HISTORICAS])
FEATURES_B = FEATURES_A + EXTRA_CLIMA


def hhmm_a_minutos(serie):
    horas = serie // 100
    minutos = serie % 100
    valido = serie.notna() & horas.between(0, 23) & minutos.between(0, 59)
    return (60 * horas + minutos).where(valido)


def crear_programacion(tabla):
    resultado = tabla.copy()
    resultado['salida_min'] = hhmm_a_minutos(resultado['sched_dep_time'])
    resultado['llegada_min'] = hhmm_a_minutos(resultado['sched_arr_time'])
    resultado['dia_semana'] = pd.to_datetime(resultado[['year', 'month', 'day']]).dt.dayofweek
    resultado['mes_sin'] = np.sin(2 * np.pi * (resultado['month'] - 1) / 12)
    resultado['mes_cos'] = np.cos(2 * np.pi * (resultado['month'] - 1) / 12)
    resultado['salida_programada_utc'] = resultado['time_hour'] + pd.to_timedelta(resultado['minute'], unit='m')
    resultado['prediccion_utc'] = resultado['salida_programada_utc'] - pd.Timedelta(hours=2)
    return resultado


def agregar_clima_historico(preparados, clima):
    """Asigna a cada vuelo la última observación de su aeropuerto disponible
    (hora + 2 h) en el instante de predicción, con antigüedad de 2 a 6 h.
    Es la misma regla del notebook 02."""
    clima_limpio = clima[['origin', 'time_hour'] + CLIMA].copy()
    clima_limpio['viento_sospechoso'] = (clima_limpio['wind_speed'] < 0) | (clima_limpio['wind_speed'] > 200)
    clima_limpio.loc[clima_limpio['viento_sospechoso'], 'wind_speed'] = np.nan
    clima_limpio['disponible_estimada_utc'] = clima_limpio['time_hour'] + pd.Timedelta(hours=2)
    clima_limpio = clima_limpio.rename(columns={'time_hour': 'hora_clima_utc',
                                               **{c: 'hist_' + c for c in CLIMA}})
    historico = pd.merge_asof(
        preparados[['id_vuelo', 'origin', 'prediccion_utc']].sort_values('prediccion_utc'),
        clima_limpio.sort_values('disponible_estimada_utc'),
        left_on='prediccion_utc', right_on='disponible_estimada_utc',
        by='origin', direction='backward'
    )
    historico['edad_clima_h'] = (historico['prediccion_utc'] - historico['hora_clima_utc']).dt.total_seconds() / 3600
    historico['clima_disponible'] = historico['hora_clima_utc'].notna() & historico['edad_clima_h'].between(2, 6)
    historico.loc[~historico['clima_disponible'], HISTORICAS] = np.nan
    historico['sin_clima'] = (~historico['clima_disponible']).astype(int)
    historico['hist_viento_sospechoso'] = historico['viento_sospechoso'].fillna(False).astype(int)
    for c in HISTORICAS:
        historico[c + '_faltante'] = historico[c].isna().astype(int)
    validos = historico.loc[historico['clima_disponible']]
    assert (validos['disponible_estimada_utc'] <= validos['prediccion_utc']).all()
    assert validos['edad_clima_h'].between(2, 6).all()
    assert len(historico) == len(preparados) and historico['id_vuelo'].is_unique
    trazabilidad = ['hora_clima_utc', 'disponible_estimada_utc', 'clima_disponible']
    return preparados.merge(historico[['id_vuelo'] + EXTRA_CLIMA + trazabilidad],
                             on='id_vuelo', how='left', validate='one_to_one')


def cargar_features():
    """Reconstruye A y B desde nycflights13 (enero-noviembre).

    No lee resultados de otros notebooks, no usa diciembre y no ajusta
    transformadores."""
    raw = load_raw()
    vuelos = raw['flights'].loc[lambda t: t['month'].between(1, 11)].copy()
    clima = raw['weather'].loc[lambda t: t['month'].between(1, 11)].copy()
    del raw
    vuelos = vuelos.reset_index(names='id_vuelo')
    vuelos['time_hour'] = pd.to_datetime(vuelos['time_hour'], utc=True)
    clima['time_hour'] = pd.to_datetime(clima['time_hour'], utc=True)
    vuelos[TARGET_COL] = (vuelos['arr_delay'] >= 15).astype('Int64')
    vuelos.loc[vuelos['arr_delay'].isna(), TARGET_COL] = pd.NA
    preparados = agregar_clima_historico(crear_programacion(vuelos), clima)
    assert preparados['month'].between(1, 11).all()
    assert preparados['id_vuelo'].is_unique
    assert not preparados.duplicated(['carrier', 'flight', 'origin', 'time_hour']).any()
    assert preparados[TARGET_COL].isna().equals(preparados['arr_delay'].isna())
    assert (preparados['salida_programada_utc'] - preparados['prediccion_utc']).eq(pd.Timedelta(hours=2)).all()
    return preparados
