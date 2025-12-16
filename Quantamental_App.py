import os
import sys

# CEIC API 클라이언트 경로 추가
ceic_path = os.path.join(os.path.dirname(__file__), 'ceic_api_client')
if os.path.exists(ceic_path):
    sys.path.insert(0, ceic_path)

import pandas as pd
import numpy as np
import itertools
from ceic_api_client.pyceic import Ceic
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import JsCode
import plotly.graph_objects as go
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

# 필요한 함수들을 import (또는 여기에 정의)
# 사용자가 제공한 코드에서 필요한 모든 함수들을 포함해야 합니다

token = st.secrets["CEIC_token"]
Ceic.set_token(token)

def clean_unit(u):
    if u is None or pd.isna(u):
        return u
    u = str(u)
    return u.replace('/', '')

@dataclass
class TradingConfig:
    """트레이딩 분석 설정"""
    analysis_window: int
    slope_threshold: float
    fd_drop_threshold: float
    trading_period: int
    fd_level_threshold: float = 1.1
    fd_level_lookback: int = 3
    slope_metric: str = "rate"
    perf_metric: str = "rate"

def import_time_series_data(file_path: str, date_column: str = 'DATE') -> pd.DataFrame:
    """시계열 데이터프레임을 import합니다."""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {file_path}")
    
    if date_column in df.columns:
        cols = [date_column] + [c for c in df.columns if c != date_column]
        df = df[cols]
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    else:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: date_column})
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        cols = [date_column] + [c for c in df.columns if c != date_column]
        df = df[cols]
    return df

def create_analysis_dataframe(df: pd.DataFrame, analysis_column: str, date_column: str = 'DATE') -> pd.DataFrame:
    """DATE 칼럼과 분석대상 칼럼만 있는 데이터프레임을 생성합니다."""
    if analysis_column not in df.columns:
        raise ValueError(f"분석 칼럼 '{analysis_column}'이 데이터프레임에 없습니다.")
    result_df = df[[date_column, analysis_column]].copy()
    result_df = result_df.dropna(subset=[date_column, analysis_column])
    result_df = result_df.sort_values(by=date_column).reset_index(drop=True)
    return result_df

def calculate_fractal_dimension_higuchi(series: np.ndarray, k_max: Optional[int] = None) -> float:
    """Higuchi 방법을 사용하여 Fractal Dimension을 계산합니다."""
    n = len(series)
    if n < 2:
        return np.nan
    
    series = np.asarray(series, dtype=float)
    series = series[~np.isnan(series)]
    series = series[~np.isinf(series)]
    
    if len(series) < 2:
        return np.nan
    
    series_min = series.min()
    series_max = series.max()
    if series_max == series_min:
        return 1.0
    
    series = (series - series_min) / (series_max - series_min)
    n = len(series)
    
    if k_max is None:
        k_max = max(2, min(10, n // 4))
    k_max = min(k_max, n - 1)
    
    if k_max < 2:
        return np.nan
    
    lk_values = []
    for k in range(1, k_max + 1):
        lk_sum = 0.0
        valid_m_count = 0
        for m in range(1, k + 1):
            indices = np.arange(m - 1, n, k, dtype=int)
            if len(indices) < 2:
                continue
            sampled = series[indices]
            if len(sampled) > 1:
                length = np.sum(np.abs(np.diff(sampled)))
                length = length * (n - 1) / ((len(indices) - 1) * k)
                lk_sum += length
                valid_m_count += 1
        
        if valid_m_count > 0:
            lk = lk_sum / valid_m_count
            if lk > 0 and np.isfinite(lk):
                log_k = np.log(1.0 / k)
                log_lk = np.log(lk)
                if np.isfinite(log_k) and np.isfinite(log_lk):
                    lk_values.append((log_k, log_lk))
    
    if len(lk_values) < 2:
        return np.nan
    
    try:
        x_vals = np.array([v[0] for v in lk_values])
        y_vals = np.array([v[1] for v in lk_values])
        slope = np.polyfit(x_vals, y_vals, 1)[0]
        fd = 2 - slope
        fd = float(np.clip(fd, 1.0, 2.0))
        return fd
    except Exception:
        return np.nan

def calculate_slope(series: np.ndarray) -> float:
    """선형회귀 기울기를 이용해 등락률 형태로 반환."""
    if len(series) < 2:
        return np.nan
    x = np.arange(len(series))
    slope = np.polyfit(x, series, 1)[0]
    if series[0] != 0:
        return (series[-1] - series[0]) / series[0]
    else:
        return np.nan

def calculate_rolling_fd(df: pd.DataFrame, analysis_column: str, window: int,
                         date_column: str = 'DATE') -> pd.DataFrame:
    """일별로 이동하면서 Fractal Dimension 값을 계산합니다."""
    result_df = df.copy()
    result_df['FD'] = np.nan
    values = df[analysis_column].values
    
    for i in range(window - 1, len(df)):
        window_data = values[i - window + 1:i + 1]
        fd = calculate_fractal_dimension_higuchi(window_data)
        result_df.iloc[i, result_df.columns.get_loc('FD')] = fd
    
    return result_df

def analyze_trading_signals(df: pd.DataFrame, config: TradingConfig,
                            analysis_column: str, date_column: str = 'DATE') -> pd.DataFrame:
    """트레이딩 시그널을 분석하고 트레이딩 결과를 기록합니다. (패턴 A 기준)"""
    base_columns = [
        '분석칼럼이름', '시그널발생일', '시그널기울기', '시그널기울기(등락폭)',
        '트레이딩진입일', '트레이딩종료일',
        '트레이딩진입일가격', '트레이딩종료일가격',
        '변화율', '변화폭',
        '성과',
        '연장여부', '연장사유', '연장발생일', '트레이딩기간'
    ]
    
    results = []
    df_with_fd = df[df['FD'].notna()].copy().reset_index(drop=True)
    
    if len(df_with_fd) == 0:
        return pd.DataFrame(columns=base_columns)
    
    FD_LEVEL_THRESHOLD = config.fd_level_threshold
    FD_LEVEL_LOOKBACK = config.fd_level_lookback
    slope_metric = (config.slope_metric or "rate").lower()
    perf_metric = (config.perf_metric or "rate").lower()
    
    if slope_metric not in ("rate", "amount"):
        raise ValueError("slope_metric must be 'rate' or 'amount'")
    if perf_metric not in ("rate", "amount"):
        raise ValueError("perf_metric must be 'rate' or 'amount'")
    
    min_len = max(config.analysis_window, FD_LEVEL_LOOKBACK + 1)
    if len(df_with_fd) < min_len:
        return pd.DataFrame(columns=base_columns)
    
    dates = df_with_fd[date_column].values
    prices = df_with_fd[analysis_column].values
    fd_values = df_with_fd['FD'].values
    
    active_trading = None
    start_idx = max(config.analysis_window - 1, FD_LEVEL_LOOKBACK)
    
    for i in range(start_idx, len(df_with_fd)):
        # --- (1) 기울기(등락률/등락폭) 계산 ---
        window_start = i - config.analysis_window + 1
        window_end = i + 1
        if window_start < 0:
            continue
        
        window_prices = prices[window_start:window_end]
        start_price = window_prices[0]
        end_price = window_prices[-1]
        change_amount = end_price - start_price
        change_rate = change_amount / start_price if start_price != 0 else np.nan
        
        if slope_metric == "rate":
            slope_value = change_rate
        else:
            slope_value = change_amount
        
        if np.isnan(slope_value):
            slope_condition = False
        else:
            slope_condition = abs(slope_value) > config.slope_threshold
        
        # --- (2) FD 절대 레벨 브레이크 조건 ---
        if i < FD_LEVEL_LOOKBACK:
            fd_level_condition = False
        else:
            prev_fds = fd_values[i - FD_LEVEL_LOOKBACK:i]
            curr_fd = fd_values[i]
            if np.isnan(curr_fd) or np.any(np.isnan(prev_fds)):
                fd_level_condition = False
            else:
                fd_level_condition = (
                        np.all(prev_fds >= FD_LEVEL_THRESHOLD) and
                        curr_fd < FD_LEVEL_THRESHOLD
                )
        
        # --- (3) 신규 시그널 조건 ---
        signal_condition = slope_condition and fd_level_condition
        
        # =============================
        #   활성 트레이딩이 있는 경우
        # =============================
        if active_trading is not None:
            (signal_idx, entry_idx, ext_total_count,
             last_signal_idx, expected_end_idx,
             initial_signal_slope,
             cond1_ext_count, cond2_ext_count,
             fd_drop_phase_active,
             extension_dates) = active_trading
            
            extension_triggered = False
            extension_type = None
            fd_drop_phase_active_new = fd_drop_phase_active
            
            # (2-b) 조건2: 기울기+FD 재시그널에 의한 연장
            if signal_condition and i > last_signal_idx and i < expected_end_idx:
                slope_sign = np.sign(slope_value) if np.isfinite(slope_value) else 0.0
                if slope_sign != 0 and i + 1 < len(df_with_fd):
                    next_price = prices[i + 1]
                    signal_price_now = prices[i]
                    if slope_sign > 0:
                        trend_continuation = next_price > signal_price_now
                    else:
                        trend_continuation = next_price < signal_price_now
                    if trend_continuation:
                        extension_triggered = True
                        extension_type = "cond2"
            
            # (2-a) 조건1: 진입 이후 연속 FD 하락 구간에서만 연장 (패턴 A)
            if (not extension_triggered) and (i >= entry_idx + 1) and (i < expected_end_idx):
                prev_fd = fd_values[i - 1]
                curr_fd = fd_values[i]
                if np.isfinite(prev_fd) and np.isfinite(curr_fd):
                    if fd_drop_phase_active:
                        if curr_fd < prev_fd:
                            extension_triggered = True
                            extension_type = "cond1"
                        else:
                            fd_drop_phase_active_new = False
            
            # 실제 연장 처리
            if extension_triggered:
                new_expected_end_idx = i + config.trading_period
                new_last_signal_idx = last_signal_idx
                new_cond1_ext_count = cond1_ext_count
                new_cond2_ext_count = cond2_ext_count
                new_extension_dates = extension_dates.copy()
                new_extension_dates.append(dates[i])
                
                if extension_type == "cond2":
                    new_last_signal_idx = i
                    new_cond2_ext_count += 1
                elif extension_type == "cond1":
                    new_cond1_ext_count += 1
                
                active_trading = (
                    signal_idx,
                    entry_idx,
                    ext_total_count + 1,
                    new_last_signal_idx,
                    new_expected_end_idx,
                    initial_signal_slope,
                    new_cond1_ext_count,
                    new_cond2_ext_count,
                    fd_drop_phase_active_new,
                    new_extension_dates
                )
                continue
            
            # --- 트레이딩 종료 체크 ---
            if i >= expected_end_idx:
                end_idx = expected_end_idx - 1
                if end_idx >= len(df_with_fd):
                    end_idx = len(df_with_fd) - 1
                
                entry_price = prices[entry_idx]
                end_price = prices[end_idx]
                change_amount_trade = end_price - entry_price
                change_rate_trade = (
                    change_amount_trade / entry_price if entry_price != 0 else np.nan
                )
                
                signal_window_start = signal_idx - config.analysis_window + 1
                if signal_window_start < 0:
                    signal_window_start = 0
                signal_start_price = prices[signal_window_start]
                signal_end_price = prices[signal_idx]
                signal_change_amount = signal_end_price - signal_start_price
                
                if perf_metric == "rate":
                    raw_perf = change_rate_trade
                else:
                    raw_perf = change_amount_trade
                
                if np.isfinite(initial_signal_slope) and initial_signal_slope > 0:
                    perf = -raw_perf
                elif np.isfinite(initial_signal_slope) and initial_signal_slope < 0:
                    perf = raw_perf
                else:
                    perf = raw_perf
                
                reason_parts = []
                if cond1_ext_count > 0:
                    reason_parts.append(f"조건1(FD추가하락) {cond1_ext_count}회")
                if cond2_ext_count > 0:
                    reason_parts.append(f"조건2(기울기+FD재시그널) {cond2_ext_count}회")
                extension_reason = "; ".join(reason_parts)
                
                if extension_dates:
                    extension_dates_str = ", ".join([pd.to_datetime(d).strftime('%Y-%m-%d') for d in extension_dates])
                else:
                    extension_dates_str = ""
                
                trading_days = end_idx - entry_idx + 1
                
                results.append({
                    '분석칼럼이름': analysis_column,
                    '시그널발생일': dates[signal_idx],
                    '시그널기울기': initial_signal_slope,
                    '시그널기울기(등락폭)': signal_change_amount,
                    '트레이딩진입일': dates[entry_idx],
                    '트레이딩종료일': dates[end_idx],
                    '트레이딩진입일가격': entry_price,
                    '트레이딩종료일가격': end_price,
                    '변화율': change_rate_trade,
                    '변화폭': change_amount_trade,
                    '성과': perf,
                    '연장여부': 'Y' if ext_total_count > 0 else 'N',
                    '연장사유': extension_reason,
                    '연장발생일': extension_dates_str,
                    '트레이딩기간': trading_days
                })
                
                active_trading = None
            else:
                active_trading = (
                    signal_idx,
                    entry_idx,
                    ext_total_count,
                    last_signal_idx,
                    expected_end_idx,
                    initial_signal_slope,
                    cond1_ext_count,
                    cond2_ext_count,
                    fd_drop_phase_active_new,
                    extension_dates
                )
        
        # =============================
        #   활성 트레이딩이 없는 경우 (신규 진입)
        # =============================
        if signal_condition and active_trading is None:
            if i + 1 < len(df_with_fd):
                entry_idx = i + 1
                expected_end_idx = entry_idx + config.trading_period
                initial_signal_slope = slope_value
                active_trading = (
                    i,
                    entry_idx,
                    0,
                    i,
                    expected_end_idx,
                    initial_signal_slope,
                    0,
                    0,
                    True,
                    []  # 연장 발생일 리스트
                )
    
    # 루프 종료 후 미청산 트레이딩 처리
    if active_trading is not None:
        (signal_idx, entry_idx, ext_total_count,
         last_signal_idx, expected_end_idx,
         initial_signal_slope,
         cond1_ext_count, cond2_ext_count,
         fd_drop_phase_active,
         extension_dates) = active_trading
        
        end_idx = len(df_with_fd) - 1
        entry_price = prices[entry_idx]
        end_price = prices[end_idx]
        change_amount_trade = end_price - entry_price
        change_rate_trade = (
            change_amount_trade / entry_price if entry_price != 0 else np.nan
        )
        
        signal_window_start = signal_idx - config.analysis_window + 1
        if signal_window_start < 0:
            signal_window_start = 0
        signal_start_price = prices[signal_window_start]
        signal_end_price = prices[signal_idx]
        signal_change_amount = signal_end_price - signal_start_price
        
        if perf_metric == "rate":
            raw_perf = change_rate_trade
        else:
            raw_perf = change_amount_trade
        
        if np.isfinite(initial_signal_slope) and initial_signal_slope > 0:
            perf = -raw_perf
        elif np.isfinite(initial_signal_slope) and initial_signal_slope < 0:
            perf = raw_perf
        else:
            perf = raw_perf
        
        reason_parts = []
        if cond1_ext_count > 0:
            reason_parts.append(f"조건1(FD추가하락) {cond1_ext_count}회")
        if cond2_ext_count > 0:
            reason_parts.append(f"조건2(기울기+FD재시그널) {cond2_ext_count}회")
        extension_reason = "; ".join(reason_parts)
        
        if extension_dates:
            extension_dates_str = ", ".join([pd.to_datetime(d).strftime('%Y-%m-%d') for d in extension_dates])
        else:
            extension_dates_str = ""
        
        trading_days = end_idx - entry_idx + 1
        
        results.append({
            '분석칼럼이름': analysis_column,
            '시그널발생일': dates[signal_idx],
            '시그널기울기': initial_signal_slope,
            '시그널기울기(등락폭)': signal_change_amount,
            '트레이딩진입일': dates[entry_idx],
            '트레이딩종료일': dates[end_idx],
            '트레이딩진입일가격': entry_price,
            '트레이딩종료일가격': end_price,
            '변화율': change_rate_trade,
            '변화폭': change_amount_trade,
            '성과': perf,
            '연장여부': 'Y' if ext_total_count > 0 else 'N',
            '연장사유': extension_reason,
            '연장발생일': extension_dates_str,
            '트레이딩기간': trading_days
        })
    
    if len(results) == 0:
        return pd.DataFrame(columns=base_columns)
    return pd.DataFrame(results)[base_columns]

def run_fractal_dimension_analysis(
        file_path: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        analysis_column: str = None,
        config: TradingConfig = None,
        date_column: str = 'date'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fractal Dimension 기반 트레이딩 시그널 분석을 실행합니다."""
    if file_path is None and df is None:
        raise ValueError("file_path 또는 df 중 하나는 필수입니다.")
    if analysis_column is None:
        raise ValueError("analysis_column은 필수입니다.")
    if config is None:
        raise ValueError("config는 필수입니다.")
    
    if file_path is not None:
        df = import_time_series_data(file_path, date_column)
    else:
        if date_column not in df.columns:
            first_col = df.columns[0]
            df = df.rename(columns={first_col: date_column})
            cols = [date_column] + [c for c in df.columns if c != date_column]
            df = df[cols]
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    analysis_df = create_analysis_dataframe(df, analysis_column, date_column)
    fd_df = calculate_rolling_fd(analysis_df, analysis_column, config.analysis_window, date_column)
    trading_results = analyze_trading_signals(fd_df, config, analysis_column, date_column)
    
    return fd_df, trading_results

# Streamlit 앱 시작
st.set_page_config(layout="wide", page_title="Fractal Dimension Trading Analysis")

# CSS로 폰트 크기 20% 감소
st.markdown("""
    <style>
    .stApp {
        font-size: 0.8em;
    }
    h1 {
        font-size: 2.4em;
    }
    h2 {
        font-size: 2.0em;
    }
    h3 {
        font-size: 1.6em;
    }
    .stDataFrame {
        font-size: 0.8em;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Quantamental Analysis Dashboard")

# ISM PMI 데이터 로딩 함수 (캐싱 추가)
@st.cache_data(ttl=604800)  # 1주일 (604800초)
def load_ism_pmi_data():
    """ISM Man. PMI 데이터를 로드합니다."""
    codes_U = ['41044601',
               '41044701', '41044801', '41045501', '41044901', '41045001',
               '41045101', '41045201', '41045301', '41045401', '212050202']
    
    df_U = Ceic.series(codes_U, start_date='2025-01-01').as_pandas()
    meta_U = Ceic.series_metadata(codes_U).as_pandas()
    
    meta_U["name"] = (
        meta_U["name"]
        .str.replace("Report On Business: Purchasing Managers' Index", "ISM Man. PMI", regex=False)
        .str.replace("Report On Business: PMI: ", "", regex=False)
        .str.replace(" Index", "", regex=False)
    )
    
    df_U['id'] = df_U['id'].astype(str)
    meta_U['id'] = meta_U['id'].astype(str)
    df_pivot = df_U.pivot(index='date', columns='id', values='value')
    id_to_unit = meta_U.set_index('id')['name']
    df_pivot = df_pivot.rename(columns=lambda x: clean_unit(id_to_unit.get(x, x)))
    df_U_tr = df_pivot.sort_index()
    df_U_tr = df_U_tr.reset_index().rename(columns={'index': 'date'})
    df_U_tr = df_U_tr[["date", "ISM Man. PMI",
                       "New Orders", "Production", "Employment", "Supplier Deliveries", "Inventories",
                       "New Export Orders", "Imports", "Prices", "Customers Inventories", "Backlog of Orders"]]
    
    df_oriU = pd.read_csv('ori_MPMI.csv')
    df_oriU['date'] = pd.to_datetime(df_oriU['date'])
    df_U_tr['date'] = pd.to_datetime(df_U_tr['date'])
    df_U = pd.concat([df_oriU, df_U_tr], ignore_index=True)
    raw_df = df_U.copy()
    raw_df = raw_df[raw_df['date'] >= '1992-01-01']
    
    return raw_df

@st.cache_data(ttl=604800)  # 1주일 (604800초)
def load_ism_srv_data():
    """ISM Srv. PMI 데이터를 로드합니다."""
    codes_S = ['212050702',
               '41045601', '41045701', '41046501', '41045901', '41045801',
               '41046001', '41046101', '41046201', '41046301', '41046401']
    
    df_S = Ceic.series(codes_S, start_date='2025-01-01').as_pandas()
    meta_S = Ceic.series_metadata(codes_S).as_pandas()
    
    meta_S["name"] = (
        meta_S["name"]
        .str.replace("Report On Business: Purchasing Manager Index (PMI): Services", "ISM Srv. PMI", regex=False)
        .str.replace("Report On Business: PMI: Services: ", "", regex=False)
        .str.replace(" Index", "", regex=False)
    )
    
    df_S['id'] = df_S['id'].astype(str)
    meta_S['id'] = meta_S['id'].astype(str)
    df_pivot = df_S.pivot(index='date', columns='id', values='value')
    id_to_unit = meta_S.set_index('id')['name']
    df_pivot = df_pivot.rename(columns=lambda x: clean_unit(id_to_unit.get(x, x)))
    df_S_tr = df_pivot.sort_index()
    df_S_tr = df_S_tr.reset_index().rename(columns={'index': 'date'})
    df_S_tr = df_S_tr[["date", "ISM Srv. PMI",
                       "Business Activity", "New Orders", "Employment", "Supplier Deliveries", "Backlog of Orders",
                       "New Export Orders", "Imports", "Inventory Change", "Inventory Sentiment", "Prices"]]
    
    df_oriS = pd.read_csv('ori_SPMI.csv')
    df_oriS['date'] = pd.to_datetime(df_oriS['date'])
    df_S_tr['date'] = pd.to_datetime(df_S_tr['date'])
    df_S = pd.concat([df_oriS, df_S_tr], ignore_index=True)
    raw_df = df_S.copy()
    raw_df = raw_df[raw_df['date'] >= '1992-01-01']
    
    return raw_df
    
# 데이터 로딩 및 분석 실행
@st.cache_data(ttl=604800)  # 1주일
def load_and_analyze_data():
    """데이터를 로드하고 분석을 실행합니다."""
    today = pd.Timestamp.today().normalize()
    
    # CEIC 데이터 로딩 (기존 코드)
    codes_U = ['241331802', '248237602', '204838802', '3824501', '19329601', '1330801', '474236057',
               '110134508', '1026501', '384682017', '45055901', '51270501', '13629001']
    df_U = Ceic.series(codes_U, start_date='2025-12-01').as_pandas()
    meta_U = Ceic.series_metadata(codes_U).as_pandas()
    
    df_U['id'] = df_U['id'].astype(str)
    meta_U['id'] = meta_U['id'].astype(str)
    df_pivot = df_U.pivot(index='date', columns='id', values='value')
    id_to_unit = meta_U.set_index('id')['unit']
    df_pivot = df_pivot.rename(columns=lambda x: clean_unit(id_to_unit.get(x, x)))
    df_U_tr = df_pivot.sort_index()
    df_U_tr = df_U_tr.reset_index().rename(columns={'index': 'date'})
    
    codes_K = ['28432001', '28434901', '28433901', '28432101', '28432901', '28434501', '28432301', '28432601',
               '28432801',
               '28433001', '28433301', '28433501', '28433601', '28433701', '28433801', '28432701', '28434001',
               '28434101',
               '28434201', '28434301', '28434401', '28434801']
    df_K = Ceic.series(codes_K, start_date='2025-12-01').as_pandas()
    meta_K = Ceic.series_metadata(codes_K).as_pandas()
    
    df_K['id'] = df_K['id'].astype(str)
    meta_K['id'] = meta_K['id'].astype(str)
    df_pivot = df_K.pivot(index='date', columns='id', values='value')
    id_to_unit = meta_K.set_index('id')['unit']
    df_pivot = df_pivot.rename(columns=lambda x: clean_unit(id_to_unit.get(x, x)))
    df_K_tr = df_pivot.sort_index()
    df_K_tr = df_K_tr.reset_index().rename(columns={'index': 'date'})
    
    codes_P1 = ['248650402', '40759601']
    df_P1 = Ceic.series(codes_P1, start_date='2025-12-01').as_pandas()
    id_name_list = [
        ("248650402", "DXY"),
        ("40759601", "SPX"),
    ]
    id_to_name = dict(id_name_list)
    df_P1['id'] = df_P1['id'].astype(str)
    df_pivot = df_P1.pivot(index='date', columns='id', values='value')
    df_pivot = df_pivot.rename(columns=lambda x: id_to_name.get(str(x), str(x)))
    df_P1_tr = df_pivot.sort_index()
    df_P1_tr = df_P1_tr.reset_index()
    
    codes_P2 = ['546475177', '546429197', '40754701', '40754901', '40755101', '40755301']
    df_P2 = Ceic.series(codes_P2, start_date='2025-12-01').as_pandas()
    id_name_list = [
        ("546475177", "SPR_IG"),
        ("546429197", "SPR_HY"),
        ("40754701", "US02"),
        ("40754901", "US05"),
        ("40755101", "US10"),
        ("40755301", "US30"),
    ]
    id_to_name = dict(id_name_list)
    df_P2['id'] = df_P2['id'].astype(str)
    df_pivot = df_P2.pivot(index='date', columns='id', values='value')
    df_pivot = df_pivot.rename(columns=lambda x: id_to_name.get(str(x), str(x)))
    df_P2_tr = df_pivot.sort_index()
    df_P2_tr = df_P2_tr.reset_index()
    
    # CSV 백업 파일 로드
    df_oriU = pd.read_csv('ori_USD.csv')
    df_oriK = pd.read_csv('ori_KRW.csv')
    df_oriP1 = pd.read_csv('ori_P1.csv')
    df_oriP2 = pd.read_csv('ori_P2.csv')
    
    df_oriU['date'] = pd.to_datetime(df_oriU['date'])
    df_oriK['date'] = pd.to_datetime(df_oriK['date'])
    df_oriP1['date'] = pd.to_datetime(df_oriP1['date'])
    df_oriP2['date'] = pd.to_datetime(df_oriP2['date'])
    
    df_U_tr['date'] = pd.to_datetime(df_U_tr['date'])
    df_K_tr['date'] = pd.to_datetime(df_K_tr['date'])
    df_P1_tr['date'] = pd.to_datetime(df_P1_tr['date'])
    df_P2_tr['date'] = pd.to_datetime(df_P2_tr['date'])
    
    df_U = pd.concat([df_oriU, df_U_tr], ignore_index=True)
    df_K = pd.concat([df_oriK, df_K_tr], ignore_index=True)
    df_P1 = pd.concat([df_oriP1, df_P1_tr], ignore_index=True)
    df_P2 = pd.concat([df_oriP2, df_P2_tr], ignore_index=True)
    
    df_all1 = pd.merge(df_U, df_K, on='date', how='outer')
    df_all1 = pd.merge(df_all1, df_P1, on='date', how='outer')
    df_all1 = df_all1.rename(columns={'100 IDRKRW': 'IDRKRW'})
    df_all2 = df_P2.copy()
    
    all1_selv = ['date', 'USDKRW', 'GBPKRW', 'AUDKRW', 'JPYKRW', 'INRKRW', 'EURKRW', 'RMBKRW', 'DXY', 'SPX']
    df_all1 = df_all1[all1_selv]
    df_all1 = df_all1[df_all1['date'] < today]
    
    df_all2['Crv_2_10'] = df_all2['US10'] - df_all2['US02']
    df_all2['Crv_10_30'] = df_all2['US30'] - df_all2['US10']
    df_all2['Crv_2_30'] = df_all2['US30'] - df_all2['US02']
    df_all2['IGHY'] = df_all2['SPR_HY'] - df_all2['SPR_IG']
    
    all2_selv = ['date', 'US10', 'US30', 'SPR_HY', 'SPR_IG', 'Crv_2_10', 'Crv_10_30', 'Crv_2_30', 'IGHY']
    df_all2 = df_all2[all2_selv]
    
    div100 = ['SPR_IG', 'SPR_HY', 'IGHY']
    df_all2[div100] = df_all2[div100].div(100)
    df_all2 = df_all2[df_all2['date'] < today]
    
    df_all = pd.merge(df_all1, df_all2, on='date', how='outer')
    raw_df = df_all.copy()
    raw_df = raw_df[raw_df['date'] >= '2015-01-01']
    
    return raw_df

@st.cache_data(ttl=604800)  # 1주일
def run_analysis(raw_df):
    """선택된 케이스에 대해 분석을 실행합니다."""
    sel_case = [
        ('AUDKRW', 20, 0.02, 60, 1.3, 5, 'rate', 'rate'),
        ('AUDKRW', 60, 0.02, 40, 1.35, 3, 'rate', 'rate'),
        ('AUDKRW', 40, 0.02, 40, 1.35, 3, 'rate', 'rate'),
        ('Crv_2_10', 20, 0.2, 60, 1.4, 3, 'amount', 'amount'),
        ('Crv_2_10', 40, 0.2, 10, 1.4, 5, 'amount', 'amount'),
        ('Crv_2_30', 20, 0.2, 10, 1.35, 3, 'amount', 'amount'),
        ('Crv_2_30', 40, 0.2, 20, 1.3, 5, 'amount', 'amount'),
        ('DXY', 60, 0.02, 60, 1.4, 5, 'rate', 'rate'),
        ('DXY', 60, 0.02, 20, 1.4, 3, 'rate', 'rate'),
        ('EURKRW', 60, 0.02, 20, 1.4, 5, 'rate', 'rate'),
        ('EURKRW', 40, 0.02, 60, 1.35, 3, 'rate', 'rate'),
        ('IGHY', 20, 0.2, 60, 1.35, 3, 'amount', 'amount'),
        ('IGHY', 60, 0.2, 40, 1.4, 5, 'amount', 'amount'),
        ('INRKRW', 40, 0.02, 40, 1.4, 3, 'rate', 'rate'),
        ('JPYKRW', 40, 0.02, 60, 1.3, 5, 'rate', 'rate'),
        ('JPYKRW', 60, 0.02, 40, 1.35, 3, 'rate', 'rate'),
        ('JPYKRW', 20, 0.02, 60, 1.25, 5, 'rate', 'rate'),
        ('RMBKRW', 40, 0.02, 60, 1.4, 5, 'rate', 'rate'),
        ('SPR_HY', 60, 0.5, 10, 1.4, 3, 'amount', 'amount'),
        ('SPX', 20, 0.02, 20, 1.3, 5, 'rate', 'rate'),
        ('SPX', 20, 0.05, 20, 1.4, 3, 'rate', 'rate'),
        ('US10', 20, 0.2, 10, 1.3, 5, 'amount', 'amount'),
        ('US10', 40, 0.2, 40, 1.4, 5, 'amount', 'amount'),
        ('US10', 60, 0.2, 20, 1.35, 3, 'amount', 'amount'),
        ('USDKRW', 60, 0.02, 40, 1.4, 3, 'rate', 'rate'),
        ('USDKRW', 20, 0.02, 40, 1.4, 5, 'rate', 'rate'),
    ]
    
    selected_detail_results = {}
    selected_timeseries_results = {}
    selected_summary_rows = []
    date_column = 'date'
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for case_idx, case in enumerate(sel_case, 1):
        col, aw, st_val, tp, fd_th, fd_lb, slope_metric, perf_metric = case
        
        status_text.text(
            f"[{case_idx}/{len(sel_case)}] 실행 중: {col}, aw={aw}, st={st_val}, tp={tp}, fd_th={fd_th}, fd_lb={fd_lb}")
        progress_bar.progress(case_idx / len(sel_case))
        
        if col not in raw_df.columns:
            continue
        
        config = TradingConfig(
            analysis_window=aw,
            slope_threshold=st_val,
            fd_drop_threshold=0.1,
            trading_period=tp,
            fd_level_threshold=fd_th,
            fd_level_lookback=fd_lb,
            slope_metric=slope_metric,
            perf_metric=perf_metric
        )
        
        try:
            fd_df, trades_df = run_fractal_dimension_analysis(
                df=raw_df,
                file_path=None,
                analysis_column=col,
                config=config,
                date_column=date_column
            )
            
            trade_count = len(trades_df)
            if trade_count > 0:
                hit_ratio = (trades_df['성과'] > 0).mean()
                avg_perf = trades_df['성과'].mean()
            else:
                hit_ratio = np.nan
                avg_perf = np.nan
            
            selected_summary_rows.append({
                'analysis_column': col,
                'analysis_window': aw,
                'slope_threshold': st_val,
                'trading_period': tp,
                'fd_level_threshold': fd_th,
                'fd_level_lookback': fd_lb,
                'slope_metric': slope_metric,
                'perf_metric': perf_metric,
                'trade_count': trade_count,
                'hit_ratio': hit_ratio,
                'avg_perf': avg_perf
            })
            
            cfg_key = (col, aw, st_val, tp, fd_th, fd_lb, slope_metric, perf_metric)
            selected_detail_results[cfg_key] = trades_df
            
            timeseries_df = fd_df[[date_column, col, 'FD']].copy()
            timeseries_df = timeseries_df.rename(columns={
                col: 'value',
                'FD': 'fractal_dimension_value'
            })
            timeseries_df[date_column] = pd.to_datetime(timeseries_df[date_column])
            timeseries_df['signal여부'] = 0
            timeseries_df['signal_slope_sign'] = 0  # 0: no signal, 1: positive slope, -1: negative slope
            
            if len(trades_df) > 0 and '시그널발생일' in trades_df.columns and '시그널기울기' in trades_df.columns:
                signal_info = {}
                for _, row in trades_df.iterrows():
                    signal_date = pd.to_datetime(row['시그널발생일'])
                    signal_slope = row['시그널기울기']
                    if pd.notna(signal_slope):
                        slope_sign = 1 if signal_slope > 0 else -1
                    else:
                        slope_sign = 0
                    signal_info[signal_date] = slope_sign
                
                for idx, row in timeseries_df.iterrows():
                    date_val = row[date_column]
                    if date_val in signal_info:
                        timeseries_df.loc[idx, 'signal여부'] = 1
                        timeseries_df.loc[idx, 'signal_slope_sign'] = signal_info[date_val]
            
            timeseries_df = timeseries_df[
                [date_column, 'value', 'fractal_dimension_value', 'signal여부', 'signal_slope_sign']]
            selected_timeseries_results[cfg_key] = timeseries_df
            
        except Exception as e:
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    selected_summary_df = pd.DataFrame(selected_summary_rows)
    if len(selected_summary_df) > 0:
        indicator_order = ['USDKRW', 'EURKRW', 'JPYKRW', 'INRKRW', 'RMBKRW', 'AUDKRW',
                           'US10', 'Crv_2_10', 'Crv_2_30', 'SPR_HY', 'IGHY', 'DXY', 'SPX']
        
        selected_summary_df['order'] = selected_summary_df['analysis_column'].apply(
            lambda x: indicator_order.index(x) if x in indicator_order else 999
        )
        selected_summary_df_sorted = selected_summary_df.sort_values(
            by=['order', 'analysis_window'],
            ascending=[True, False]
        ).drop(columns=['order'])
    else:
        selected_summary_df_sorted = pd.DataFrame()
    
    return selected_summary_df_sorted, selected_timeseries_results, selected_detail_results

# 탭 생성
tab1, tab2, tab3 = st.tabs(["US Man.PMI", "US Srv.PMI", "FDS"])

with tab1:
    st.header("US ISM Man. PMI")
    
    # 새로고침 버튼
    col_btn, col_info = st.columns([10, 1])
    with col_btn:
        if st.button("새로고침(30초 이내)", key="refresh_ism", help="최신 데이터를 즉시 불러옵니다"):
            load_ism_pmi_data.clear()
            st.success("데이터를 새로 불러오는 중...")
            st.rerun()    
        st.caption("💡 기본적으로 캐시된 데이터를 사용합니다. 최신 데이터가 필요할 때만 새로고침 버튼을 클릭하세요.")
    
    # 캐싱된 함수 호출
    raw_df = load_ism_pmi_data()
    
    # 데이터의 최신 날짜 표시
    if len(raw_df) > 0:
        latest_date = raw_df['date'].max()
        st.caption(f"📅 데이터 최신 날짜: {latest_date.strftime('%Y-%m-%d')}")
    
    # 최근 6개 날짜 기준으로 데이터 필터링 및 전치
    n_show = 6
    latest_dates = raw_df['date'].sort_values(ascending=False).head(n_show).sort_values(ascending=False)
    
    df_for_disp = raw_df.copy()
    df_for_disp = df_for_disp[df_for_disp['date'].isin(latest_dates)].sort_values('date', ascending=False)
    df_for_disp = df_for_disp.reset_index(drop=True)
    df_for_disp_disp = df_for_disp.drop(columns=['date'])
    
    original_columns = list(df_for_disp_disp.columns)
    
    transposed = df_for_disp_disp.T
    transposed.columns = [dt.strftime('%Y.%m') for dt in df_for_disp['date']]
    transposed.index.name = None
    transposed.reset_index(inplace=True)
    transposed.rename(columns={'index': '항목'}, inplace=True)
    
    delta_cols = []
    for i in range(1, n_show):
        chg_col = f'Chg{i}'
        delta_vals = transposed.iloc[:, i + 1] - transposed.iloc[:, i]
        delta_vals = delta_vals.apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")
        delta_cols.append((chg_col, delta_vals))
        transposed[chg_col] = delta_vals
    
    date_cols = [dt.strftime('%Y.%m') for dt in df_for_disp['date']]
    chg_cols = [f'Chg{i}' for i in range(1, n_show)]
    transposed = transposed[['항목'] + date_cols + chg_cols]
    
    transposed['항목'] = pd.Categorical(transposed['항목'], categories=original_columns, ordered=True)
    transposed = transposed.sort_values('항목').reset_index(drop=True)
    
    st.subheader("미국 ISM 제조업 PMI")
    
    gb = GridOptionsBuilder.from_dataframe(transposed)
    gb.configure_default_column(resizable=True, filter=True, sortable=True)
    
    for col in date_cols + chg_cols:
        gb.configure_column(col, cellStyle={"textAlign": "center"})
    
    def get_row_style_js():
        return JsCode("""
        function(params) {
            if (params.node.rowIndex === 0) {
                return {
                    'backgroundColor': '#1565c0',
                    'color': 'white',
                    'fontWeight': 'bold'
                }
            } else {
                return {
                    'fontFamily': 'inherit',
                    'paddingLeft': '20px'
                }
            }
        }
        """)
    
    indent_js = JsCode("""
    function(params) {
        if (params.node.rowIndex === 0) {
            return params.value;
        } else {
            return '\\u00A0\\u00A0' + params.value;
        }
    }
    """)
    
    gb.configure_column("항목", cellRenderer=indent_js)
    gb.configure_grid_options(getRowStyle=get_row_style_js())
    
    AgGrid(
        transposed,
        gridOptions=gb.build(),
        height=400,
        width='100%',
        fit_columns_on_grid_load=True,
        theme="streamlit",
        allow_unsafe_jscode=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        chg_cols = ["Chg1", "Chg2", "Chg3"]
        bar_colors = [
            "rgb(245,130,32)",
            "rgb(4,59,114)",
            "rgb(0,169,206)"
        ]
        
        x_vals = transposed["항목"].tolist()
        y1 = transposed[chg_cols[0]].tolist()
        y2 = transposed[chg_cols[1]].tolist()
        y3 = transposed[chg_cols[2]].tolist()
        
        chg_labels = list(transposed.columns[1:4])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=x_vals,
            y=y1,
            name=chg_labels[0],
            marker_color=bar_colors[0]
        ))
        fig.add_trace(go.Bar(
            x=x_vals,
            y=y2,
            name=chg_labels[1],
            marker_color=bar_colors[1]
        ))
        fig.add_trace(go.Bar(
            x=x_vals,
            y=y3,
            name=chg_labels[2],
            marker_color=bar_colors[2]
        ))
        
        fig.update_layout(
            barmode='group',
            xaxis_title="항목",
            yaxis_title="변화량",
            margin=dict(l=20, r=20, t=40, b=40),
            legend_title="날짜"
        )
        
        st.subheader("Change")
        st.plotly_chart(fig, use_container_width=True)
    
    ism_items = ["ISM Man. PMI", "New Orders", "Production", "Employment", "Supplier Deliveries", "Inventories"]
    
    try:
        base_df = raw_df.copy()
    except NameError:
        base_df = None
    
    if base_df is not None:
        date_col_candidates = [col for col in base_df.columns if 'date' in col.lower() or '날짜' in col]
        if len(date_col_candidates) > 0:
            date_col = date_col_candidates[0]
        else:
            date_col = base_df.columns[0]
        
        base_df[date_col] = pd.to_datetime(base_df[date_col])
        
        min_date = base_df[date_col].min()
        max_date = base_df[date_col].max()
        default_start = pd.to_datetime("2023-01-01")
        default_start = max(default_start, min_date)
        
        with col2:
            st.subheader("Time Series")
            
            col_start, col_end = st.columns(2)
            
            with col_start:
                start_date_input = st.date_input(
                    "시작일",
                    value=default_start.date(),
                    min_value=min_date.date(),
                    max_value=max_date.date(),
                    key="ism_start_date"
                )
            
            with col_end:
                end_date_input = st.date_input(
                    "종료일",
                    value=max_date.date(),
                    min_value=min_date.date(),
                    max_value=max_date.date(),
                    key="ism_end_date"
                )
            
            start_date = pd.to_datetime(start_date_input)
            end_date = pd.to_datetime(end_date_input)
            
            if start_date > end_date:
                st.warning("⚠️ 시작일이 종료일보다 늦습니다. 시작일을 종료일 이전으로 설정해주세요.")
                end_date = start_date
            
            mask = (base_df[date_col] >= start_date) & (base_df[date_col] <= end_date)
            plot_df = base_df.loc[mask, [date_col] + [col for col in ism_items if col in base_df.columns]].copy()
            
            ism_fig = go.Figure()
            ism_colors = ["#146aff", "#f0580a", "#489904", "#b21c7e", "#daa900", "#18827c"]
            
            for i, col in enumerate(ism_items):
                if col in plot_df.columns:
                    ism_fig.add_trace(
                        go.Scatter(
                            x=plot_df[date_col],
                            y=plot_df[col],
                            mode="lines+markers",
                            name=col,
                            line=dict(color=ism_colors[i % len(ism_colors)])
                        )
                    )
            
            ism_fig.update_layout(
                xaxis_title="날짜",
                yaxis_title="수치",
                legend_title="항목",
                margin=dict(l=20, r=20, t=40, b=40)
            )
            
            st.plotly_chart(ism_fig, use_container_width=True)

with tab2:
    st.header("US ISM Srv. PMI")
    
    # 새로고침 버튼
    col_btn, col_info = st.columns([10, 1])
    with col_btn:
        if st.button("새로고침(30초 이내)", key="refresh_srv", help="최신 데이터를 즉시 불러옵니다"):
            load_ism_srv_data.clear()
            st.success("데이터를 새로 불러오는 중...")
            st.rerun()    
        st.caption("💡 기본적으로 캐시된 데이터를 사용합니다. 최신 데이터가 필요할 때만 새로고침 버튼을 클릭하세요.")
    
    # 캐싱된 함수 호출
    raw_df = load_ism_srv_data()
    
    # 데이터의 최신 날짜 표시
    if len(raw_df) > 0:
        latest_date = raw_df['date'].max()
        st.caption(f"📅 데이터 최신 날짜: {latest_date.strftime('%Y-%m-%d')}")
    
    # 최근 6개 날짜 기준으로 데이터 필터링 및 전치
    n_show = 6
    latest_dates = raw_df['date'].sort_values(ascending=False).head(n_show).sort_values(ascending=False)
    
    df_for_disp = raw_df.copy()
    df_for_disp = df_for_disp[df_for_disp['date'].isin(latest_dates)].sort_values('date', ascending=False)
    df_for_disp = df_for_disp.reset_index(drop=True)
    df_for_disp_disp = df_for_disp.drop(columns=['date'])
    
    original_columns = list(df_for_disp_disp.columns)
    
    transposed = df_for_disp_disp.T
    transposed.columns = [dt.strftime('%Y.%m') for dt in df_for_disp['date']]
    transposed.index.name = None
    transposed.reset_index(inplace=True)
    transposed.rename(columns={'index': '항목'}, inplace=True)
    
    delta_cols = []
    for i in range(1, n_show):
        chg_col = f'Chg{i}'
        delta_vals = transposed.iloc[:, i + 1] - transposed.iloc[:, i]
        delta_vals = delta_vals.apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")
        delta_cols.append((chg_col, delta_vals))
        transposed[chg_col] = delta_vals
    
    date_cols = [dt.strftime('%Y.%m') for dt in df_for_disp['date']]
    chg_cols = [f'Chg{i}' for i in range(1, n_show)]
    transposed = transposed[['항목'] + date_cols + chg_cols]
    
    transposed['항목'] = pd.Categorical(transposed['항목'], categories=original_columns, ordered=True)
    transposed = transposed.sort_values('항목').reset_index(drop=True)
    
    st.subheader("미국 ISM 서비스업 PMI")
    
    gb = GridOptionsBuilder.from_dataframe(transposed)
    gb.configure_default_column(resizable=True, filter=True, sortable=True)
    
    for col in date_cols + chg_cols:
        gb.configure_column(col, cellStyle={"textAlign": "center"})
    
    def get_row_style_js():
        return JsCode("""
        function(params) {
            if (params.node.rowIndex === 0) {
                return {
                    'backgroundColor': '#1565c0',
                    'color': 'white',
                    'fontWeight': 'bold'
                }
            } else {
                return {
                    'fontFamily': 'inherit',
                    'paddingLeft': '20px'
                }
            }
        }
        """)
    
    indent_js = JsCode("""
    function(params) {
        if (params.node.rowIndex === 0) {
            return params.value;
        } else {
            return '\\u00A0\\u00A0' + params.value;
        }
    }
    """)
    
    gb.configure_column("항목", cellRenderer=indent_js)
    gb.configure_grid_options(getRowStyle=get_row_style_js())
    
    AgGrid(
        transposed,
        gridOptions=gb.build(),
        height=400,
        width='100%',
        fit_columns_on_grid_load=True,
        theme="streamlit",
        allow_unsafe_jscode=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        chg_cols = ["Chg1", "Chg2", "Chg3"]
        bar_colors = [
            "rgb(245,130,32)",
            "rgb(4,59,114)",
            "rgb(0,169,206)"
        ]
        
        x_vals = transposed["항목"].tolist()
        y1 = transposed[chg_cols[0]].tolist()
        y2 = transposed[chg_cols[1]].tolist()
        y3 = transposed[chg_cols[2]].tolist()
        
        chg_labels = list(transposed.columns[1:4])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=x_vals,
            y=y1,
            name=chg_labels[0],
            marker_color=bar_colors[0]
        ))
        fig.add_trace(go.Bar(
            x=x_vals,
            y=y2,
            name=chg_labels[1],
            marker_color=bar_colors[1]
        ))
        fig.add_trace(go.Bar(
            x=x_vals,
            y=y3,
            name=chg_labels[2],
            marker_color=bar_colors[2]
        ))
        
        fig.update_layout(
            barmode='group',
            xaxis_title="항목",
            yaxis_title="변화량",
            margin=dict(l=20, r=20, t=40, b=40),
            legend_title="날짜"
        )
        
        st.subheader("Change")
        st.plotly_chart(fig, use_container_width=True)
    
    srv_items = ["ISM Srv. PMI", "Business Activity", "New Orders", "Employment", "Supplier Deliveries"]
    
    try:
        base_df = raw_df.copy()
    except NameError:
        base_df = None
    
    if base_df is not None:
        date_col_candidates = [col for col in base_df.columns if 'date' in col.lower() or '날짜' in col]
        if len(date_col_candidates) > 0:
            date_col = date_col_candidates[0]
        else:
            date_col = base_df.columns[0]
        
        base_df[date_col] = pd.to_datetime(base_df[date_col])
        
        min_date = base_df[date_col].min()
        max_date = base_df[date_col].max()
        default_start = pd.to_datetime("2023-01-01")
        default_start = max(default_start, min_date)
        
        with col2:
            st.subheader("Time Series")
            
            col_start, col_end = st.columns(2)
            
            with col_start:
                start_date_input = st.date_input(
                    "시작일",
                    value=default_start.date(),
                    min_value=min_date.date(),
                    max_value=max_date.date(),
                    key="srv_start_date"
                )
            
            with col_end:
                end_date_input = st.date_input(
                    "종료일",
                    value=max_date.date(),
                    min_value=min_date.date(),
                    max_value=max_date.date(),
                    key="srv_end_date"
                )
            
            start_date = pd.to_datetime(start_date_input)
            end_date = pd.to_datetime(end_date_input)
            
            if start_date > end_date:
                st.warning("⚠️ 시작일이 종료일보다 늦습니다. 시작일을 종료일 이전으로 설정해주세요.")
                end_date = start_date
            
            mask = (base_df[date_col] >= start_date) & (base_df[date_col] <= end_date)
            plot_df = base_df.loc[mask, [date_col] + [col for col in srv_items if col in base_df.columns]].copy()
            
            srv_fig = go.Figure()
            srv_colors = ["#146aff", "#f0580a", "#489904", "#b21c7e", "#daa900"]
            
            for i, col in enumerate(srv_items):
                if col in plot_df.columns:
                    srv_fig.add_trace(
                        go.Scatter(
                            x=plot_df[date_col],
                            y=plot_df[col],
                            mode="lines+markers",
                            name=col,
                            line=dict(color=srv_colors[i % len(srv_colors)])
                        )
                    )
            
            srv_fig.update_layout(
                xaxis_title="날짜",
                yaxis_title="수치",
                legend_title="항목",
                margin=dict(l=20, r=20, t=40, b=40)
            )
            
            st.plotly_chart(srv_fig, use_container_width=True)
            
with tab3:
    st.header("Fractal Dimension Trading Analysis")
    
    # 새로고침 버튼
    col_btn, col_info = st.columns([10, 1])
    with col_btn:
        if st.button("새로고침(3분 이내)", key="refresh_fds", help="최신 데이터를 즉시 불러옵니다"):
            load_and_analyze_data.clear()
            run_analysis.clear()
            st.success("데이터를 새로 불러오는 중...")
            st.rerun()    
        st.caption("💡 기본적으로 캐시된 데이터를 사용합니다. 최신 데이터가 필요할 때만 새로고침 버튼을 클릭하세요.")
    
    # 메인 실행
    with st.spinner("데이터를 로딩하고 분석을 실행하는 중..."):
        raw_df = load_and_analyze_data()
        selected_summary_df, selected_timeseries_results, selected_detail_results = run_analysis(raw_df)
    
    # 데이터의 최신 날짜 표시
    if len(raw_df) > 0:
        latest_date = raw_df['date'].max()
        st.caption(f"📅 데이터 최신 날짜: {latest_date.strftime('%Y-%m-%d')}")
    
    # 요약 테이블 표시
    today = pd.Timestamp.today().normalize()
    today_str = today.strftime('%Y-%m-%d')
    
    case_count = len(selected_summary_df) if len(selected_summary_df) > 0 else 0
    st.subheader(f"(1) 현재 모니터링 중인 '지표/경우의 수'에 대한 2015년 이후의 Trading 성과 ({case_count}개 경우의 수)")
    
    if len(selected_summary_df) > 0:
        display_df = selected_summary_df.copy()
        display_df = display_df.rename(columns={
            'analysis_column': '분석지표',
            'analysis_window': '분석(FD계산)일',
            'slope_threshold': '직전기울기요건',
            'trading_period': '트레이딩기간',
            'fd_level_threshold': 'FD기준',
            'fd_level_lookback': 'FD Lookback',
            'slope_metric': 'Slope Metric',
            'perf_metric': 'Perf Metric',
            'trade_count': 'Signal발생횟수',
            'hit_ratio': 'Hit Ratio',
            'avg_perf': '평균성과'
        })
        
        display_df = display_df.drop(columns=['Slope Metric', 'Perf Metric', 'FD Lookback'], errors='ignore')
        
        if 'Hit Ratio' in display_df.columns:
            display_df['Hit Ratio'] = display_df['Hit Ratio'].apply(
                lambda x: f"{x * 100:.1f}%" if pd.notna(x) else ""
            )
        if '평균성과' in display_df.columns:
            display_df['평균성과'] = display_df['평균성과'].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else ""
            )
        if 'FD기준' in display_df.columns:
            display_df['FD기준'] = display_df['FD기준'].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else ""
            )
        
        gb = GridOptionsBuilder.from_dataframe(display_df)
        gb.configure_default_column(
            resizable=True,
            sortable=True,
            filterable=True
        )
        
        for col in display_df.columns:
            gb.configure_column(
                col,
                cellStyle={"textAlign": "center", "display": "flex", "justifyContent": "center",
                           "alignItems": "center"},
                headerClass="ag-center-header"
            )
        
        gb.configure_pagination(
            enabled=True,
            paginationAutoPageSize=False,
            paginationPageSize=20
        )
        gb.configure_selection('single')
        grid_options = gb.build()
        
        st.markdown("""
        <style>
        .ag-cell {
            text-align: center !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        .ag-header-cell,
        .ag-center-header {
            text-align: center !important;
        }
        .ag-header-cell-label {
            justify-content: center !important;
            width: 100% !important;
            display: flex !important;
            align-items: center !important;
            text-align: center !important;
        }
        .ag-header-cell-text {
            margin: 0 auto !important;
            text-align: center !important;
        }
        div[class*="ag-header-cell"] {
            text-align: center !important;
        }
        div[class*="ag-header-cell-label"] {
            justify-content: center !important;
            text-align: center !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        grid_response = AgGrid(
            display_df,
            gridOptions=grid_options,
            height=400,
            width='100%',
            theme='streamlit',
            allow_unsafe_jscode=True
        )
        
        st.markdown("""
        <script>
        function centerHeaders() {
            var headers = document.querySelectorAll('.ag-header-cell-label');
            headers.forEach(function(header) {
                header.style.justifyContent = 'center';
                header.style.textAlign = 'center';
                header.style.display = 'flex';
                header.style.alignItems = 'center';
            });
            var headerTexts = document.querySelectorAll('.ag-header-cell-text');
            headerTexts.forEach(function(text) {
                text.style.margin = '0 auto';
                text.style.textAlign = 'center';
            });
        }
        setTimeout(centerHeaders, 500);
        setTimeout(centerHeaders, 1000);
        setTimeout(centerHeaders, 2000);
        </script>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="font-size: 0.85em; color: #666;">
        지표에 따라 기울기요건과 평균성과는 %와 Profit으로 표기되었습니다.(예. 환율은 %, 금리는 Profit)<br>    
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("분석 결과가 없습니다.")
    
    # 각 지표별 상태 표시
    st.markdown("---")
    st.subheader("(2) 지표별 현재 상태")
    
    indicator_order = ['USDKRW', 'EURKRW', 'JPYKRW', 'INRKRW', 'RMBKRW', 'AUDKRW',
                       'US10', 'Crv_2_10', 'Crv_2_30', 'SPR_HY', 'IGHY', 'DXY', 'SPX']
    
    indicator_status = {}
    
    for cfg_key, trades_df in selected_detail_results.items():
        if len(trades_df) == 0:
            continue
        
        col, aw, st_val, tp, fd_th, fd_lb, slope_m, perf_m = cfg_key
        
        if col not in indicator_status:
            indicator_status[col] = {
                'cases': [],
                'order': indicator_order.index(col) if col in indicator_order else 999
            }
        
        latest_signal_for_case = None
        
        for _, trade in trades_df.iterrows():
            signal_date = pd.to_datetime(trade['시그널발생일'])
            entry_date = pd.to_datetime(trade['트레이딩진입일'])
            end_date = pd.to_datetime(trade['트레이딩종료일'])
            perf = trade['성과']
            trading_days = trade['트레이딩기간']
            entry_price = trade['트레이딩진입일가격']
            signal_slope = trade['시그널기울기']
            extension_flag = trade.get('연장여부', 'N')
            extension_reason = trade.get('연장사유', '')
            extension_dates_str = trade.get('연장발생일', '')
            
            if latest_signal_for_case is None or signal_date > pd.to_datetime(latest_signal_for_case['signal_date']):
                current_price = None
                for ts_cfg_key, ts_df in selected_timeseries_results.items():
                    ts_col, _, _, _, _, _, _, _ = ts_cfg_key
                    if ts_col == col:
                        date_col = ts_df.columns[0]
                        latest_data = ts_df[ts_df[date_col] <= today]
                        if len(latest_data) > 0:
                            current_price = latest_data.iloc[-1]['value']
                            break
                
                expected_end_date = entry_date + pd.Timedelta(days=tp)
                is_completed = expected_end_date < today
                
                latest_signal_for_case = {
                    'signal_date': signal_date,
                    'entry_date': entry_date,
                    'end_date': end_date,
                    'expected_end_date': expected_end_date,
                    'trading_period': tp,
                    'trading_days': trading_days,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'signal_slope': signal_slope,
                    'perf_metric': perf_m,
                    'perf': perf,
                    'is_completed': is_completed,
                    'extension_flag': extension_flag,
                    'extension_reason': extension_reason,
                    'extension_dates_str': extension_dates_str,
                    'cfg_key': cfg_key
                }
        
        if latest_signal_for_case is not None:
            indicator_status[col]['cases'].append(latest_signal_for_case)
    
    sorted_indicators = sorted(indicator_status.items(), key=lambda x: (x[1]['order'], x[0]))
    
    for indicator, status in sorted_indicators:
        if len(status['cases']) == 0:
            st.markdown(f"**{indicator}는 현재 Signal 발생 / Trading 중이 아닙니다.**")
        else:
            cases = status['cases']
            for case_idx, ls in enumerate(cases, 1):
                if len(cases) > 1:
                    indicator_name = f"{indicator}({case_idx})"
                else:
                    indicator_name = indicator
                
                signal_date_str = ls['signal_date'].strftime('%Y-%m-%d')
                
                if ls['is_completed']:
                    expected_end_date_str = ls['expected_end_date'].strftime('%Y-%m-%d')
                    trading_period = ls['trading_period']
                    perf = ls['perf']
                    
                    if pd.notna(perf):
                        if ls['perf_metric'] == 'rate':
                            perf_str = f"{perf * 100:.1f}%"
                        else:
                            perf_str = f"{perf * 100:.2f}bp"
                    else:
                        perf_str = "N/A"
                    
                    st.markdown(
                        f"**{indicator_name}의 최근 신호는 {signal_date_str}였고, {expected_end_date_str}일에 {trading_period}일의 trading이 종료되었습니다. 수익은 {perf_str}입니다.**")
                else:
                    trading_period = ls['trading_period']
                    elapsed_days = (today - ls['entry_date']).days + 1
                    is_extended = elapsed_days > trading_period
                    extension_note = ""
                    current_target_end_date = ls['expected_end_date']
                    current_target_period = trading_period
                    
                    if is_extended:
                        if ls.get('extension_dates_str', '') and ls['extension_dates_str'].strip():
                            extension_dates_list = [d.strip() for d in ls['extension_dates_str'].split(',')]
                            extension_dates_parsed = []
                            for ext_date_str in extension_dates_list:
                                try:
                                    ext_date = pd.to_datetime(ext_date_str)
                                    extension_dates_parsed.append(ext_date)
                                except:
                                    continue
                            
                            if extension_dates_parsed:
                                last_extension_date = max(extension_dates_parsed)
                                current_target_end_date = last_extension_date + pd.Timedelta(days=trading_period)
                                days_since_last_extension = (today - last_extension_date).days + 1
                                current_target_period = days_since_last_extension
                                extension_dates_display = ls['extension_dates_str']
                                
                                if ls['extension_flag'] == 'Y' and ls.get('extension_reason', ''):
                                    extension_note = f", 시그널 연장 발생 {extension_dates_display} ({ls['extension_reason']})"
                                else:
                                    extension_note = f", 시그널 연장 발생 {extension_dates_display}"
                            else:
                                extension_start_date = ls['expected_end_date']
                                extension_start_date_str = extension_start_date.strftime('%Y-%m-%d')
                                extended_days = elapsed_days - trading_period
                                
                                if ls['extension_flag'] == 'Y' and ls.get('extension_reason', ''):
                                    extension_note = f", 시그널 연장 발생 {extension_start_date_str}부터 {extended_days}일 연장됨 ({ls['extension_reason']})"
                                else:
                                    extension_note = f", 시그널 연장 발생 {extension_start_date_str}부터 {extended_days}일 연장됨"
                        else:
                            extension_start_date = ls['expected_end_date']
                            extension_start_date_str = extension_start_date.strftime('%Y-%m-%d')
                            extended_days = elapsed_days - trading_period
                            current_target_end_date = extension_start_date + pd.Timedelta(days=trading_period)
                            days_since_extension = (today - extension_start_date).days + 1
                            current_target_period = days_since_extension
                            
                            if ls['extension_flag'] == 'Y' and ls.get('extension_reason', ''):
                                extension_note = f", 시그널 연장 발생 {extension_start_date_str} ({ls['extension_reason']})"
                            else:
                                extension_note = f", 시그널 연장 발생 {extension_start_date_str}"
                    
                    current_perf = None
                    if ls['current_price'] is not None and pd.notna(ls['entry_price']) and ls['entry_price'] != 0:
                        if ls['perf_metric'] == 'rate':
                            current_perf = (ls['current_price'] - ls['entry_price']) / ls['entry_price']
                            if pd.notna(ls['signal_slope']) and ls['signal_slope'] > 0:
                                current_perf = -current_perf
                            perf_str = f"{current_perf * 100:.1f}%"
                        else:
                            current_perf = ls['current_price'] - ls['entry_price']
                            if pd.notna(ls['signal_slope']) and ls['signal_slope'] > 0:
                                current_perf = -current_perf
                            perf_str = f"{current_perf * 100:.2f}bp"
                    else:
                        perf_str = "N/A"
                    
                    if is_extended:
                        st.markdown(
                            f"**{indicator_name}의 최근 신호는 {signal_date_str}였고, 목표트레이딩일 {current_target_period}일 중 {elapsed_days}일이 경과했습니다{extension_note}. 현재 수익은 {perf_str} 입니다.**")
                    else:
                        st.markdown(
                            f"**{indicator_name}의 최근 신호는 {signal_date_str}였고, 목표트레이딩일 {trading_period}일 중 {elapsed_days}일이 경과했습니다. 현재 수익은 {perf_str} 입니다.**")
    
    # 차트 표시
    st.markdown("---")
    st.subheader("(3) 시계열 차트")
    st.text("남색은 상승 신호 / 하늘색은 하락 신호입니다.")
    
    indicator_order = ['USDKRW', 'EURKRW', 'JPYKRW', 'INRKRW', 'RMBKRW', 'AUDKRW',
                       'US10', 'Crv_2_10', 'Crv_2_30', 'SPR_HY', 'IGHY', 'DXY', 'SPX']
    
    indicator_groups = {}
    for cfg_key, ts_df in selected_timeseries_results.items():
        col, aw, st_val, tp, fd_th, fd_lb, slope_m, perf_m = cfg_key
        if col not in indicator_groups:
            indicator_groups[col] = []
        indicator_groups[col].append((cfg_key, ts_df))
    
    max_cols = max([len(cases) for cases in indicator_groups.values()]) if indicator_groups else 1
    
    for indicator in indicator_order:
        if indicator not in indicator_groups:
            continue
        
        cases = indicator_groups[indicator]
        if len(cases) == 0:
            continue
        
        st.markdown(indicator)
        cols = st.columns(max_cols)
        
        for idx, (cfg_key, ts_df) in enumerate(cases):
            col, aw, st_val, tp, fd_th, fd_lb, slope_m, perf_m = cfg_key
            
            with cols[idx]:
                title = f"aw={aw}, st={st_val}, tp={tp}, fd_th={fd_th}, fd_lb={fd_lb}"
                st.markdown(f"**{title}**")
                
                ts = ts_df.copy()
                date_col = ts.columns[0]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ts[date_col],
                    y=ts['value'],
                    mode='lines',
                    name='Value',
                    line=dict(width=2, color='rgb(245,130,32)'),
                    showlegend=False
                ))
                
                signal_mask = ts['signal여부'] == 1
                if signal_mask.any():
                    signal_data = ts.loc[signal_mask]
                    
                    negative_mask = signal_data['signal_slope_sign'] == -1
                    if negative_mask.any():
                        fig.add_trace(go.Scatter(
                            x=signal_data.loc[negative_mask, date_col],
                            y=signal_data.loc[negative_mask, 'value'],
                            mode='markers',
                            name='Signal (-)',
                            marker=dict(size=8, color='rgb(4,59,114)', symbol='circle'),
                            showlegend=False
                        ))
                    
                    positive_mask = signal_data['signal_slope_sign'] == 1
                    if positive_mask.any():
                        fig.add_trace(go.Scatter(
                            x=signal_data.loc[positive_mask, date_col],
                            y=signal_data.loc[positive_mask, 'value'],
                            mode='markers',
                            name='Signal (+)',
                            marker=dict(size=8, color='rgb(0,169,206)', symbol='circle'),
                            showlegend=False
                        ))
                
                fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Value',
                    height=350,
                    hovermode='x unified',
                    font=dict(size=10),
                    margin=dict(l=40, r=20, t=20, b=40)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        for idx in range(len(cases), max_cols):
            with cols[idx]:
                st.empty()








