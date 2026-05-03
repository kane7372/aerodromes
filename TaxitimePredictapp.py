import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="❄️ ATD-RAM 예측 랩", layout="wide")

# ==========================================
# 1. 데이터 로드
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_parquet('data/processed/ATD_RAM_Master.parquet')
        target_col = 'Target_ATD_RAM'
        drop_from_train = ['Year', 'FLT', 'RAM_Datetime', target_col]
        available_features = [c for c in df.columns if c not in drop_from_train]
        return df, available_features, target_col
    except Exception as e:
        return None, None, None

master_df, available_features, target_col = load_data()
if master_df is None:
    st.error("🚨 `data/processed/ATD_RAM_Master.parquet` 파일이 없습니다. `processor.py`를 먼저 실행해주세요.")
    st.stop()

# ==========================================
# 2. 사이드바 컨트롤러 (강력한 필터링 장착!)
# ==========================================
st.sidebar.header("🎛️ 학습 및 시뮬레이션 세팅")

learning_mode = st.sidebar.radio(
    "학습 모드 선택 (Speed vs Accuracy)",
    ["🚀 빠른 분석 (XGBoost 단일)", "🎯 영혼 끌어모으기 (Stacking)"]
)

# 🌟 데이터 정밀 필터링 스위치 구역
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 데이터 정밀 필터링 (Data Filters)")

remove_outliers = st.sidebar.toggle("🚨 3-Sigma 극단치(대규모 지연) 제외", value=True, 
                                    help="현업 기준 240분 등 통계적 극단치를 제외하고 평균적인 성능을 높일지 선택합니다.")

cargo_filter = st.sidebar.radio("✈️ 운항편 타입", ["전체 (All)", "여객기만 (Passenger)", "화물기만 (Cargo)"])

if 'STS' in master_df.columns:
    available_sts = master_df['STS'].unique().tolist()
    selected_sts = st.sidebar.multiselect("📌 운항 상태 (STS)", available_sts, default=available_sts)
else:
    selected_sts = []

if 'Snow_Phase' in master_df.columns:
    available_phases = master_df['Snow_Phase'].unique().tolist()
    default_phases = [p for p in available_phases if 'Clear' not in p] 
    selected_phases = st.sidebar.multiselect("❄️ 강설 라이프사이클 (Snow Phase)", available_phases, default=default_phases)
else:
    selected_phases = []

st.sidebar.markdown("---")
n_trials = st.sidebar.slider("Optuna 탐색 횟수", 10, 100, 30, 10)

exclude_from_train = ['Year', 'FLT', 'RAM_Datetime', target_col, 'Snow_Phase', 'STS']
trainable_features = [c for c in master_df.columns if c not in exclude_from_train]

selected_features = st.sidebar.multiselect("⚙️ 학습 변수 (Feature Selection)", trainable_features, default=trainable_features)
start_training = st.sidebar.button("🚀 모델 학습 시작", type="primary", use_container_width=True)


# ==========================================
# 2-5. 필터링 적용 로직
# ==========================================
def apply_filters(df):
    filtered_df = df.copy()
    
    if remove_outliers:
        mean_delay, std_delay = filtered_df[target_col].mean(), filtered_df[target_col].std()
        threshold = max(mean_delay + (3 * std_delay), 240.0)
        filtered_df = filtered_df[filtered_df[target_col] <= threshold]
        
    if cargo_filter == "여객기만 (Passenger)":
        filtered_df = filtered_df[filtered_df['Is_Cargo'] == 0]
    elif cargo_filter == "화물기만 (Cargo)":
        filtered_df = filtered_df[filtered_df['Is_Cargo'] == 1]
        
    if selected_sts:
        filtered_df = filtered_df[filtered_df['STS'].isin(selected_sts)]
        
    if selected_phases:
        filtered_df = filtered_df[filtered_df['Snow_Phase'].isin(selected_phases)]
        
    return filtered_df

current_df = apply_filters(master_df)

# ==========================================
# 3. 모델 학습 함수
# ==========================================
def run_training(df, features, mode, trials):
    X = df[features]
    y = np.log1p(df[target_col])
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42, shuffle=False)
    
    # XGBoost 튜닝
    def xgb_obj(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 1500, step=500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'objective': 'reg:squarederror', 'random_state': 42, 'n_jobs': -1
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        return mean_absolute_error(np.expm1(y_valid), np.expm1(model.predict(X_valid)))

    study_xgb = optuna.create_study(direction='minimize')
    study_xgb.optimize(xgb_obj, n_trials=trials)
    xgb_best = xgb.XGBRegressor(**study_xgb.best_params, objective='reg:squarederror', random_state=42, n_jobs=-1)
    xgb_best.fit(X_train_full, y_train_full)
    
    final_model_name = "XGBoost 단일"
    
    if mode == "🚀 빠른 분석 (XGBoost 단일)":
        final_preds_log = xgb_best.predict(X_test)
        meta_model = None
        
    else: # 스태킹 모드
        def lgb_obj(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 1500, step=500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'objective': 'regression', 'random_state': 42, 'n_jobs': -1, 'verbose': -1
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
            return mean_absolute_error(np.expm1(y_valid), np.expm1(model.predict(X_valid)))

        study_lgb = optuna.create_study(direction='minimize')
        study_lgb.optimize(lgb_obj, n_trials=trials)
        lgb_best = lgb.LGBMRegressor(**study_lgb.best_params, objective='regression', random_state=42, n_jobs=-1, verbose=-1)
        lgb_best.fit(X_train_full, y_train_full)
        
        xgb_val = xgb_best.predict(X_valid)
        lgb_val = lgb_best.predict(X_valid)
        meta_model = LinearRegression(positive=True)
        meta_model.fit(pd.DataFrame({'XGB': xgb_val, 'LGBM': lgb_val}), y_valid)
        
        xgb_test = xgb_best.predict(X_test)
        lgb_test = lgb_best.predict(X_test)
        final_preds_log = meta_model.predict(pd.DataFrame({'XGB': xgb_test, 'LGBM': lgb_test}))
        final_model_name = "Stacking 앙상블"

    final_preds = np.expm1(final_preds_log)
    if 'Physical_Min_Taxi' in X_test.columns:
        final_preds = np.maximum(final_preds, X_test['Physical_Min_Taxi'].values)
    y_test_real = np.expm1(y_test)
    
    res = {
        'Model': final_model_name,
        'RMSE': np.sqrt(mean_squared_error(y_test_real, final_preds)),
        'MAE': mean_absolute_error(y_test_real, final_preds),
        'R2': r2_score(y_test_real, final_preds)
    }
    
    if meta_model is not None:
        res['XGB_W'] = meta_model.coef_[0]
        res['LGB_W'] = meta_model.coef_[1]
        
    return xgb_best, meta_model, res, y_test_real, final_preds, X_test

# ==========================================
# 4. 화면 구성 (탭 부활!)
# ==========================================
st.title("📊 ATD-RAM 예측 랩 (Lab)")
st.info(f"💡 현재 설정된 필터 기준 데이터: **총 {len(current_df):,} 건** (전체 {len(master_df):,} 건)")

# 🌟 탭 생성
tab1, tab2, tab3, tab4 = st.tabs(["📊 모델 평가", "🧠 SHAP 분석", "🔗 다중공선성(VIF)", "🎯 핀셋 튜닝"])

with tab1:
    if start_training:
        if len(selected_features) < 5:
            st.warning("변수를 최소 5개 이상 선택해주세요!")
        else:
            with st.spinner(f"{learning_mode} 진행 중... (Optuna {n_trials}회 탐색)"):
                xgb_model, meta_model, metrics, y_actual, y_pred, X_test = run_training(current_df, selected_features, learning_mode, n_trials)
                
                # 탭 이동을 위해 세션 저장
                st.session_state['xgb_model'] = xgb_model
                st.session_state['meta_model'] = meta_model
                st.session_state['test_actual'] = y_actual
                st.session_state['test_pred'] = y_pred
                st.session_state['X_test'] = X_test
                st.session_state['mode'] = learning_mode
                st.session_state['selected_features'] = selected_features
            
            st.success(f"✅ {metrics['Model']} 학습 완료!")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("R² Score", f"{metrics['R2']:.4f}")
            c2.metric("MAE (평균 오차)", f"{metrics['MAE']:.2f} 분")
            c3.metric("RMSE", f"{metrics['RMSE']:.2f} 분")
            
            if meta_model is not None:
                st.info(f"⚖️ **스태킹 가중치** - XGBoost: {metrics['XGB_W']:.3f} | LightGBM: {metrics['LGB_W']:.3f}")
                
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(x=y_actual, y=y_pred, alpha=0.5, ax=ax)
            ax.plot([0, max(y_actual)], [0, max(y_actual)], 'r--', lw=2)
            ax.set_xlabel('실제 지연 (분)')
            ax.set_ylabel('예측 지연 (분)')
            ax.set_title(f'실제 vs 예측 ({metrics["Model"]})')
            st.pyplot(fig)
    else:
        st.info("👈 사이드바에서 세팅을 마치고 '학습 시작'을 눌러주세요.")

with tab2:
    st.subheader("🧠 SHAP 분석 (개발 대기 중...)")
    
with tab3:
    st.subheader("🔗 다중공선성(VIF) 검사 (개발 대기 중...)")

with tab4:
    st.subheader("🎯 핀셋 튜닝 해부 (개발 대기 중...)")
