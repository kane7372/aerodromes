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
import time

warnings.filterwarnings('ignore')

# 폰트 깨짐 방지를 위해 영문 라벨 사용 (한글 폰트 설정 제거)
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="❄️ ATD-RAM 예측 랩", layout="wide")

# ==========================================
# 0. 사이드바 - 데이터 업로드 구역
# ==========================================
st.sidebar.header("📁 데이터 업로드")
st.sidebar.info("코랩에서 만든 `ATD_RAM_Master.parquet` 파일을 여기에 올려주세요!")
uploaded_file = st.sidebar.file_uploader("데이터 파일 업로드", type=['parquet'])

# ==========================================
# 1. 데이터 로드 (업로드된 파일 읽기)
# ==========================================
@st.cache_data
def load_data(file):
    try:
        df = pd.read_parquet(file)
        target_col = 'Target_ATD_RAM'
        drop_from_train = ['Year', 'FLT', 'RAM_Datetime', target_col]
        available_features = [c for c in df.columns if c not in drop_from_train]
        return df, available_features, target_col
    except Exception as e:
        return None, None, None

if uploaded_file is None:
    st.title("📊 ATD-RAM 예측 대시보드")
    st.warning("👈 사이드바에서 데이터 파일(`.parquet`)을 먼저 업로드해주세요!")
    st.stop() 

master_df, available_features, target_col = load_data(uploaded_file)

if master_df is None:
    st.error("🚨 파일을 읽는 중 오류가 발생했습니다. 정상적인 Parquet 파일인지 확인해주세요.")
    st.stop()

# ==========================================
# 2. 사이드바 컨트롤러 (강력한 필터링 장착!)
# ==========================================
st.sidebar.header("🎛️ 학습 및 시뮬레이션 세팅")

learning_mode = st.sidebar.radio(
    "학습 모드 선택 (Speed vs Accuracy)",
    ["🚀 빠른 분석 (XGBoost 단일", "🎯 영혼 끌어모으기 (Stacking)"]
)

# 🌟 데이터 정밀 필터링 스위치 구역
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 데이터 정밀 필터링 (Data Filters)")

remove_outliers = st.sidebar.toggle("🚨 3-Sigma 극단치(대규모 지연) 제외", value=True)

# 1. 기상 현상 (Weather Type) 필터
if 'Weather_Type' in master_df.columns:
    available_weather = master_df['Weather_Type'].dropna().unique().tolist()
    selected_weather = st.sidebar.multiselect("🌤️ 기상 현상 (Weather Type)", available_weather, default=available_weather)
else:
    selected_weather = []

# 2. 강설 페이즈 필터
if 'Snow_Phase' in master_df.columns:
    available_phases = master_df['Snow_Phase'].dropna().unique().tolist()
    default_phases = [p for p in available_phases if 'Clear' not in p] 
    selected_phases = st.sidebar.multiselect("❄️ 강설 라이프사이클 (Snow Phase)", available_phases, default=default_phases)
else:
    selected_phases = []

# 3. 여객/화물 구분 (NAT 열 기반)
if 'NAT' in master_df.columns:
    available_nats = master_df['NAT'].dropna().unique().tolist()
    selected_nats = st.sidebar.multiselect("✈️ 운항편 타입 (NAT)", available_nats, default=available_nats)
else:
    selected_nats = []

# 4. 운항 상태 (STS) 필터
if 'STS' in master_df.columns:
    available_sts = master_df['STS'].dropna().unique().tolist()
    selected_sts = st.sidebar.multiselect("📌 운항 상태 (STS)", available_sts, default=available_sts)
else:
    selected_sts = []

# ==========================================
# 🌟 [NEW] 학습 및 평가 대상 '연도(Year)' 자유 조립기
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("📅 데이터 연도(Year) 조립기")

if 'Year' in master_df.columns:
    available_years = sorted(master_df['Year'].dropna().unique().astype(int).tolist())
    
    train_years = st.sidebar.multiselect(
        "🧠 학습(Train)에 사용할 연도 선택", 
        available_years, 
        default=available_years,
        help="선택한 연도의 데이터만 모아서 AI를 학습시킵니다."
    )
    
    test_mode = st.sidebar.radio(
        "🎯 평가(Test) 데이터 추출 방식",
        [
            "학습 데이터 내에서 10% 자동 분할 (기본)", 
            "특정 연도를 통째로 평가(Test)에 배정",
            "선택한 연도 전체를 학습하고 자체 평가 (In-Sample)"
        ],
        help="'자동 분할'은 학습 데이터의 마지막 10%를 씁니다. '통째로 평가'는 아예 본 적 없는 연도로 백테스트할 때 쓰며, '자체 평가'는 Test 데이터 없이 학습한 데이터를 그대로 다시 풀어보는 방식입니다."
    )
    
    if test_mode == "특정 연도를 통째로 평가(Test)에 배정":
        target_test_years = st.sidebar.multiselect(
            "실전 예측(Test)할 연도 선택",
            available_years,
            default=[available_years[-1]]
        )
    else:
        target_test_years = []
else:
    train_years = []
    test_mode = "학습 데이터 내에서 10% 자동 분할 (기본)"
    target_test_years = []

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Optuna 튜닝 설정")
n_trials = st.sidebar.slider("Optuna 최대 탐색 횟수", 10, 100, 30, 10)
early_stop_rounds = st.sidebar.number_input("조기 종료 브레이크 (0=끄기)", min_value=0, max_value=50, value=10, step=1)

exclude_from_train = ['Year', 'FLT', 'RAM_Datetime', target_col, 'Snow_Phase', 'STS', 'NAT', 'Weather_Type']
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
        
    if selected_weather:
        filtered_df = filtered_df[filtered_df['Weather_Type'].isin(selected_weather)]
        
    if selected_phases:
        filtered_df = filtered_df[filtered_df['Snow_Phase'].isin(selected_phases)]

    if selected_nats:
        filtered_df = filtered_df[filtered_df['NAT'].isin(selected_nats)]
        
    if selected_sts:
        filtered_df = filtered_df[filtered_df['STS'].isin(selected_sts)]
        
    return filtered_df

current_df = apply_filters(master_df)

# ==========================================
# 🌟 Optuna 스트림릿 전용 콜백 클래스
# ==========================================
class StreamlitOptunaCallback:
    def __init__(self, n_trials, early_stopping_rounds, model_name, pbar, status_text):
        self.n_trials = n_trials
        self.early_stopping_rounds = early_stopping_rounds
        self.model_name = model_name
        self.pbar = pbar
        self.status_text = status_text
        self.best_score = float('inf')
        self.no_improvement_count = 0

    def __call__(self, study, trial):
        current_trial = trial.number + 1
        progress = min(current_trial / self.n_trials, 1.0)
        
        if trial.value is not None:
            if trial.value < self.best_score:
                self.best_score = trial.value
                self.no_improvement_count = 0
                improvement_flag = "✨ **최고 기록 갱신!**"
            else:
                self.no_improvement_count += 1
                improvement_flag = ""

            self.pbar.progress(progress)
            self.status_text.markdown(f"**[{self.model_name}]** 진행: {current_trial} / {self.n_trials} | 현재 최고 MAE: `{self.best_score:.4f}` | 정체 카운트: {self.no_improvement_count}/{self.early_stopping_rounds} {improvement_flag}")

            if self.early_stopping_rounds > 0 and self.no_improvement_count >= self.early_stopping_rounds:
                self.status_text.warning(f"🛑 **{self.model_name} 조기 종료:** {self.early_stopping_rounds}회 연속 개선이 없어 튜닝을 멈춥니다.")
                study.stop()

# ==========================================
# 3. 모델 학습 함수
# ==========================================
def run_training(df, features, mode, trials, early_stop_rounds, pbar, status_text, train_years, test_mode, target_test_years):
    X_all = df[features]
    y_all = np.log1p(df[target_col])
    year_col = df['Year'].astype(int)
    
    # 🌟 연도 분할 로직 적용
    if test_mode == "학습 데이터 내에서 10% 자동 분할 (기본)":
        mask = year_col.isin(train_years)
        X_selected = X_all[mask]
        y_selected = y_all[mask]
        
        if len(X_selected) < 100:
            st.error("🚨 선택한 학습 연도에 데이터가 너무 적습니다. 조건을 완화해주세요!")
            st.stop()
            
        X_train_full, X_test, y_train_full, y_test = train_test_split(X_selected, y_selected, test_size=0.1, random_state=42, shuffle=False)
        
    elif test_mode == "특정 연도를 통째로 평가(Test)에 배정":
        train_mask = year_col.isin(train_years)
        test_mask = year_col.isin(target_test_years)
        
        X_train_full = X_all[train_mask]
        y_train_full = y_all[train_mask]
        X_test = X_all[test_mask]
        y_test = y_all[test_mask]
        
        if len(X_train_full) < 50 or len(X_test) == 0:
            st.error("🚨 학습 또는 테스트 데이터가 비어있습니다. 연도를 다시 선택해주세요!")
            st.stop()
            
    else: # 선택한 연도 전체를 학습하고 자체 평가 (In-Sample)
        mask = year_col.isin(train_years)
        X_train_full = X_all[mask]
        y_train_full = y_all[mask]
        
        if len(X_train_full) < 50:
            st.error("🚨 선택한 학습 연도에 데이터가 너무 적습니다!")
            st.stop()
            
        X_test = X_train_full.copy()
        y_test = y_train_full.copy()

    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42, shuffle=False)
    
    # 1. XGBoost 튜닝
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
    xgb_callback = StreamlitOptunaCallback(trials, early_stop_rounds, "XGBoost", pbar, status_text)
    study_xgb.optimize(xgb_obj, n_trials=trials, callbacks=[xgb_callback])
    
    time.sleep(1)
    
    xgb_best = xgb.XGBRegressor(**study_xgb.best_params, objective='reg:squarederror', random_state=42, n_jobs=-1)
    xgb_best.fit(X_train_full, y_train_full)
    
    final_model_name = "XGBoost (Single)"
    
    if mode == "🚀 빠른 분석 (XGBoost 단일)":
        final_preds_log = xgb_best.predict(X_test)
        meta_model = None
        
    else: # 스태킹 모드
        pbar.progress(0)
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
        lgb_callback = StreamlitOptunaCallback(trials, early_stop_rounds, "LightGBM", pbar, status_text)
        study_lgb.optimize(lgb_obj, n_trials=trials, callbacks=[lgb_callback])
        
        lgb_best = lgb.LGBMRegressor(**study_lgb.best_params, objective='regression', random_state=42, n_jobs=-1, verbose=-1)
        lgb_best.fit(X_train_full, y_train_full)
        
        status_text.success("🎉 메타 모델(Stacking) 가중치 조율 중...")
        
        xgb_val = xgb_best.predict(X_valid)
        lgb_val = lgb_best.predict(X_valid)
        meta_model = LinearRegression(positive=True)
        meta_model.fit(pd.DataFrame({'XGB': xgb_val, 'LGBM': lgb_val}), y_valid)
        
        xgb_test = xgb_best.predict(X_test)
        lgb_test = lgb_best.predict(X_test)
        final_preds_log = meta_model.predict(pd.DataFrame({'XGB': xgb_test, 'LGBM': lgb_test}))
        final_model_name = "Stacking (Ensenble)"

    final_preds = np.expm1(final_preds_log)
    if 'Physical_Min_Taxi' in X_test.columns:
        final_preds = np.maximum(final_preds, X_test['Physical_Min_Taxi'].values)
    y_test_real = np.expm1(y_test)
    
    # 🌟 연도별 성능 리포트용 DataFrame 생성
    results_df = pd.DataFrame({
        'Year': df.loc[X_test.index, 'Year'],
        'Actual': y_test_real,
        'Pred': final_preds
    })
    
    res = {
        'Model': final_model_name,
        'RMSE': np.sqrt(mean_squared_error(y_test_real, final_preds)),
        'MAE': mean_absolute_error(y_test_real, final_preds),
        'R2': r2_score(y_test_real, final_preds),
        'Yearly_Results': results_df
    }
    
    if meta_model is not None:
        res['XGB_W'] = meta_model.coef_[0]
        res['LGB_W'] = meta_model.coef_[1]
        
    return xgb_best, meta_model, res, y_test_real, final_preds, X_test

# ==========================================
# 4. 화면 구성
# ==========================================
st.title("📊 ATD-RAM 예측 랩 (Lab)")
st.info(f"💡 현재 설정된 필터 기준 데이터: **총 {len(current_df):,} 건** (전체 {len(master_df):,} 건)")

tab1, tab2, tab3, tab4, tab5 = st.tabs([ # 👈 tab5 추가!
    "📊 모델 평가", "🧠 SHAP 분석", "🔗 다중공선성(VIF)", "🎯 핀셋 튜닝", "🚀 실시간 예측(Deploy)" # 👈 이름 추가!
])

with tab1:
    if start_training:
        if len(selected_features) < 5:
            st.warning("변수를 최소 5개 이상 선택해주세요!")
        else:
            st.markdown("### 🏃‍♂️ 실시간 튜닝 진행 상황")
            pbar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("AI가 최적의 파라미터를 찾는 중입니다..."):
                xgb_model, meta_model, metrics, y_actual, y_pred, X_test = run_training(
                    current_df, selected_features, learning_mode, n_trials, early_stop_rounds, pbar, status_text,
                    train_years, test_mode, target_test_years
                )
                
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
            c2.metric("MAE (Mean Abs Error)", f"{metrics['MAE']:.2f} Min")
            c3.metric("RMSE", f"{metrics['RMSE']:.2f} Min")
            
            if meta_model is not None:
                st.info(f"⚖️ **Stacking Weights** - XGBoost: {metrics['XGB_W']:.3f} | LightGBM: {metrics['LGB_W']:.3f}")
                
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(x=y_actual, y=y_pred, alpha=0.5, ax=ax)
            ax.plot([0, max(y_actual)], [0, max(y_actual)], 'r--', lw=2)
            ax.set_xlabel('Actual Delay (Minutes)')
            ax.set_ylabel('Predicted Delay (Minutes)')
            ax.set_title(f'Actual vs Predicted Delay ({metrics["Model"]})')
            st.pyplot(fig)
            
            # 🌟 [NEW] 연도별 세부 성능 리포트 출력
            st.markdown("---")
            st.subheader("📅 연도별 세부 성능 리포트 (Year-wise Analysis)")
            
            yearly_res = metrics['Yearly_Results']
            summary_list = []
            
            for year in sorted(yearly_res['Year'].unique()):
                y_sub = yearly_res[yearly_res['Year'] == year]
                mae = mean_absolute_error(y_sub['Actual'], y_sub['Pred'])
                rmse = np.sqrt(mean_squared_error(y_sub['Actual'], y_sub['Pred']))
                r2 = r2_score(y_sub['Actual'], y_sub['Pred'])
                summary_list.append({
                    'Year': f"{int(year)}",
                    'Count': f"{len(y_sub):,} rows",
                    'MAE (Min)': round(mae, 2),
                    'RMSE (Min)': round(rmse, 2),
                    'R2 Score': round(r2, 4)
                })
            
            st.table(pd.DataFrame(summary_list))
            st.caption("※ 평가(Test) 대상 데이터 내에 포함된 연도별 성능입니다.")
            
    else:
        st.info("👈 사이드바에서 세팅을 마치고 '학습 시작'을 눌러주세요.")

with tab2:
    st.subheader("🧠 SHAP 분석 (지연 원인 해부)")
    if 'xgb_model' in st.session_state:
        st.info("💡 **SHAP (SHapley Additive exPlanations)**: AI가 특정 항공편의 지연을 예측할 때, 어떤 변수가 지연을 늘렸고(+) 어떤 변수가 줄였는지(-) 기여도를 분석합니다.")
        
        with st.spinner("SHAP Value 계산 중입니다. 데이터가 많으면 수십 초 정도 걸릴 수 있습니다..."):
            model = st.session_state['xgb_model']
            X_test_shap = st.session_state['X_test']
            
            # 속도를 위해 최대 1000건만 샘플링하여 분석
            if len(X_test_shap) > 1000:
                X_test_shap = X_test_shap.sample(1000, random_state=42)
                
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_test_shap)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 변수 중요도 (Summary Plot)")
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                # 한글 깨짐 방지를 위해 내부 폰트 사이즈 조정
                shap.summary_plot(shap_values, X_test_shap, plot_type="bar", show=False)
                plt.title("Overall Feature Importance", fontsize=12)
                st.pyplot(fig1)
                
            with col2:
                st.markdown("#### 🐝 변수 영향도 (Beeswarm Plot)")
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                shap.plots.beeswarm(shap_values, show=False)
                plt.title("Impact on Delay (+: Increase, -: Decrease)", fontsize=12)
                st.pyplot(fig2)
                
            st.markdown("---")
            st.markdown("#### 🔍 개별 항공편 원인 분석 (Waterfall Plot)")
            sample_idx = st.number_input("분석할 항공편의 인덱스(순번)를 입력하세요 (0 ~ 999):", min_value=0, max_value=min(999, len(X_test_shap)-1), value=0)
            
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap_values[sample_idx], show=False)
            plt.title(f"Why was Flight #{sample_idx} delayed?", fontsize=12)
            st.pyplot(fig3)
    else:
        st.warning("👈 1번 탭에서 '모델 학습'을 먼저 완료해야 분석이 가능합니다!")

with tab3:
    st.subheader("🔗 다중공선성(VIF) 검사기")
    if 'X_test' in st.session_state:
        st.info("💡 **VIF (Variance Inflation Factor)**: 변수들끼리 의미가 겹치는지(상관관계) 확인합니다. VIF가 10 이상이면 다른 변수와 의미가 중복되므로 모델에서 빼는 것이 좋습니다.")
        
        with st.spinner("다중공선성을 계산 중입니다..."):
            X_vif = st.session_state['X_test'].copy()
            # 결측치나 무한대 값이 있으면 VIF 에러가 나므로 0으로 채움
            X_vif = X_vif.replace([np.inf, -np.inf], 0).fillna(0)
            
            vif_data = pd.DataFrame()
            vif_data["Feature"] = X_vif.columns
            vif_data["VIF_Score"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
            vif_data = vif_data.sort_values(by="VIF_Score", ascending=False).reset_index(drop=True)
            
            # VIF가 10 이상인 '위험' 변수와 안전한 변수 분리
            danger_vif = vif_data[vif_data["VIF_Score"] >= 10.0]
            safe_vif = vif_data[vif_data["VIF_Score"] < 10.0]
            
            c1, c2 = st.columns(2)
            with c1:
                st.error(f"🚨 제거 권장 변수 (VIF >= 10) : {len(danger_vif)} 개")
                st.dataframe(danger_vif, use_container_width=True)
            with c2:
                st.success(f"✅ 안전한 변수 (VIF < 10) : {len(safe_vif)} 개")
                st.dataframe(safe_vif, use_container_width=True)
    else:
        st.warning("👈 1번 탭에서 '모델 학습'을 먼저 완료해야 분석이 가능합니다!")

with tab4:
    st.subheader("🎯 핀셋 튜닝 (What-If 시뮬레이터)")
    if 'xgb_model' in st.session_state:
        st.info("💡 **What-If 분석**: 특정 항공편의 조건(날씨, 거리 등)을 통제센터에서 인위적으로 변경해 보았을 때, 지연 시간이 어떻게 변하는지 실시간으로 시뮬레이션합니다.")
        
        model = st.session_state['xgb_model']
        X_test_sim = st.session_state['X_test'].copy()
        
        # 특정 상황의 비행기 고르기
        row_idx = st.selectbox("✈️ 시뮬레이션 할 항공편 선택 (Test 데이터 기준)", X_test_sim.index.tolist())
        original_data = X_test_sim.loc[[row_idx]]
        
        # 원본 예측값
        orig_pred = np.expm1(model.predict(original_data))[0]
        actual_delay = st.session_state['test_actual'][X_test_sim.index.get_loc(row_idx)]
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("실제 지연 시간 (Actual)", f"{actual_delay:.1f} 분")
        col_m2.metric("AI 예측 지연 (Original)", f"{orig_pred:.1f} 분")
        
        st.markdown("---")
        st.markdown("### 🎛️ 변수 조작 패널 (값을 더블클릭해서 수정해보세요!)")
        
        # 🌟 핵심! st.data_editor를 통한 실시간 데이터 수정
        edited_data = st.data_editor(original_data, num_rows="fixed", use_container_width=True)
        
        if st.button("🚀 조작된 데이터로 다시 예측하기 (Re-Predict)"):
            new_pred = np.expm1(model.predict(edited_data))[0]
            diff = new_pred - orig_pred
            
            # 결과 출력
            if diff > 0:
                col_m3.metric("시뮬레이션 결과 (New)", f"{new_pred:.1f} 분", f"+{diff:.1f} 분 악화됨", delta_color="inverse")
            else:
                col_m3.metric("시뮬레이션 결과 (New)", f"{new_pred:.1f} 분", f"{diff:.1f} 분 단축됨!", delta_color="normal")
            
            st.success("✅ 시뮬레이션 완료! 조작한 조건에 따라 지연 시간이 위와 같이 변동됩니다.")
    else:
        st.warning("👈 1번 탭에서 '모델 학습'을 먼저 완료해야 분석이 가능합니다!")

with tab5:
    st.subheader("🚀 실시간 지연 예측기 (Live Inference)")
    if 'xgb_model' in st.session_state:
        st.info("💡 **통제센터 실무 모드**: 현재 들어온 비행 스케줄과 기상 상황을 입력하면, 학습된 챔피언 모델(XGBoost)이 즉시 예상 지연 시간을 도출합니다.")
        
        # 모델이 학습할 때 썼던 변수 목록과 데이터의 형태(타입/범위) 가져오기
        selected_features = st.session_state['selected_features']
        X_template = st.session_state['X_test']
        
        st.markdown("### 📝 운항 및 기상 조건 수동 입력")
        
        # st.form을 사용하여 값을 하나 입력할 때마다 화면이 깜빡이는 것을 방지
        with st.form("prediction_form"):
            cols = st.columns(3)
            user_inputs = {}
            
            # 선택된 변수 개수만큼 알아서 입력 칸(Widget) 생성
            for i, col_name in enumerate(selected_features):
                col = cols[i % 3]
                
                # 숫자형 변수인 경우
                if pd.api.types.is_numeric_dtype(X_template[col_name]):
                    unique_vals = X_template[col_name].dropna().unique()
                    
                    # 1. 0/1 등 범주가 5개 이하인 숫자 (예: Is_Cargo) -> 선택창(Selectbox)
                    if len(unique_vals) <= 5:
                        unique_vals = sorted(unique_vals.tolist())
                        default_val = int(X_template[col_name].mode()[0])
                        idx = unique_vals.index(default_val) if default_val in unique_vals else 0
                        user_inputs[col_name] = col.selectbox(f"🗂️ {col_name}", unique_vals, index=idx)
                        
                    # 2. 연속된 진짜 숫자 (예: 이동거리, 온도 등) -> 숫자 입력창(Number_input)
                    else:
                        min_v = float(X_template[col_name].min())
                        max_v = float(X_template[col_name].max())
                        mean_v = float(X_template[col_name].mean())
                        step = (max_v - min_v) / 100 if max_v != min_v else 0.1
                        user_inputs[col_name] = col.number_input(f"🔢 {col_name}", min_value=min_v, max_value=max_v, value=mean_v, step=step)
                
                # 문자형 변수인 경우 (텍스트) -> 선택창
                else:
                    unique_vals = X_template[col_name].dropna().unique().tolist()
                    user_inputs[col_name] = col.selectbox(f"🔠 {col_name}", unique_vals)
            
            st.markdown("---")
            submit_btn = st.form_submit_button("🔮 AI 실시간 예측 수행", type="primary", use_container_width=True)
            
        # [예측 실행 로직]
        if submit_btn:
            # 유저가 입력한 딕셔너리를 1줄짜리 데이터프레임으로 변환
            input_df = pd.DataFrame([user_inputs])
            model = st.session_state['xgb_model']
            
            with st.spinner("AI가 지연 시간을 계산하고 있습니다..."):
                # 예측 (Log로 변환해서 학습했으므로 다시 expm1으로 원복)
                pred_log = model.predict(input_df)
                pred_minutes = np.expm1(pred_log)[0]
                
                # '물리적 최소 지상 이동시간' 변수가 있다면 그 이하로는 예측하지 못하게 방어
                if 'Physical_Min_Taxi' in input_df.columns:
                    min_taxi = input_df['Physical_Min_Taxi'].values[0]
                    pred_minutes = max(pred_minutes, min_taxi)
            
            st.markdown("---")
            st.success("✅ 타겟 비행편의 ATD-RAM (주기장 출발 ~ 실제 이륙) 소요 시간 분석 완료!")
            
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("🤖 AI 예상 지연 시간", f"{pred_minutes:.1f} 분")
            
            # 현업 기준 지연 경보등(Traffic Light) 신호
            if pred_minutes <= 15:
                res_col2.success("🟢 **정상 운항 예상** (15분 이내)")
            elif pred_minutes <= 45:
                res_col2.warning("🟡 **주의 요망** (15~45분 지연 예상)")
            else:
                res_col2.error("🔴 **심각 단계** (45분 이상 대규모 지연 예상. 선제적 통제 필요!)")
                st.snow() # 폭설 지연의 느낌을 살려 눈 내리는 이펙트! ❄️
                
    else:
        st.warning("👈 1번 탭에서 과거 데이터로 '모델 학습'을 먼저 완료해야 실시간 예측이 가능합니다!")
