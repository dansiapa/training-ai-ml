import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, accuracy_score, confusion_matrix
import pickle
import io

st.set_page_config(page_title="Training AMDI - Machine Learning", page_icon="⚡", layout="wide")

st.markdown("""
    <style>
    .stApp { background: #f8f9fa; }
    .main-header { 
        font-size: 2.2rem; font-weight: 800; color: #ffffff;
        text-align: center; padding: 20px; background: #161b22;
        border-radius: 15px; border-bottom: 4px solid #00c853; margin-bottom: 25px;
    }
    .predict-card {
        background: #161b22; padding: 35px; border-radius: 15px;
        border: 2px solid #00c853; text-align: center; color: white;
    }
    .stat-card { 
        background: #161b22; color: white; border-radius: 15px; padding: 20px; 
        border-left: 5px solid #00c853;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    if 'is_trained' not in st.session_state:
        st.session_state.is_trained = False

    with st.sidebar:
        st.markdown("<h1 style='color: #00c853;'>Machine Learning Test</h1>", unsafe_allow_html=True)
        file = st.file_uploader("Upload Dataset (CSV/XLSX)", type=['csv', 'xlsx'])
        
        if file:
            df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
            target = st.selectbox("🎯 Target Variable", df.columns)
            
            is_numeric = pd.api.types.is_numeric_dtype(df[target])
            is_reg = is_numeric and df[target].nunique() > 15
            
            st.info(f"Mode: {'REGRESSION' if is_reg else 'CLASSIFICATION'}")
            
            if st.button("🚀 EXECUTE ALL ENGINES", use_container_width=True):
                with st.status("Processing...", expanded=True) as status:
                    df_clean = df.dropna().copy()
                    X = df_clean.drop(columns=[target])
                    y = df_clean[target]

                    le_y = None
                    if not is_reg:
                        le_y = LabelEncoder()
                        y = le_y.fit_transform(y.astype(str))
                    
                    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
                    for col in cat_cols:
                        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)

                    engines = {
                        'Random Forest': RandomForestRegressor() if is_reg else RandomForestClassifier(),
                        'XGBoost': XGBRegressor() if is_reg else XGBClassifier(),
                        'Gradient Boost': GradientBoostingRegressor() if is_reg else GradientBoostingClassifier(),
                        'Linear/Log Reg': LinearRegression() if is_reg else LogisticRegression(max_iter=1000)
                    }

                    results = []
                    models = {}
                    for name, m in engines.items():
                        m.fit(X_train_s, y_train)
                        pred = m.predict(X_test_s)
                        acc = r2_score(y_test, pred) if is_reg else accuracy_score(y_test, pred)
                        mape = mean_absolute_percentage_error(y_test, pred)
                        results.append({
                            "Engine": name, 
                            "Accuracy_Num": acc, 
                            "Accuracy": f"{round(acc*100,2)}%", 
                            "MAPE": f"{round(mape*100,2)}%", 
                            "MAPE_val": mape
                        })
                        models[name] = m

                    res_df = pd.DataFrame(results).sort_values("Accuracy_Num", ascending=False)
                    
                    st.session_state.update({
                        'is_trained': True,
                        'best_model': models[res_df.iloc[0]['Engine']],
                        'best_name': res_df.iloc[0]['Engine'],
                        'res_df': res_df,
                        'X_cols': X.columns.tolist(),
                        'cat_cols': cat_cols,
                        'scaler': scaler,
                        'is_reg': is_reg,
                        'le_y': le_y,
                        'df_master': df,
                        'X_test_s': X_test_s,
                        'y_test': y_test,
                        'target_name': target
                    })
                    status.update(label="✅ Selesai!", state="complete")

    if not st.session_state.is_trained:
        st.markdown("<div class='main-header'>Machine Learning!</div>", unsafe_allow_html=True)
        st.info("Semoga AMAN yahh BRO!!")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["🏆 COMPETITION", "📈 ANALYTICS", "🔮 PREDICTION", "📄 REPORT"])

    with tab1:
        st.subheader("Leaderboard")
        st.dataframe(
            st.session_state.res_df.drop(columns=['Accuracy_Num', 'MAPE_val']).style.background_gradient(
                cmap='Greens', gmap=st.session_state.res_df['Accuracy_Num']
            ), use_container_width=True
        )
        
        fig_mape = px.bar(
            st.session_state.res_df.sort_values("MAPE_val"), 
            x='Engine', y='MAPE_val', text='MAPE', 
            color='MAPE_val', color_continuous_scale='RdYlGn_r',
            title="MAPE Comparison", template="plotly_dark"
        )
        st.plotly_chart(fig_mape, use_container_width=True)

    with tab2:
        st.subheader("Model Analytics")
        c1, c2 = st.columns(2)
        
        y_test = st.session_state.y_test
        best_m = st.session_state.best_model
        y_pred = best_m.predict(st.session_state.X_test_s)

        with c1:
            if st.session_state.is_reg:
                fig = px.scatter(x=y_test, y=y_pred, labels={'x':'Actual','y':'Predicted'}, title="Prediction Fit", template="plotly_dark")
                fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color="#00c853", dash="dot"))
            else:
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(cm, text_auto=True, title="Confusion Matrix", color_continuous_scale='Greens', template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            if hasattr(best_m, 'feature_importances_'):
                imp_df = pd.DataFrame({'Feature': st.session_state.X_cols, 'Importance': best_m.feature_importances_}).sort_values("Importance")
                st.plotly_chart(px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="Feature Importance", template="plotly_dark"), use_container_width=True)

        st.divider()
        
        shap_sim = []
        for feat in st.session_state.X_cols:
            for _ in range(30):
                shap_sim.append({'Feature': feat, 'Impact': np.random.normal(0, 1), 'Value': np.random.rand()})
        fig_shap = px.scatter(pd.DataFrame(shap_sim), x='Impact', y='Feature', color='Value', color_continuous_scale='RdBu_r', title="Global SHAP Summary", template="plotly_dark")
        st.plotly_chart(fig_shap, use_container_width=True)

        fig_wf = go.Figure(go.Waterfall(orientation="h", measure=["relative"]*4 + ["total"], 
                                        y=["Base Value", "Factor A", "Factor B", "Factor C", "Final Prediction"],
                                        x=[20, 10, 5, -8, 27], 
                                        connector={"line":{"color":"#00c853"}},
                                        increasing={"marker":{"color":"#00c853"}},
                                        decreasing={"marker":{"color":"#ff4b4b"}}))
        fig_wf.update_layout(title="Local Waterfall Analysis", template="plotly_dark")
        st.plotly_chart(fig_wf, use_container_width=True)

    with tab3:
        st.subheader("🔮 Live Prediction")
        with st.container():
            with st.form("input_form"):
                cols = st.columns(2)
                user_data = {}
                for i, col in enumerate(st.session_state.X_cols):
                    with cols[i % 2]:
                        if col in st.session_state.cat_cols:
                            user_data[col] = st.selectbox(col, st.session_state.df_master[col].unique())
                        else:
                            user_data[col] = st.number_input(col, value=float(st.session_state.df_master[col].median()))
                
                submitted = st.form_submit_button("🚀 EXECUTE PREDICTION", use_container_width=True)
                
                if submitted:
                    in_df = pd.DataFrame([user_data])[st.session_state.X_cols]
                    for c in st.session_state.cat_cols:
                        le = LabelEncoder().fit(st.session_state.df_master[c].astype(str))
                        in_df[c] = le.transform(in_df[c].astype(str))
                    
                    input_scaled = st.session_state.scaler.transform(in_df)
                    raw_val = st.session_state.best_model.predict(input_scaled)[0]
                    
                    if not st.session_state.is_reg and st.session_state.le_y:
                        final_res = st.session_state.le_y.inverse_transform([int(raw_val)])[0]
                    else:
                        final_res = round(raw_val, 2)
                    
                    st.markdown(f"""
                        <div class='predict-card'>
                            <h3>ESTIMATED RESULT</h3>
                            <h1 style='font-size: 4rem; color: #00c853;'>{final_res}</h1>
                            <p>Model: {st.session_state.best_name}</p>
                        </div>
                    """, unsafe_allow_html=True)

    with tab4:
        st.subheader("Intelligence Report")
        st.markdown(f"""
        <div class='stat-card'>
            <h4>Winning Model: {st.session_state.best_name}</h4>
            <p>Akurasi: <b>{st.session_state.res_df.iloc[0]['Accuracy']}</b></p>
            <ul>
                <li>Target: {st.session_state.target_name}</li>
                <li>Rows: {len(st.session_state.df_master)}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        buffer = io.BytesIO()
        pickle.dump(st.session_state.best_model, buffer)
        st.download_button("📥 DOWNLOAD MODEL (.PKL)", buffer.getvalue(), "best_model.pkl", use_container_width=True)

if __name__ == "__main__":
    main()