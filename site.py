import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Saúde Mental no Trabalho",
    page_icon="🧠",
    layout="centered",
)

@st.cache_resource
def load_artifacts():
    model    = joblib.load("modelo_osmi_final.joblib")
    encoders = joblib.load("encoders_features.joblib")
    feats    = joblib.load("features_finais.joblib")
    return model, encoders, feats

model, encoders, FEAT_COLS = load_artifacts()


QUESTIONS = [
    {
        "key": "age_bucket",
        "label": "Qual é a sua faixa etária?",
        "options": [
            ("18–24 anos", 21.0),
            ("25–34 anos", 29.0),
            ("35–44 anos", 39.0),
            ("45–54 anos", 49.0),
            ("55 anos ou mais", 57.0),
        ],
    },
    {
        "key": "gender",
        "model_key": "gender",
        "label": "Com qual gênero você se identifica?",
        "options": [
            ("Masculino", "Male"),
            ("Feminino", "Female"),
            ("Não-binário", "Other"),
            ("Outro", "Other"),
        ],
    },
    {
        "key": "self_employed",
        "label": "Você trabalha como autônomo ou freelancer?",
        "options": [
            ("Sim", 1),
            ("Não", 0),
        ],
    },
    {
        "key": "tech_company",
        "label": "Você trabalha em uma empresa de tecnologia?",
        "options": [
            ("Sim", 1.0),
            ("Não", 0.0),
            ("Prefiro não informar", None),
        ],
    },
    {
        "key": "company_size",
        "label": "Qual é o tamanho da sua empresa?",
        "options": [
            ("Até 5 pessoas", "1-5"),
            ("6 a 25 pessoas", "6-25"),
            ("26 a 100 pessoas", "26-100"),
            ("100 a 500 pessoas", "100-500"),
            ("500 a 1000 pessoas", "500-1000"),
            ("Mais de 1000 pessoas", "More than 1000"),
        ],
    },
    {
        "key": "mental_health_benefits",
        "model_key": "mental_health_benefits",
        "label": "Sua empresa oferece benefícios de saúde mental?",
        "options": [
            ("Sim", "Yes"),
            ("Não", "No"),
            ("Não sei", "I don't know"),
            ("Não se aplica / N/A", "Not eligible for coverage / N/A"),
        ],
    },
    {
        "key": "family_history",
        "model_key": "family_history",
        "label": "Alguém da sua família tem histórico de problemas de saúde mental?",
        "options": [
            ("Sim", "Yes"),
            ("Não", "No"),
            ("Não sei", "I don't know"),
        ],
    },
    {
        "key": "coworkers_view_negative",
        "model_key": "coworkers_view_negative",
        "label": "Você acha que seus colegas veriam negativamente alguém com problema de saúde mental?",
        "options": [
            ("Não, definitivamente não", "No, they do not"),
            ("Não, acho que não", "No, I don't think they would"),
            ("Talvez", "Maybe"),
            ("Sim, acho que sim", "Yes, I think they would"),
            ("Sim, com certeza", "Yes, they do"),
        ],
    },
    {
        "key": "share_with_family",
        "model_key": "share_with_family",
        "label": "Quão aberto(a) você é para falar sobre saúde mental com familiares?",
        "options": [
            ("Muito aberto(a)", "Very open"),
            ("Razoavelmente aberto(a)", "Somewhat open"),
            ("Neutro(a)", "Neutral"),
            ("Pouco aberto(a)", "Somewhat not open"),
            ("Nada aberto(a)", "Not open at all"),
            ("Não se aplica", "Not applicable to me (I do not have a mental illness)"),
        ],
    },
    {
        "key": "comfortable_coworkers",
        "model_key": "comfortable_coworkers",
        "label": "Você se sentiria confortável falando sobre saúde mental com colegas de trabalho?",
        "options": [
            ("Sim", "Yes"),
            ("Talvez", "Maybe"),
            ("Não", "No"),
        ],
    },
    {
        "key": "comfortable_supervisor",
        "label": "Você se sentiria confortável falando sobre saúde mental com seu supervisor?",
        "options": [
            ("Sim", "Yes"),
            ("Talvez", "Maybe"),
            ("Não", "No"),
        ],
    },
]

TOTAL = len(QUESTIONS)

if "step" not in st.session_state:
    st.session_state.step = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}

step = st.session_state.step


st.title("🧠 Saúde Mental no Trabalho")


if step >= TOTAL:
    ans = st.session_state.answers


    row = {col: str(ans[col]) for col in FEAT_COLS}
    df  = pd.DataFrame([row])

    for col in FEAT_COLS:
        le  = encoders[col]
        val = df[col].iloc[0]
        if val not in le.classes_:
            val = "Unknown"
        df[col] = le.transform([val])

    df = df.astype(int)

    st.markdown("### Resultado da análise")
    try:
        prediction = model.predict(df)[0]
        proba      = model.predict_proba(df)[0]
        
        print(f"Prediction: {prediction}")
        print(f"Probabilities: {proba}")
        print(f"DataFrame features: {df.to_dict()}")

        # KPI Display with percentage
        percentage = proba[1] * 100
        
        if percentage >= 50:
            color = "#E74C3C"  # Red
            kpi_message = "risco elevado"
        else:
            color = "#2ECC71"  # Green
            kpi_message = "risco baixo"

        kpi_html = f"""
        <div style="
            background: linear-gradient(135deg, {color}15, {color}25);
            border-left: 5px solid {color};
            border-radius: 10px;
            padding: 30px;
            text-align: center;
            margin: 20px 0;
        ">
            <div style="font-size: 64px; font-weight: bold; color: {color}; margin-bottom: 10px;">
                {percentage:.1f}%
            </div>
            <div style="font-size: 16px; color: #333; line-height: 1.6; font-weight: 500;">
                De chance de <b>precisar de ajuda profissional</b> ({kpi_message})
            </div>
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)
        
        # Brief description based on risk level
        if percentage >= 50:
            st.markdown(
                "💡 **Alto risco detectado.** De acordo com o perfil, você apresenta uma tendência significativa a buscar ajuda profissional. "
                "Considere conversar com um psicólogo ou profissional de saúde mental para melhor compreender suas necessidades."
            )
        else:
            st.markdown(
                "✅ **Risco baixo detectado.** De acordo com o perfil, você não apresenta uma tendência tão forte em buscar ajuda profissional. "
                "Mantenha o autocuidado e fique atento à sua saúde mental."
            )

    except Exception as e:
        st.error(f"Erro ao processar a predição: {e}")

    st.caption(
        "⚠️ Este resultado é apenas uma estimativa baseada em dados estatísticos "
        "e **não substitui avaliação clínica profissional**."
    )
    st.divider()
    if st.button("🔄 Refazer o questionário", use_container_width=True):
        st.session_state.step = 0
        st.session_state.answers = {}
        st.rerun()


else:
    q = QUESTIONS[step]

    st.progress(step / TOTAL, text=f"Pergunta {step + 1} de {TOTAL}")
    st.markdown("")
    st.markdown(f"### {q['label']}")
    st.markdown("")

    for label, value in q["options"]:
        if st.button(label, use_container_width=True, key=f"q{step}_{label}"):
            save_key = q.get("model_key", q["key"])
            st.session_state.answers[save_key] = value
            st.session_state.step += 1
            st.rerun()

    st.markdown("")
    if step > 0:
        if st.button("← Voltar", key="back"):
            st.session_state.step -= 1
            st.rerun()