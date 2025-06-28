import streamlit as st
import plotly.graph_objects as go
from model import train_model, predict_depression

# Page setup
st.set_page_config(page_title="💬 Depression Detection", layout="centered", page_icon="🧠")

# 🌈 Custom CSS styling + animation
st.markdown("""
    <style>
        body {
            background-color: #f7f9fc;
        }
        .main-title {
            font-size: 48px;
            font-weight: bold;
            color: #2d3436;
            text-align: center;
            margin-bottom: 10px;
            animation: fadeInDown 2s ease-out;
        }
        .sub-title {
            font-size: 20px;
            text-align: center;
            color: #636e72;
            margin-bottom: 40px;
            animation: fadeIn 3s ease-out;
        }
        .stTextInput > div > div > input {
            border: 2px solid #0984e3;
            border-radius: 10px;
            padding: 10px;
        }
        .stButton > button {
            background-color: #6c5ce7;
            color: white;
            border-radius: 25px;
            padding: 10px 25px;
            font-size: 16px;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background-color: #341f97;
            transform: scale(1.05);
        }
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>
""", unsafe_allow_html=True)

# 🎯 Title
st.markdown('<div class="main-title">💬 Depression Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Analyze social media messages for mental health insights</div>', unsafe_allow_html=True)

# 🚀 Model loading
with st.spinner("⏳ Training the model, please wait..."):
    model, vectorizer = train_model()
st.success("✅ Model is ready!")

# ✏️ Input section
st.subheader("📝 Enter a message:")
user_input = st.text_input("E.g., I feel worthless and alone.")

if st.button("🔍 Analyze Message"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        # Predict
        result_label, prob = predict_depression(user_input, model, vectorizer)
        st.markdown("### 🧠 Prediction Result:")
        st.success(result_label if "Not" in result_label else f"❗ {result_label}")

        # Pie Chart
        st.markdown("### 📊 Depression Probability:")
        fig = go.Figure(data=[go.Pie(
            labels=["✅ Not Depressed", "🚨 Depressed"],
            values=[prob[0], prob[1]],
            hole=0.45,
            marker=dict(colors=["#00b894", "#d63031"]),
            textinfo="label+percent",
            pull=[0, 0.1] if prob[1] > 0.5 else [0.1, 0]
        )])
        fig.update_layout(
            showlegend=True,
            margin=dict(t=20, b=0, l=0, r=0)
        )
        st.plotly_chart(fig)

# ℹ️ Footer
st.markdown("""
    <hr style="border: 1px solid #dfe6e9;" />
    <div style="text-align: center; font-size: 13px; color: #636e72;">
        © 2025 MoodSense AI | Built with ❤️ using Streamlit
    </div>
""", unsafe_allow_html=True)
