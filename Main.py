# main.py

import streamlit as st
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, Runner
from dotenv import load_dotenv
import os
import asyncio
import nest_asyncio

# Step 1: Allow nested async loops (for Streamlit compatibility)
nest_asyncio.apply()

# Step 2: Load the Gemini API key
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("❌ GEMINI_API_KEY not found. Please add it in the .env file.")
    st.stop()

# Step 3: Setup Gemini client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

Config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

Materials_Engineering = Agent(
    name='Materials Engineering Agent',
    instructions='You are a Materials Engineering Agent, your job is to provide all information (in depth) about any material in the world w.r.t materials & metallurgical engineering.'
)

# Step 4: Streamlit frontend
st.set_page_config(page_title="Materials Engineering Agent", page_icon="🧪")
st.title("🔍 Materials Engineering Agent")
st.markdown("Enter a material name to get detailed metallurgical and engineering information.")

with st.form("material_form"):
    material_name_info = st.text_input("Enter Material Name (e.g., Steel, PVC, Copper):")
    submit = st.form_submit_button("Get Info")

if submit and material_name_info:
    with st.spinner("Fetching data from Gemini API..."):
        try:
            response = Runner.run_sync(
                Materials_Engineering,
                input=f"Enter name of material or related materials you want to get info: {material_name_info}",
                run_config=Config
            )
            st.success("✅ Information retrieved successfully!")
            
           
            st.markdown("### 📘 Detailed Information:")
            st.write(response.final_output)
        except Exception as e:
            st.error(f"❌ Error: {e}")
