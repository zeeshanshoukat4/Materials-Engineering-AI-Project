import streamlit as st
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, Runner
from dotenv import load_dotenv, find_dotenv
import os
import nest_asyncio

# Allow nested loops for Streamlit compatibility
nest_asyncio.apply()

# Load API key
load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("❌ GEMINI_API_KEY not found in .env file.")
    st.stop()

# Setup Gemini client
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

# Agent definition
Materials_Engineering = Agent(
    name='Materials Engineering Agent',
    instructions='You are a Materials Engineering Agent. Provide in-depth engineering information about any material, including properties, applications, limitations, and comparisons.'
)

# Streamlit UI setup
st.set_page_config(page_title="Materials Engineering Agent", page_icon="🧪")
st.title("🔍 Materials Engineering Agent")
st.markdown("Enter a material name to get detailed metallurgical and engineering information with optional TXT report download.")

material = st.text_input("Enter Material Name (e.g., Steel, PVC, Titanium):")
submit = st.button("Get Material Info")

# On submit
if submit and material:
    with st.spinner("🔄 Fetching material data from Gemini..."):
        try:
            prompt = (
                f"Provide a comprehensive report on the material '{material}'. "
                "Include:\n"
                "- Physical and chemical properties in table form\n"
                "- Mechanical properties in table form\n"
                "- Thermal and electrical behavior in table form \n"
                "- Engineering and industrial applications in table form\n"
                "- Cost, availability, limitations\n"
                "- Common substitutes or alternatives"
            )

            response = Runner.run_sync(
                Materials_Engineering,
                input=prompt,
                run_config=Config
            )

            final_output = response.final_output

            # Display on screen
            st.success("✅ Material information retrieved!")
            st.markdown("### 📘 Detailed Engineering Info:")
            st.write(final_output)

            # Generate .txt file content
            txt_content = f"Material: {material}\n\n{final_output}"
            txt_file_name = f"{material.replace(' ', '_')}_Engineering_Report.txt"

            st.download_button(
                label="📥 Download TXT Report",
                data=txt_content,
                file_name=txt_file_name,
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")
