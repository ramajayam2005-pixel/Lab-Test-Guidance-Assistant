
import os
import io
import textwrap
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="LabCare | Lab Test Guidance",
    page_icon="🧪",
    layout="wide"
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

DB_PATH = "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ==========================================
# 2. LOAD EXTERNAL CSS
# ==========================================

def load_css():

    css_path = os.path.join(
        os.path.dirname(__file__),
        "style.css"
    )

    if os.path.exists(css_path):

        with open(css_path, "r", encoding="utf-8") as file:

            st.markdown(
                f"<style>{file.read()}</style>",
                unsafe_allow_html=True
            )

    else:

        st.warning("style.css file not found.")


load_css()


# ==========================================
# 3. KNOWLEDGE BASE
# ==========================================

KNOWLEDGE_BASE = [

    {
        "title": "Complete Blood Count (CBC)",
        "department": "Hematology",
        "content": """
        Complete Blood Count measures red blood cells,
        white blood cells, hemoglobin and platelets.

        Department: Hematology.

        Process:
        1. Verify the laboratory request.
        2. Confirm patient identity.
        3. Collect the blood sample by trained staff.
        4. Label and transport the sample.
        5. Process the sample.
        6. Review and release the report.

        Preparation:
        Fasting is generally not required for CBC alone.
        Follow laboratory instructions.
        """
    },

    {
        "title": "Blood Glucose",
        "department": "Biochemistry",
        "content": """
        Blood glucose testing measures the amount of glucose in blood.

        Department: Biochemistry.

        Types include fasting, random and post-meal blood glucose.

        Process:
        1. Check the requested test type.
        2. Confirm preparation requirements.
        3. Collect the blood sample.
        4. Process the sample.
        5. Validate the result.
        6. Release the report.

        Preparation:
        Requirements depend on the type of glucose test.
        Confirm fasting instructions with the laboratory.
        """
    },

    {
        "title": "Lipid Profile",
        "department": "Biochemistry",
        "content": """
        A lipid profile measures total cholesterol,
        HDL, LDL and triglycerides.

        Department: Biochemistry.

        Process:
        1. Verify the test request.
        2. Confirm preparation instructions.
        3. Collect the blood sample.
        4. Process the sample.
        5. Analyze the sample.
        6. Review and release the report.

        Preparation:
        Fasting is not required for every lipid profile.
        Confirm requirements with the clinician or laboratory.
        """
    },

    {
        "title": "Urine Routine",
        "department": "Clinical Pathology",
        "content": """
        Urine routine examination evaluates physical,
        chemical and microscopic urine characteristics.

        Department: Clinical Pathology.

        Process:
        1. Obtain a suitable sample container.
        2. Follow collection instructions.
        3. Collect the requested urine sample.
        4. Label and submit the sample.
        5. Laboratory examination is performed.
        6. Review and release the report.

        Preparation:
        Follow the sample collection instructions provided by the lab.
        """
    },

    {
        "title": "Thyroid Profile",
        "department": "Biochemistry",
        "content": """
        Thyroid profile may include TSH, T3 and T4 measurements.

        Department: Biochemistry.

        Process:
        1. Verify requested thyroid tests.
        2. Confirm preparation instructions.
        3. Collect the blood sample.
        4. Process the sample.
        5. Analyze requested parameters.
        6. Review and release the report.

        Preparation:
        Fasting is not routinely required for every thyroid test.
        Confirm medication timing and preparation with the clinician.
        """
    }
]


# ==========================================
# 4. TEST DETAILS
# ==========================================

TEST_DETAILS = {

    "Complete Blood Count (CBC)": {
        "department": "Hematology",
        "sample": "Blood",
        "purpose": "Measures blood cells, hemoglobin and platelets.",
        "preparation": "Fasting is generally not needed for CBC alone. Confirm with your lab.",
        "steps": [
            "Check the test request.",
            "Verify patient identity.",
            "Collect the blood sample.",
            "Label and transport the sample.",
            "Complete laboratory analysis.",
            "Review and release the report."
        ]
    },

    "Blood Glucose": {
        "department": "Biochemistry",
        "sample": "Blood",
        "purpose": "Measures glucose levels in blood.",
        "preparation": "Preparation depends on the requested glucose test type.",
        "steps": [
            "Check the requested test type.",
            "Confirm preparation and timing.",
            "Collect the blood sample.",
            "Process the sample.",
            "Validate the result.",
            "Release the report."
        ]
    },

    "Lipid Profile": {
        "department": "Biochemistry",
        "sample": "Blood",
        "purpose": "Measures cholesterol and triglyceride levels.",
        "preparation": "Fasting requirements vary. Confirm with the laboratory.",
        "steps": [
            "Verify the test request.",
            "Confirm fasting instructions.",
            "Collect the blood sample.",
            "Process the sample.",
            "Analyze lipid measurements.",
            "Release the report."
        ]
    },

    "Urine Routine": {
        "department": "Clinical Pathology",
        "sample": "Urine",
        "purpose": "Examines physical, chemical and microscopic urine characteristics.",
        "preparation": "Follow the laboratory's sample collection instructions.",
        "steps": [
            "Obtain a suitable sample container.",
            "Follow collection instructions.",
            "Collect the requested sample.",
            "Label and submit the sample.",
            "Perform laboratory examination.",
            "Review and release the report."
        ]
    },

    "Thyroid Profile": {
        "department": "Biochemistry",
        "sample": "Blood",
        "purpose": "Measures thyroid-related hormones such as TSH, T3 and T4.",
        "preparation": "Confirm preparation and medication timing with the clinician.",
        "steps": [
            "Verify requested thyroid tests.",
            "Check preparation instructions.",
            "Collect the blood sample.",
            "Process the sample.",
            "Analyze requested parameters.",
            "Release the report."
        ]
    }
}


# ==========================================
# 5. CHROMA VECTOR DATABASE
# ==========================================

@st.cache_resource(show_spinner="Loading AI knowledge base...")
def get_vector_store():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    documents = [

        Document(
            page_content=item["content"],
            metadata={
                "source": item["title"],
                "department": item["department"]
            }
        )

        for item in KNOWLEDGE_BASE
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    store = Chroma(
        collection_name="lab_guidance",
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )

    if store._collection.count() == 0:

        store.add_documents(chunks)

    return store


# ==========================================
# 6. RETRIEVAL
# ==========================================

def retrieve_information(question):

    store = get_vector_store()

    results = store.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source', 'Lab Guide')}\n"
            f"Department: {doc.metadata.get('department', 'General')}\n"
            f"{doc.page_content}"
            for doc in results
        ]
    )

    sources = list(
        dict.fromkeys(
            doc.metadata.get("source", "Lab Guide")
            for doc in results
        )
    )

    return context, sources


# ==========================================
# 7. AI ANSWER GENERATION
# ==========================================

def answer_question(question):

    context, sources = retrieve_information(question)

    if GROQ_API_KEY:

        try:

            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                api_key=GROQ_API_KEY
            )

            prompt = f"""
            You are LabCare, a hospital laboratory
            process guidance assistant.

            Answer using only the provided context.

            Rules:
            - Use simple professional English.
            - Explain laboratory processes.
            - Do not diagnose diseases.
            - Do not prescribe medicines.
            - Do not interpret individual patient reports.
            - Do not invent medical instructions.
            - Refer users to a qualified professional
              when information is unavailable.

            CONTEXT:
            {context}

            QUESTION:
            {question}

            ANSWER:
            """

            response = llm.invoke(prompt)

            return response.content, sources, "Groq AI + ChromaDB"

        except Exception:

            pass

    fallback = (
        "Relevant information from the laboratory knowledge base:\n\n"
        + context
        + "\n\nPlease contact the laboratory or clinician "
          "for individual medical instructions."
    )

    return fallback, sources, "ChromaDB Retrieval"


# ==========================================
# 8. SUMMARY GENERATOR
# ==========================================

def generate_summary(
    reference,
    test,
    department,
    status,
    notes
):

    return f"""
LAB TEST GUIDANCE SUMMARY

Reference Name: {reference}

Test Requested: {test}

Department: {department}

Workflow Status: {status}

Additional Notes:
{notes if notes else "No additional notes."}

Generated On:
{datetime.now().strftime("%d-%m-%Y %I:%M %p")}

This is an administrative workflow summary.
It is not a diagnostic report or medical interpretation.

Please contact the laboratory or healthcare professional
for individual medical instructions.
""".strip()


# ==========================================
# 9. PDF GENERATION
# ==========================================

def create_pdf(summary):

    buffer = io.BytesIO()

    pdf = canvas.Canvas(
        buffer,
        pagesize=A4
    )

    width, height = A4

    pdf.setTitle("LabCare Guidance Summary")

    pdf.setFont("Helvetica-Bold", 20)

    pdf.drawString(
        50,
        height - 55,
        "LABCARE"
    )

    pdf.setFont("Helvetica", 10)

    pdf.drawString(
        50,
        height - 75,
        "Lab Test Guidance Assistant"
    )

    pdf.line(
        50,
        height - 90,
        width - 50,
        height - 90
    )

    y = height - 120

    for line in summary.split("\n"):

        wrapped = textwrap.wrap(
            line,
            width=90
        ) or [""]

        for part in wrapped:

            if y < 55:

                pdf.showPage()
                y = height - 55

            pdf.setFont("Helvetica", 10)

            pdf.drawString(
                50,
                y,
                part
            )

            y -= 17

    pdf.save()

    buffer.seek(0)

    return buffer.getvalue()


# ==========================================
# 10. SIDEBAR
# ==========================================

st.sidebar.markdown(
    """
    <h1 style="color:#1768ac;">🧪 LabCare</h1>
    <p style="color:#718096;">Lab Test Guidance Assistant</p>
    """,
    unsafe_allow_html=True
)

st.sidebar.divider()

role = st.sidebar.selectbox(
    "Select User Role",
    ["Patient", "Staff"]
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Lab Test Guidance",
        "AI Assistant",
        "Summary Generator"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Academic Project Demonstration"
)

st.sidebar.info(
    "Role selection demonstrates different views. "
    "It is not secure authentication."
)


# ==========================================
# 11. APPLICATION HEADER
# ==========================================

st.markdown(
    '<div class="main-title">LabCare Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Smart laboratory workflow and test guidance system'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# 12. DASHBOARD
# ==========================================

if page == "Dashboard":

    st.subheader("Dashboard Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Available Tests",
        len(TEST_DETAILS)
    )

    col2.metric(
        "Departments",
        3
    )

    col3.metric(
        "Current Role",
        role
    )

    st.markdown("### Welcome to LabCare")

    st.write(
        "LabCare helps patients and hospital staff understand "
        "common laboratory tests, departments and workflow processes."
    )

    st.markdown("### Available Services")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="info-card">
                <h4>🧪 Lab Test Guidance</h4>
                <p>Explore test purposes, sample types,
                preparation instructions and laboratory workflows.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-card">
                <h4>📄 Summary Generator</h4>
                <p>Create administrative test workflow summaries
                and download them as PDF documents.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="info-card">
                <h4>🤖 AI Assistant</h4>
                <p>Ask laboratory-related questions using
                retrieval-augmented generation.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-card">
                <h4>📥 PDF Export</h4>
                <p>Export generated workflow summaries
                for demonstration and documentation.</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# 13. LAB TEST GUIDANCE
# ==========================================

elif page == "Lab Test Guidance":

    st.subheader("Laboratory Test Guidance")

    selected_test = st.selectbox(
        "Select a laboratory test",
        list(TEST_DETAILS.keys())
    )

    details = TEST_DETAILS[selected_test]

    st.markdown(f"### {selected_test}")

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            f"Department: {details['department']}"
        )

    with col2:

        st.info(
            f"Sample Type: {details['sample']}"
        )

    st.markdown("### Purpose")

    st.write(details["purpose"])

    st.markdown("### Preparation Instructions")

    st.warning(details["preparation"])

    st.markdown("### Laboratory Workflow")

    for index, step in enumerate(
        details["steps"],
        start=1
    ):

        st.markdown(
            f"**Step {index}:** {step}"
        )

    st.caption(
        "Actual collection and processing must follow "
        "the laboratory's approved procedures."
    )


# ==========================================
# 14. AI ASSISTANT
# ==========================================

elif page == "AI Assistant":

    st.subheader("AI Laboratory Assistant")

    st.write(
        "Ask questions about laboratory tests, departments and processes."
    )

    st.info(
        "Please do not enter patient names, phone numbers, "
        "medical record numbers or personal health information."
    )

    if "chat_history" not in st.session_state:

        st.session_state.chat_history = []

    for chat in st.session_state.chat_history:

        with st.chat_message(chat["role"]):

            st.markdown(chat["content"])

    question = st.chat_input(
        "Ask a laboratory-related question..."
    )

    if question:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Searching knowledge base..."):

                try:

                    answer, sources, mode = answer_question(question)

                    st.markdown(answer)

                    st.caption(
                        f"Answer Mode: {mode}"
                    )

                    st.markdown("**Sources:**")

                    for source in sources:

                        st.write(f"- {source}")

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as error:

                    st.error(
                        "Unable to access the knowledge base. "
                        "Check your internet connection and try again."
                    )

                    st.caption(str(error))


# ==========================================
# 15. SUMMARY GENERATOR
# ==========================================

elif page == "Summary Generator":

    st.subheader("Lab Test Summary Generator")

    st.write(
        "Use fictional reference details for this demonstration."
    )

    with st.form("summary_form"):

        reference = st.text_input(
            "Reference Name / Demo ID",
            placeholder="Example: DEMO-001"
        )

        selected_test = st.selectbox(
            "Select Test",
            list(TEST_DETAILS.keys())
        )

        department = st.text_input(
            "Department",
            value=TEST_DETAILS[selected_test]["department"]
        )

        status = st.selectbox(
            "Workflow Status",
            [
                "Request Received",
                "Sample Collection Pending",
                "Sample Collected",
                "Processing",
                "Completed"
            ]
        )

        notes = st.text_area(
            "Additional Workflow Notes",
            placeholder="Enter non-identifying administrative notes..."
        )

        submitted = st.form_submit_button(
            "Generate Summary",
            use_container_width=True
        )

    if submitted:

        if not reference.strip():

            st.error(
                "Please enter a demo reference name or ID."
            )

        else:

            summary = generate_summary(
                reference,
                selected_test,
                department,
                status,
                notes
            )

            st.session_state.generated_summary = summary

    if "generated_summary" in st.session_state:

        st.markdown("### Generated Summary")

        st.text_area(
            "Summary Preview",
            value=st.session_state.generated_summary,
            height=300
        )

        pdf_file = create_pdf(
            st.session_state.generated_summary
        )

        st.download_button(
            label="📥 Download PDF Summary",
            data=pdf_file,
            file_name="LabCare_Summary.pdf",
            mime="application/pdf",
            use_container_width=True
        )


# ==========================================
# 16. FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div class="footer-text">
        LabCare | Lab Test Guidance Assistant | Academic Project Demo
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "This application provides general laboratory workflow information. "
    "It is not a substitute for professional medical advice, diagnosis "
    "or treatment."
)
