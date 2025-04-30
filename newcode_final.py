import os
import streamlit as st
import pandas as pd
import sqlite3
import requests
from io import StringIO
import json  # Needed for parsing LLM response
from datetime import datetime, timedelta
import re  # Import regex for parsing plot commands and cleaning JSON
import time # For potential future use (e.g., simulated delays)
from fpdf import FPDF # Needed for PDF export (from v3.py)
import plotly.express as px

# Ensure necessary Langchain/Groq imports are present
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
    # from langchain_core.runnables import RunnablePassthrough # Not explicitly used?
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.output_parsers import StrOutputParser

    LANGCHAIN_AVAILABLE = True
except ImportError:
    st.warning(
        "Langchain/Groq libraries not found. AI features will be disabled. Install with: pip install langchain langchain-groq fpdf"
    ) # Added fpdf to install instruction
    LANGCHAIN_AVAILABLE = False

    # Define dummy classes/functions if Langchain is not available to avoid NameErrors later
    class ChatGroq:
        pass

    class PromptTemplate:
        @staticmethod
        def from_template(text):
            # Store template text for potential inspection later if needed
            prompt_template = lambda: None # Simple object to hold attributes
            prompt_template.template = text
            # Basic extraction of variables (might not be perfectly robust for complex templates)
            prompt_template.input_variables = re.findall(r"\{(\w+)\}", text)

            # Return a dummy object that can be invoked (returning None)
            class DummyPrompt:
                def __init__(self, template, variables):
                    self.template = template
                    self.input_variables = variables

                def invoke(self, input_dict):
                    print(f"DummyPrompt invoked with: {input_dict}")
                    return None # Cannot execute LLM

                def format(self, **kwargs):
                    formatted_text = self.template
                    for key, value in kwargs.items():
                        formatted_text = formatted_text.replace(f"{{{key}}}", str(value))
                    # Handle potential escaped braces from the template itself if needed
                    formatted_text = formatted_text.replace("{{", "{").replace("}}", "}")
                    return formatted_text

            return DummyPrompt(text, prompt_template.input_variables)

    # class RunnablePassthrough: pass # Not explicitly used
    class HumanMessage:
         def __init__(self, content): self.content = content # Basic init for dummy

    class AIMessage:
         def __init__(self, content): self.content = content # Basic init for dummy

    class StrOutputParser:
        def __init__(self): pass # Dummy init
        def invoke(self, input_data): return str(input_data) # Dummy invoke

# === Constants ===
DB_NAME = "unified_compliance.db"

# === Authentication ===
def login():
    """Handles user login via sidebar."""
    st.sidebar.title("🔐 Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    # Default credentials if not found in secrets
    app_user = "admin"
    app_pass = "pass123"
    secrets_available = hasattr(st, 'secrets')

    if secrets_available:
        try:
            # Use .get() with default values for robustness
            app_user = st.secrets.get("APP_USERNAME", "admin")
            app_pass = st.secrets.get("APP_PASSWORD", "pass123")
        except Exception as e:
            st.warning(f"Could not read login secrets: {e}. Using defaults.")
            app_user = "admin"
            app_pass = "pass123"

    if st.sidebar.button("Login"):
        if username == app_user and password == app_pass:
            st.session_state.logged_in = True
            # Use st.rerun() for cleaner state update propagation
            st.rerun()
        else:
            st.sidebar.error("Invalid username or password")

# Initialize login state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Enforce login
if not st.session_state.logged_in:
    login()
    # Provide guidance on default credentials and secrets
    if hasattr(st, 'secrets'):
        try:
            using_default_creds = (st.secrets.get("APP_USERNAME") is None or
                                   st.secrets.get("APP_PASSWORD") is None)
            if using_default_creds:
                st.info("Default login: admin / pass123 (Consider setting APP_USERNAME/APP_PASSWORD in secrets.toml)")
        except Exception:
            st.info("Default login: admin / pass123 (Create .streamlit/secrets.toml to set custom login)")
    else:
        st.info("Default login: admin / pass123 (Secrets management may not be available)")
    st.stop() # Stop execution if not logged in

# === App Setup ===
st.set_page_config(page_title="Unified Banking Compliance Solution", layout="wide")

# === Load LLM (Requires secrets.toml and Langchain installation) ===
@st.cache_resource(show_spinner="Loading AI Assistant...")
def load_llm():
    """Loads the Groq LLM using API key from st.secrets."""
    if not LANGCHAIN_AVAILABLE: return None

    api_key = None
    if not hasattr(st, 'secrets'):
        st.error("Streamlit secrets management is not available in this environment.")
        return None

    try:
        # Use .get() for safer access
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception as e:
        st.error(f"Error accessing Streamlit secrets: {e}. Ensure '.streamlit/secrets.toml' exists.")
        return None

    if not api_key:
        st.error("❗️ GROQ API Key not found in Streamlit secrets.")
        st.info(
            "To use the AI features, please ensure:\n"
            "1. The file `.streamlit/secrets.toml` exists in your project directory.\n"
            "2. It contains the line: `GROQ_API_KEY = \"YOUR_ACTUAL_GROQ_API_KEY\"`\n"
            "3. You have restarted the Streamlit app after saving the file."
        )
        return None

    try:
        # Increased timeout might be needed for complex LLM responses
        llm_instance = ChatGroq(temperature=0.2, model_name="llama3-70b-8192", api_key=api_key, request_timeout=120)
        return llm_instance
    except Exception as e:
        # Provide specific feedback if possible (e.g., authentication error)
        st.error(f"🚨 Failed to initialize Groq client: {e}")
        st.info("Please verify your GROQ_API_KEY value in secrets.toml and ensure you have internet connectivity.")
        return None

llm = load_llm()

# === Database Setup ===
def init_db(db_path=DB_NAME):
    """ Initializes the SQLite database and tables. Returns the db path."""
    try:
        with sqlite3.connect(db_path) as conn: # Use context manager
            cursor = conn.cursor()
            # Accounts Data Table (Ensure consistent naming with parse_data output)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS accounts_data (
                    "Account_ID" TEXT, "Account_Type" TEXT, "Last_Transaction_Date" TEXT,
                    "Account_Status" TEXT, "Email_Contact_Attempt" TEXT, "SMS_Contact_Attempt" TEXT,
                    "Phone_Call_Attempt" TEXT, "KYC_Status" TEXT, "Branch" TEXT
                    -- Add other potential standardized columns if known, e.g., Balance
                )""")
            # Dormant Flags Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dormant_flags (
                    account_id TEXT PRIMARY KEY,
                    flag_instruction TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )""")
            # Dormant Ledger Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dormant_ledger (
                    account_id TEXT PRIMARY KEY,
                    classification TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )""")
            # Insight Log Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS insight_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    observation TEXT,
                    trend TEXT,
                    insight TEXT,
                    action TEXT
                )""")
            # SQL Query History Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sql_query_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    natural_language_query TEXT,
                    sql_query TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )""")
            # Commit happens automatically with context manager exit
        return db_path
    except sqlite3.Error as e:
        st.error(f"Database Initialization Error: {e}")
        st.stop() # Stop if DB can't be initialized
    except Exception as e:
        st.error(f"An unexpected error occurred during DB initialization: {e}")
        st.stop()

db_name = init_db()

# === Helper Functions ===
@st.cache_data(show_spinner="Parsing data...")
def parse_data(file_input):
    """Parses data, standardizes column names, converts types, and stores original names."""
    df = None
    original_columns = []
    try:
        # Handle DataFrame input directly (e.g., from URL fetch in v3 style)
        if isinstance(file_input, pd.DataFrame):
            df = file_input.copy()
            original_columns = list(df.columns)
        # Handle UploadedFile object (Streamlit native)
        elif hasattr(file_input, 'name'):
            name = file_input.name.lower()
            st.info(f"Reading: {name}") # Give user feedback
            if name.endswith('.csv'):
                df = pd.read_csv(file_input)
            elif name.endswith(('.xlsx', '.xls')):
                # Ensure openpyxl is installed: pip install openpyxl
                df = pd.read_excel(file_input, engine='openpyxl')
            elif name.endswith('.json'):
                df = pd.read_json(file_input)
            else:
                st.error("Unsupported file format. Please use CSV, XLSX, or JSON.")
                return None
            if df is not None: original_columns = list(df.columns)
        # Handle raw string input (e.g., from URL fetch returning text)
        elif isinstance(file_input, str):
            df = pd.read_csv(StringIO(file_input))
            if df is not None: original_columns = list(df.columns)
        else:
            st.error("Invalid input type for parsing.")
            return None

        if df is None:
            st.error("Failed to read data.")
            return None
        if df.empty:
            st.warning("The uploaded file is empty or could not be parsed into data.")
            return df # Return empty df, let caller handle

        # --- Standardization ---
        # 1. Strip whitespace, replace spaces with underscores, remove invalid chars
        df.columns = df.columns.str.strip().str.replace(' ', '_', regex=False).str.replace('[^A-Za-z0-9_]+', '', regex=True)
        # 2. Handle potentially empty column names after cleaning
        df.columns = [f"col_{i}" if c == "" else c for i, c in enumerate(df.columns)]
        standardized_columns = list(df.columns)

        # Store mapping from standardized to original names
        st.session_state['column_mapping'] = {std: orig for std, orig in zip(standardized_columns, original_columns)}

        # --- Type Conversion & Cleaning (using standardized names) ---
        # Define expected columns and their types
        date_cols = ['Last_Transaction_Date'] # Add others if needed
        string_cols = ["Account_ID", "Account_Type", "Account_Status", "Email_Contact_Attempt", "SMS_Contact_Attempt", "Phone_Call_Attempt", "KYC_Status", "Branch"]
        # numeric_cols = ['Balance'] # Example

        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce') # Coerce errors to NaT

        for col in string_cols:
            if col in df.columns:
                # Convert non-string to string only if necessary and col isn't all NaN
                if not pd.api.types.is_string_dtype(df[col]) and df[col].notna().any():
                    df[col] = df[col].astype(str)
                # Fill missing values with a placeholder like 'Unknown' or ''
                df[col] = df[col].fillna('Unknown').str.strip() # Also strip whitespace

        # Example for numeric columns:
        # for col in numeric_cols:
        #     if col in df.columns:
        #         df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) # Coerce, fill NaN with 0

        return df

    except Exception as e:
        st.error(f"Error during data parsing/standardization: {e}")
        st.error(f"Original columns detected: {original_columns if original_columns else 'N/A'}")
        return None

def save_to_db(df, table_name="accounts_data", db_path=DB_NAME):
    """Saves DataFrame to SQLite, replacing table. Assumes standardized column names."""
    if df is None or df.empty:
        st.sidebar.warning(f"Skipped saving empty or None DataFrame to '{table_name}'.")
        return False
    try:
        # Ensure required columns exist (based on DB schema)
        required_db_cols = ["Account_ID", "Account_Type", "Last_Transaction_Date", "Account_Status", "Email_Contact_Attempt", "SMS_Contact_Attempt", "Phone_Call_Attempt", "KYC_Status", "Branch"]
        cols_to_save = [col for col in required_db_cols if col in df.columns]
        if not cols_to_save:
             st.sidebar.error(f"No matching columns found in DataFrame for table '{table_name}'. Cannot save.")
             return False

        df_to_save = df[cols_to_save].copy() # Select only relevant columns

        # Convert datetime objects to ISO format strings for SQLite compatibility
        for col in df_to_save.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
             df_to_save[col] = df_to_save[col].dt.strftime('%Y-%m-%d %H:%M:%S') # Or another ISO format

        with sqlite3.connect(db_path) as conn_save:
            # Use standardized names directly from df_to_save
            df_to_save.to_sql(table_name, conn_save, if_exists="replace", index=False)
        return True
    except sqlite3.Error as e:
        st.sidebar.error(f"Database Save Error ('{table_name}'): {e}. Check data compatibility.")
        return False
    except KeyError as e:
         st.sidebar.error(f"Missing expected column during save preparation: {e}")
         return False
    except Exception as e:
        st.sidebar.error(f"An unexpected error occurred during DB save ('{table_name}'): {e}")
        return False

def save_summary_to_db(observation, trend, insight, action, db_path=DB_NAME):
    """Saves analysis summary to the insight log table."""
    try:
        with sqlite3.connect(db_path) as conn_log:
            cursor_log = conn_log.cursor()
            cursor_log.execute("""INSERT INTO insight_log (timestamp, observation, trend, insight, action)
                                  VALUES (?, ?, ?, ?, ?)""",
                               (datetime.now().isoformat(), str(observation), str(trend), str(insight), str(action)))
        return True
    except sqlite3.Error as e:
        st.error(f"Failed to save summary to DB: {e}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred saving summary: {e}")
        return False

# === Chatbot Backend Function (Dynamic LLM Approach) ===
def get_response_and_chart(user_query: str, current_data: pd.DataFrame, llm_model: ChatGroq):
    """
    Processes user query dynamically using LLM. Determines if it's a plot request
    or a question, generates the plot or answers accordingly.
    Handles JSON parsing and potential errors.
    """
    chart = None
    response_text = "Sorry, something went wrong processing your request." # Default error

    if not llm_model or not LANGCHAIN_AVAILABLE:
        return "⚠️ AI Assistant not available (check API key or install Langchain). Cannot process dynamic requests.", None

    if current_data is None or current_data.empty:
        return "⚠️ No data loaded. Please upload and process data first.", None

    # --- Prepare Context for LLM ---
    try:
        # Generate column descriptions using standardized names
        cols_info = []
        for col in current_data.columns:
            dtype = str(current_data[col].dtype)
            # Provide more useful info for categorical vs numeric/date
            if pd.api.types.is_numeric_dtype(current_data[col]):
                 unique_approx = f"Min: {current_data[col].min():.2f}, Max: {current_data[col].max():.2f}"
            elif pd.api.types.is_datetime64_any_dtype(current_data[col]):
                 unique_approx = f"Date Range: {current_data[col].min()} to {current_data[col].max()}" if current_data[col].notna().any() else "No valid dates"
            else: # Categorical/Object
                 unique_count = current_data[col].nunique()
                 unique_approx = f"~{unique_count} unique values"
                 if unique_count < 10: # List first few if count is low
                     unique_approx += f" (e.g., {', '.join(current_data[col].unique()[:3])})"
            cols_info.append(f"- `{col}` (Type: {dtype}, Info: {unique_approx})")

        columns_description = "\n".join(cols_info)
        num_rows = len(current_data)
        allowed_plots = ['bar', 'pie', 'histogram', 'box', 'scatter']

        # --- Define the LLM Prompt for Interpretation ---
        interpretation_prompt_text = """
You are an intelligent assistant interpreting user requests about a banking compliance dataset.
Analyze the user's query: "{user_query}"

Available Data Context:
- Number of rows: {num_rows}
- Available standardized columns and their details:
{columns_description}
- Allowed plot types: {allowed_plots_str}

Task:
1. Determine if the user query is primarily a request to **plot** data or a **question** to be answered.
2. If it's a **plotting request** AND seems feasible with the available columns and allowed plot types:
   - Identify the most appropriate plot type from the allowed list ({allowed_plots_str}).
   - Identify the necessary column(s) for that plot type using the **exact standardized column names** provided above.
     - For 'bar', 'histogram', 'box': Primarily `x_column` (category/numeric) or `y_column` (numeric for box), potentially `color_column`.
     - For 'pie': Requires `names_column` (categorical, few unique values). `values_column` is implicitly the count.
     - For 'scatter': Requires `x_column` (numeric/date) and `y_column` (numeric/date), potentially `color_column`.
   - Generate a concise, suitable title for the plot.
   - Output **ONLY** a valid JSON object with the following structure (use double braces for literal JSON braces if needed, but standard JSON is preferred):
     ```json
     {{
       "action": "plot",
       "plot_type": "chosen_plot_type",
       "x_column": "Standardized_X_Column_Name_or_null",
       "y_column": "Standardized_Y_Column_Name_or_null",
       "color_column": "Standardized_Color_Column_Name_or_null",
       "names_column": "Standardized_Pie_Names_Column_or_null",
       "title": "Suggested Plot Title"
     }}
     ```
   - IMPORTANT: Use `null` (JSON null, not the string "null") for unused keys. Ensure the JSON is valid. Use **only** the exact standardized column names listed.
3. If the query is a **question**, a request for analysis/summary, or an infeasible/unclear plot request:
   - Output **ONLY** a valid JSON object with the structure:
     ```json
     {{
       "action": "answer",
       "query_for_llm": "{user_query}"
     }}
     ```

Constraints:
- Adhere strictly to the JSON format specified for each action.
- Only use standardized column names listed in the context.
- Only use plot types from the allowed list: {allowed_plots_str}.
- If unsure about plotting feasibility (e.g., wrong data type, too many categories for pie), default to the "answer" action.
- Do NOT add any explanations, greetings, or markdown formatting around the JSON output.

JSON Output:
"""
        # Create PromptTemplate *before* filling variables
        interpretation_prompt = PromptTemplate.from_template(interpretation_prompt_text)

        # Prepare the input dictionary for the template
        prompt_input = {
            "user_query": user_query,
            "num_rows": num_rows,
            "columns_description": columns_description,
            "allowed_plots_str": ', '.join(allowed_plots)
        }

        # Define the chain for interpretation
        interpretation_chain = interpretation_prompt | llm_model | StrOutputParser()

        # --- Invoke LLM for Interpretation ---
        print(f"DEBUG: Invoking LLM for interpretation with input keys: {prompt_input.keys()}")
        with st.spinner("Interpreting request..."):
            llm_json_output_str = interpretation_chain.invoke(prompt_input)
            print(f"DEBUG: LLM Interpretation Output (raw): {llm_json_output_str}")

        # --- Parse LLM JSON Output ---
        try:
            # Clean potential markdown fences and whitespace
            cleaned_json_str = re.sub(r"^```json\s*|\s*```$", "", llm_json_output_str, flags=re.MULTILINE).strip()
            if not cleaned_json_str: raise ValueError("LLM returned an empty response after cleaning.")

            llm_output = json.loads(cleaned_json_str)
            action = llm_output.get("action")
            print(f"DEBUG: Parsed LLM Action: {action}")

            # --- Execute Action Based on LLM Interpretation ---
            if action == "plot":
                plot_type = llm_output.get("plot_type")
                x_col = llm_output.get("x_column")
                y_col = llm_output.get("y_column")
                color_col = llm_output.get("color_column")
                names_col = llm_output.get("names_column") # Specific to pie
                # values_col not needed explicitly for px.pie on counts
                title = llm_output.get("title", f"Plot based on: {user_query[:30]}...")

                # Validate suggested columns against actual dataframe columns
                all_cols = list(current_data.columns)
                def validate_col(col_name):
                    return col_name if col_name in all_cols else None

                x_col_valid = validate_col(x_col)
                y_col_valid = validate_col(y_col)
                color_col_valid = validate_col(color_col)
                names_col_valid = validate_col(names_col)

                print(f"DEBUG: Plot Params - Type:'{plot_type}', X:'{x_col_valid}', Y:'{y_col_valid}', Color:'{color_col_valid}', Names:'{names_col_valid}'")

                # --- Plotting Logic ---
                if plot_type == 'pie':
                    if not names_col_valid: raise ValueError("Valid 'names_column' needed for pie chart.")
                    # Check uniqueness for pie chart feasibility
                    unique_count = current_data[names_col_valid].nunique()
                    if unique_count > 25: # Increased limit slightly
                        raise ValueError(f"Too many unique values ({unique_count}) in '{names_col_valid}' for a pie chart. Try a bar chart.")
                    # Calculate counts for the pie chart
                    counts = current_data[names_col_valid].value_counts().reset_index()
                    # Rename columns for plotly express (common pitfall)
                    counts.columns = [names_col_valid, 'count']
                    chart = px.pie(counts, names=names_col_valid, values='count', title=title)
                    response_text = f"Generated pie chart for '{names_col_valid}'."

                elif plot_type == 'bar':
                    if not x_col_valid: raise ValueError("Valid 'x_column' needed for bar chart.")
                    # Use histogram for counts of categories
                    chart = px.histogram(current_data, x=x_col_valid, color=color_col_valid, title=title, barmode='group')
                    response_text = f"Generated bar chart showing counts for '{x_col_valid}'" + (f" grouped by '{color_col_valid}'." if color_col_valid else ".")

                elif plot_type == 'histogram':
                    if not x_col_valid: raise ValueError("Valid 'x_column' needed for histogram.")
                    # Check if column is numeric or datetime for histogram
                    if not pd.api.types.is_numeric_dtype(current_data[x_col_valid]) and not pd.api.types.is_datetime64_any_dtype(current_data[x_col_valid]):
                        raise ValueError(f"Histogram requires a numeric or date column. '{x_col_valid}' is not. Try a 'bar chart' instead for counts.")
                    chart = px.histogram(current_data, x=x_col_valid, color=color_col_valid, title=title)
                    response_text = f"Generated histogram for '{x_col_valid}'" + (f" grouped by '{color_col_valid}'." if color_col_valid else ".")

                elif plot_type == 'box':
                    # Box plot typically needs a numeric y-axis and optional categorical x-axis
                    if not y_col_valid: raise ValueError("Valid 'y_column' (numeric) needed for box plot.")
                    if not pd.api.types.is_numeric_dtype(current_data[y_col_valid]):
                         raise ValueError(f"Box plot requires a numeric 'y_column'. '{y_col_valid}' is not numeric.")
                    chart = px.box(current_data, x=x_col_valid, y=y_col_valid, color=color_col_valid, title=title, points="outliers") # x can be None
                    response_text = f"Generated box plot for '{y_col_valid}'" + (f" grouped by '{x_col_valid}'." if x_col_valid else ".") + (f" colored by '{color_col_valid}'." if color_col_valid else ".")

                elif plot_type == 'scatter':
                    if not x_col_valid or not y_col_valid: raise ValueError(f"Valid 'x_column' ('{x_col_valid}') and 'y_column' ('{y_col_valid}') needed for scatter plot.")
                     # Optional: Add checks if columns are numeric/date-like for scatter
                    chart = px.scatter(current_data, x=x_col_valid, y=y_col_valid, color=color_col_valid, title=title, hover_data=current_data.columns) # Add hover
                    response_text = f"Generated scatter plot of '{x_col_valid}' vs '{y_col_valid}'" + (f" colored by '{color_col_valid}'." if color_col_valid else ".")

                else:
                    response_text = f"⚠️ Plot type '{plot_type}' suggested by AI is not supported or was invalid in context."
                    chart = None

                # Add numeric summary if a chart was generated and the primary column is numeric
                primary_plot_col = x_col_valid or names_col_valid # Choose the main column used
                if chart and primary_plot_col:
                    temp_data_for_stats = current_data[primary_plot_col].dropna()
                    summary_text = ""
                    if pd.api.types.is_numeric_dtype(temp_data_for_stats) and not temp_data_for_stats.empty:
                        desc = temp_data_for_stats.describe()
                        summary_text = (f"**Summary for `{primary_plot_col}`:** "
                                        f"Mean: {desc.get('mean', float('nan')):.2f}, "
                                        f"Std: {desc.get('std', float('nan')):.2f}, "
                                        f"Min: {desc.get('min', float('nan')):.2f}, "
                                        f"Max: {desc.get('max', float('nan')):.2f}, "
                                        f"Count: {int(desc.get('count', 0))}")
                    elif not temp_data_for_stats.empty: # Categorical summary
                        counts = temp_data_for_stats.value_counts()
                        top_categories = [f"'{str(i)}' ({counts[i]})" for i in counts.head(3).index]
                        summary_text = (f"**Summary for `{primary_plot_col}`:** {counts.size} unique values. "
                                        f"Top: {', '.join(top_categories)}.")

                    if summary_text: response_text += f"\n\n{summary_text}"


            elif action == "answer":
                # Use the original LLM for answering the question based on data context
                query_to_answer = llm_output.get("query_for_llm", user_query) # Fallback to original query
                print(f"DEBUG: Invoking LLM for answering: {query_to_answer}")

                # Prepare context for the answering LLM
                col_context = f"Dataset has {len(current_data)} rows. Standardized columns: {', '.join(current_data.columns)}. "
                if 'column_mapping' in st.session_state and st.session_state['column_mapping']:
                     original_names = [st.session_state['column_mapping'].get(col, col) for col in current_data.columns]
                     col_context += f"Original column names might be similar to: {', '.join(original_names)}. "

                # Use a simple prompt for direct Q&A
                answer_prompt_text = """
                You are a helpful banking compliance assistant. Answer the user's question based on the provided context.
                Be concise and directly address the question.

                Context:
                {data_context}

                User Question: {user_question}

                Answer:
                """
                answer_prompt = PromptTemplate.from_template(answer_prompt_text)
                answer_chain = answer_prompt | llm_model | StrOutputParser()

                with st.spinner("🤔 Thinking..."):
                     ai_response_content = answer_chain.invoke({
                         "data_context": col_context,
                         "user_question": query_to_answer
                     })
                response_text = ai_response_content if ai_response_content else "Sorry, I couldn't formulate an answer."
                chart = None

            else:
                # Handle unexpected 'action' value from LLM
                response_text = f"Sorry, I received an unexpected instruction ('{action}') from the AI interpreter. Please rephrase your request."
                chart = None

        except json.JSONDecodeError as json_e:
            print(f"ERROR: LLM did not return valid JSON. Raw output: '{cleaned_json_str}'. Error: {json_e}")
            response_text = "Sorry, the AI interpreter gave an invalid response format. Trying to answer directly..."
            # Fallback: Try answering directly if JSON parsing fails
            try:
                col_context = f"Std columns: {', '.join(current_data.columns)}. "
                if 'column_mapping' in st.session_state and st.session_state['column_mapping']:
                    original_names = [st.session_state['column_mapping'].get(col, col) for col in current_data.columns]
                    col_context += f"Original likely: {', '.join(original_names)}. "
                answer_context = f"You are a compliance assistant. Data has {len(current_data)} rows. {col_context}. Answer user's question concisely:"
                # Use simple HumanMessage for fallback
                answer_messages = [HumanMessage(content=f"{answer_context}\n\nUser Question: {user_query}")]
                with st.spinner("🤔 Thinking (fallback)..."):
                    ai_response = llm_model.invoke(answer_messages)
                response_text = ai_response.content if ai_response and hasattr(ai_response, 'content') else "Sorry, I couldn't get an answer from the AI on fallback."
            except Exception as llm_e:
                response_text = f"Sorry, I could not process your request after an initial interpretation error. (Fallback LLM Error: {llm_e})"
            chart = None
        except ValueError as ve:
            # Catch validation errors (e.g., bad column name, too many pie slices)
            print(f"ERROR: Validation error after LLM interpretation: {ve}")
            response_text = f"❌ Error processing plot request: {ve}. Please check column names or try a different plot type."
            chart = None
        except Exception as e:
            # Catch-all for other unexpected errors during plotting or processing
            print(f"ERROR: Unexpected error processing LLM response or plotting: {e}")
            response_text = f"❌ An unexpected error occurred: {e}"
            chart = None

    except Exception as e:
        # Catch errors during the initial context preparation or LLM invocation
        print(f"ERROR: Failed to invoke LLM or prepare request context: {e}")
        response_text = f"❌ Failed to process your request due to an internal error: {e}"
        chart = None

    return response_text, chart

# === File Upload Options ===
st.sidebar.header("📤 Data Upload")
upload_method = st.sidebar.radio("Select upload method:",
                                 ["**Upload File (CSV/XLSX/JSON)**", "**Upload via Data Link**", "**Connect to DataBase**"],
                                 key="upload_method_radio")

# Initialize session state variables if they don't exist
if 'app_df' not in st.session_state: st.session_state.app_df = None
if 'data_processed' not in st.session_state: st.session_state.data_processed = False
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [{"role": "assistant", "content": "Hi! Please upload data first. Then ask compliance questions or request plots (e.g., 'show distribution of Account_Type', 'pie chart for Branch')."}]
if 'column_mapping' not in st.session_state: st.session_state.column_mapping = {} # Initialize column mapping

uploaded_data_source = None
if upload_method == "**Upload File (CSV/XLSX/JSON)**":
    # Use a consistent key for the uploader
    uploaded_file = st.sidebar.file_uploader("Upload Account Dataset", type=["csv", "xlsx", "xls", "json"], key="data_file_uploader")
    if uploaded_file:
        uploaded_data_source = uploaded_file
        st.sidebar.caption(f"Selected: {uploaded_file.name}")
elif upload_method == "**Upload via Data Link**":
    url_input = st.sidebar.text_input("Enter CSV file URL:", key="url_input")
    # Button to trigger the fetch
    if st.sidebar.button("Fetch Data from URL", key="fetch_url_button"):
        if url_input:
            try:
                with st.spinner("⏳ Fetching data from URL..."):
                    response = requests.get(url_input, timeout=30) # Add timeout
                    response.raise_for_status() # Check for HTTP errors
                    # Pass the raw text content to parse_data
                    uploaded_data_source = response.text
                    st.sidebar.success("✅ Fetched! Ready to process.")
                    # Clear potential old file selection if URL fetch is successful
                    if 'data_file_uploader' in st.session_state:
                         st.session_state.data_file_uploader = None
            except requests.exceptions.RequestException as e:
                st.sidebar.error(f"❌ URL Fetch Error: {e}")
            except Exception as e:
                st.sidebar.error(f"❌ Error processing URL data: {e}")
        else:
            st.sidebar.warning("⚠️ Please enter a valid URL")
elif upload_method == "Connect to Data Lake (Placeholder)":
    st.sidebar.text_input("Data Lake Connection String:", key="lake_conn_string_input", placeholder="e.g., JDBC or ODBC string")
    st.sidebar.text_input("Query or Table Name:", key="lake_query_input", placeholder="e.g., SELECT * FROM accounts WHERE region='X'")
    if st.sidebar.button("Connect & Load (Placeholder)", key="connect_lake_button"):
        st.sidebar.info("Data Lake connection is a placeholder. No data loaded.")
        # Clear potential old file/URL selection
        if 'data_file_uploader' in st.session_state:
            st.session_state.data_file_uploader = None
        uploaded_data_source = None # Ensure no data source is set

# Process Data Button Logic - enabled only if a source is potentially available
process_button_disabled = uploaded_data_source is None and upload_method != "**Connect to DataBase**" # Disable if no file/URL/text
if upload_method == "**Connect to DataBase**": process_button_disabled = True # Always disable for placeholder

process_clicked = st.sidebar.button("Process Uploaded/Fetched Data", key="process_data_button", disabled=process_button_disabled)

if process_clicked and uploaded_data_source is not None:
    with st.spinner("⏳ Processing and standardizing data..."):
        df_parsed = parse_data(uploaded_data_source)

    if df_parsed is not None and not df_parsed.empty:
        st.session_state.app_df = df_parsed
        st.session_state.data_processed = True
        # Provide example standardized columns in the initial chat message
        std_cols_example = ', '.join([f"`{c}`" for c in df_parsed.columns[:min(3, len(df_parsed.columns))]])
        initial_message = (f"Data ({len(df_parsed)} rows) processed successfully! "
                           f"Available standardized columns include: {std_cols_example}...\n"
                           f"Ask questions or request plots (e.g., 'Plot `Account_Status` counts', 'scatter plot `Account_ID` vs `Last_Transaction_Date` colored by `Branch`').")
        st.session_state.chat_messages = [{"role": "assistant", "content": initial_message}]

        # Attempt to save to DB after successful parsing
        with st.spinner("💾 Saving data to local database..."):
            save_success = save_to_db(st.session_state.app_df, db_path=db_name)

        if save_success:
            st.sidebar.success(f"✅ Dataset processed & saved to DB!")
        else:
            st.sidebar.error("Dataset processed, but failed to save to DB. Analysis will use in-memory data.")
            # Keep data_processed = True, as analysis can proceed in memory
    elif df_parsed is not None and df_parsed.empty:
        st.sidebar.error("The source contained data, but it resulted in an empty dataset after parsing.")
        st.session_state.data_processed = False
        st.session_state.app_df = None
    else: # Parsing failed (df_parsed is None)
        st.sidebar.error("❌ Data parsing failed. Please check the file format and content.")
        st.session_state.data_processed = False
        st.session_state.app_df = None

    # Rerun to update the main page content based on processing status
    st.rerun()

# === Main App Mode Selection ===
# Only allow mode selection if data is processed successfully
if st.session_state.data_processed:
    app_mode = st.sidebar.selectbox("Select Application Mode", [
        "🏦 Dormant Account Analyzer",
        "🔒 Compliance Multi-Agent",
        "🔍 SQL Bot",
        "💬 Chatbot Only" # Added Chatbot Only mode
    ], key="app_mode_selector")
else:
    app_mode = None # No mode selection if data isn't ready

# Display Title based on selected mode or lack thereof
if app_mode:
    st.title(f"{app_mode}")
else:
    st.title("Unified Banking Compliance Solution")

# === Main Application Area ===
# Only show main content if data is processed successfully
if st.session_state.data_processed and st.session_state.app_df is not None:
    # Use a copy of the dataframe for analysis to avoid modifying the session state directly
    current_df = st.session_state.app_df.copy()

    # --- Display Processed Data Sample (Optional) ---
    st.header("Data Overview")
    if st.checkbox("View Processed Dataset Sample", key="view_processed_data_checkbox"):
        
        display_df = current_df.head(100).copy()
        # Attempt to show original column names for display
        if 'column_mapping' in st.session_state and st.session_state['column_mapping']:
            try:
                # Create display names, falling back to standardized if mapping missing
                display_columns = {std_col: st.session_state['column_mapping'].get(std_col, std_col)
                                   for std_col in display_df.columns}
                display_df.rename(columns=display_columns, inplace=True)
                st.dataframe(display_df)
                st.caption("Displaying original column names where available.")
            except Exception as e:
                st.error(f"Error applying original column names for display: {e}")
                st.dataframe(current_df.head(100)) # Fallback to standardized
        else:
            st.dataframe(display_df)
            st.caption("Displaying standardized column names.")

    st.divider() # Separator

    # === Dormant Account Analyzer MODE ===
    if app_mode == "🏦 Dormant Account Analyzer":
        st.header("Dormant Account Analysis")
        # Define inactivity threshold (e.g., 3 years)
        # Ensure timezone awareness if necessary, though basic subtraction works here
        inactivity_threshold_days = st.number_input("Inactivity Threshold (days)", min_value=30, value=3*365, step=30, key="dormant_threshold_days")
        inactivity_threshold_date = datetime.now() - timedelta(days=inactivity_threshold_days)
        st.caption(f"Analyzing accounts with no transaction since: {inactivity_threshold_date.strftime('%Y-%m-%d')}")


        # --- Define Dormant Detection Helper Functions (USING STANDARDIZED NAMES) ---
        # These functions now return the filtered DataFrame along with count and description
        def check_safe_deposit(df, threshold):
            count = 0
            desc = "(Prerequisite check failed)"
            filtered_df = pd.DataFrame()
            required_cols = ['Account_Type', 'Last_Transaction_Date', 'Email_Contact_Attempt', 'SMS_Contact_Attempt', 'Phone_Call_Attempt', 'Account_ID'] # Need ID
            missing = [c for c in required_cols if c not in df.columns]
            if not missing:
                try:
                    # Ensure date comparison works (coerce errors during parsing)
                    df_filtered_logic = df[
                        (df['Account_Type'].str.contains("Safe Deposit", case=False, na=False)) &
                        (df['Last_Transaction_Date'].notna() & (df['Last_Transaction_Date'] < threshold)) &
                        (df['Email_Contact_Attempt'].str.lower().isin(['no', 'unknown', ''])) & # Check variations
                        (df['SMS_Contact_Attempt'].str.lower().isin(['no', 'unknown', ''])) &
                        (df['Phone_Call_Attempt'].str.lower().isin(['no', 'unknown', '']))
                    ]
                    filtered_df = df_filtered_logic.copy()
                    count = len(filtered_df)
                    desc = f"{count} Uncontacted Safe Deposit accounts inactive > {inactivity_threshold_days} days."
                except Exception as e:
                    desc = f"(Error during Safe Deposit check: {e})"
            else:
                desc = f"(Skipped: Missing standardized columns: {', '.join(missing)})"
            return filtered_df, count, desc

        def check_investment_inactivity(df, threshold):
            count = 0
            desc = "(Prerequisite check failed)"
            filtered_df = pd.DataFrame()
            required_cols = ['Account_Type', 'Last_Transaction_Date', 'Email_Contact_Attempt', 'SMS_Contact_Attempt', 'Phone_Call_Attempt', 'Account_ID'] # Need ID
            missing = [c for c in required_cols if c not in df.columns]
            if not missing:
                try:
                    df_filtered_logic = df[
                        (df['Account_Type'].str.contains("Investment", case=False, na=False)) &
                        (df['Last_Transaction_Date'].notna() & (df['Last_Transaction_Date'] < threshold)) &
                         (df['Email_Contact_Attempt'].str.lower().isin(['no', 'unknown', ''])) &
                        (df['SMS_Contact_Attempt'].str.lower().isin(['no', 'unknown', ''])) &
                        (df['Phone_Call_Attempt'].str.lower().isin(['no', 'unknown', '']))
                    ]
                    filtered_df = df_filtered_logic.copy()
                    count = len(filtered_df)
                    desc = f"{count} Uncontacted Investment accounts inactive > {inactivity_threshold_days} days."
                except Exception as e:
                    desc = f"(Error during Investment check: {e})"
            else:
                desc = f"(Skipped: Missing standardized columns: {', '.join(missing)})"
            return filtered_df, count, desc

        def check_fixed_deposit_inactivity(df, threshold):
            count = 0
            desc = "(Prerequisite check failed)"
            filtered_df = pd.DataFrame()
            required_cols = ['Account_Type', 'Last_Transaction_Date', 'Account_ID'] # Need ID
            missing = [c for c in required_cols if c not in df.columns]
            if not missing:
                 try:
                    df_filtered_logic = df[
                        (df['Account_Type'].str.contains("fixed", case=False, na=False)) &
                        (df['Account_Type'].str.contains("deposit", case=False, na=False)) &
                        (df['Last_Transaction_Date'].notna() & (df['Last_Transaction_Date'] < threshold))
                    ]
                    filtered_df = df_filtered_logic.copy()
                    count = len(filtered_df)
                    desc = f"{count} Fixed Deposit accounts inactive > {inactivity_threshold_days} days."
                 except Exception as e:
                    desc = f"(Error during Fixed Deposit check: {e})"
            else:
                desc = f"(Skipped: Missing standardized columns: {', '.join(missing)})"
            return filtered_df, count, desc

        def check_general_inactivity(df, threshold):
            count = 0
            desc = "(Prerequisite check failed)"
            filtered_df = pd.DataFrame()
            # Regex for common account types, case-insensitive
            general_types_regex = r'savings|call|current'
            required_cols = ['Account_Type', 'Last_Transaction_Date', 'Account_ID'] # Need ID
            missing = [c for c in required_cols if c not in df.columns]
            if not missing:
                try:
                    df_filtered_logic = df[
                        (df['Account_Type'].str.contains(general_types_regex, case=False, na=False, regex=True)) &
                        (df['Last_Transaction_Date'].notna() & (df['Last_Transaction_Date'] < threshold))
                    ]
                    filtered_df = df_filtered_logic.copy()
                    count = len(filtered_df)
                    desc = f"{count} Savings/Call/Current accounts inactive > {inactivity_threshold_days} days."
                except Exception as e:
                    desc = f"(Error during General Inactivity check: {e})"
            else:
                desc = f"(Skipped: Missing standardized columns: {', '.join(missing)})"
            return filtered_df, count, desc

        def check_unreachable_dormant(df):
            count = 0
            desc = "(Prerequisite check failed)"
            filtered_df = pd.DataFrame()
            required_cols = ['Email_Contact_Attempt', 'SMS_Contact_Attempt', 'Phone_Call_Attempt', 'Account_Status', 'Account_ID'] # Need ID
            missing = [c for c in required_cols if c not in df.columns]
            if not missing:
                try:
                    df_filtered_logic = df[
                        (df['Email_Contact_Attempt'].str.lower().isin(['no', 'unknown', ''])) &
                        (df['SMS_Contact_Attempt'].str.lower().isin(['no', 'unknown', ''])) &
                        (df['Phone_Call_Attempt'].str.lower().isin(['no', 'unknown', ''])) &
                        (df['Account_Status'].str.lower().str.strip() == 'dormant') # Exact match for status usually
                    ]
                    filtered_df = df_filtered_logic.copy()
                    count = len(filtered_df)
                    desc = f"{count} accounts marked 'Dormant' and failed all known contact attempts."
                except Exception as e:
                    desc = f"(Error checking Unreachable/Dormant: {e})"
            else:
                desc = f"(Skipped: Missing standardized columns: {', '.join(missing)})"
            return filtered_df, count, desc

        # --- Analysis Type Selection ---
        analysis_options = ["📊 Summarized Dormant Analysis",
                            "🔐 Safe Deposit Box",
                            "💼 Investment Inactivity",
                            "🏦 Fixed Deposit",
                            "📉 General Inactivity(Savings/Call/Current)",
                            "📵 Unreachable + Marked Dormant"]
        selected_analysis = st.selectbox("Select Dormant Analysis Type", analysis_options, key="dormant_analysis_selector")

        # --- Execute Selected Analysis ---
        if selected_analysis == "📊 Summarized Dormant Analysis":
            st.subheader("📈 Summarized Dormant Analysis Results")
            if st.button("📊 Run Summarized Dormant Analysis", key="run_summary_dormant_button"):
                # Use session state to store results to avoid recalculating on widget interaction
                if 'dormant_summary_results' not in st.session_state or st.session_state.get('dormant_summary_rerun', True):
                    with st.spinner("Running all dormant checks..."):
                        sd_df, sd_count, sd_desc = check_safe_deposit(current_df, inactivity_threshold_date)
                        inv_df, inv_count, inv_desc = check_investment_inactivity(current_df, inactivity_threshold_date)
                        fd_df, fd_count, fd_desc = check_fixed_deposit_inactivity(current_df, inactivity_threshold_date)
                        gen_df, gen_count, gen_desc = check_general_inactivity(current_df, inactivity_threshold_date)
                        unr_df, unr_count, unr_desc = check_unreachable_dormant(current_df)

                        # Store results in session state
                        st.session_state.dormant_summary_results = {
                            "sd": {"df": sd_df, "count": sd_count, "desc": sd_desc},
                            "inv": {"df": inv_df, "count": inv_count, "desc": inv_desc},
                            "fd": {"df": fd_df, "count": fd_count, "desc": fd_desc},
                            "gen": {"df": gen_df, "count": gen_count, "desc": gen_desc},
                            "unr": {"df": unr_df, "count": unr_count, "desc": unr_desc},
                            "total_accounts": len(current_df)
                        }
                        st.session_state.dormant_summary_rerun = False # Mark as run

                # Display results from session state
                results = st.session_state.dormant_summary_results
                st.subheader("🔢 Numerical Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"Uncontacted Safe Deposit (>{inactivity_threshold_days}d)", results["sd"]["count"], help=results["sd"]["desc"])
                    st.metric(f"General Inactivity (>{inactivity_threshold_days}d)", results["gen"]["count"], help=results["gen"]["desc"])
                with col2:
                    st.metric(f"Uncontacted Investment (>{inactivity_threshold_days}d)", results["inv"]["count"], help=results["inv"]["desc"])
                    st.metric("Unreachable & 'Dormant'", results["unr"]["count"], help=results["unr"]["desc"])
                with col3:
                    st.metric(f"Fixed Deposit Inactivity (>{inactivity_threshold_days}d)", results["fd"]["count"], help=results["fd"]["desc"])

                # Combine descriptions for AI summary
                summary_input_text = (f"Dormant Analysis Findings ({results['total_accounts']} total accounts analyzed, threshold >{inactivity_threshold_days} days inactive):\n"
                                    f"- {results['sd']['desc']}\n"
                                    f"- {results['inv']['desc']}\n"
                                    f"- {results['fd']['desc']}\n"
                                    f"- {results['gen']['desc']}\n"
                                    f"- {results['unr']['desc']}")

                st.subheader("📝 Narrative Summary")
                # AI Summary Generation
                if llm and LANGCHAIN_AVAILABLE:
                    try:
                        with st.spinner("Generating AI Summary..."):
                             # More specific prompt for compliance summary
                            summary_prompt_template = PromptTemplate.from_template(
                                """Act as a Senior Compliance Analyst AI. Based on the following automated checks on {total_accounts} accounts, provide a concise executive summary highlighting key risk areas, potential compliance gaps, and overall dormancy trends. Focus on actionable insights.

Analysis Findings:
{analysis_details}

Executive Summary:"""
                            )
                            summary_chain = summary_prompt_template | llm | StrOutputParser()
                            narrative_summary = summary_chain.invoke({
                                "total_accounts": results['total_accounts'],
                                "analysis_details": summary_input_text
                                })
                        st.markdown(narrative_summary)
                        st.session_state.dormant_narrative_summary = narrative_summary # Save for PDF
                    except Exception as llm_e:
                        st.error(f"AI summary generation failed: {llm_e}")
                        st.text_area("Raw Findings:", summary_input_text, height=150)
                        st.session_state.dormant_narrative_summary = f"AI Summary Failed. Raw Findings:\n{summary_input_text}" # Save fallback for PDF
                else:
                    st.warning("AI Assistant not available. Displaying raw findings.")
                    st.text_area("Raw Findings:", summary_input_text, height=150)
                    st.session_state.dormant_narrative_summary = f"AI Not Available. Raw Findings:\n{summary_input_text}" # Save raw for PDF

                # PDF Export Button (Integrated from v3.py) - Only show if summary ran
                st.subheader("⬇️ Export Summary")
                if st.button("📄 Download Summary Report (PDF)", key="download_dormant_summary_pdf"):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(0, 10, "Dormant Account Analysis Summary Report", 0, 1, 'C')
                    pdf.ln(5)

                    # Add Numerical Summary
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, "Numerical Summary", 0, 1)
                    pdf.set_font("Arial", size=10)
                    pdf.multi_cell(0, 6, f"- {results['sd']['desc']}")
                    pdf.multi_cell(0, 6, f"- {results['inv']['desc']}")
                    pdf.multi_cell(0, 6, f"- {results['fd']['desc']}")
                    pdf.multi_cell(0, 6, f"- {results['gen']['desc']}")
                    pdf.multi_cell(0, 6, f"- {results['unr']['desc']}")
                    pdf.ln(5)

                     # Add Narrative Summary
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, "Narrative Summary (AI Generated or Raw Findings)", 0, 1)
                    pdf.set_font("Arial", size=10)
                    # Use utf-8 encoding friendly method for multi_cell
                    narrative_text = st.session_state.get('dormant_narrative_summary', "Summary not generated.")
                    # Encode to latin-1, ignoring errors, as FPDF default font doesn't support all UTF-8
                    pdf.multi_cell(0, 6, narrative_text.encode('latin-1', 'replace').decode('latin-1'))
                    pdf.ln(5)

                    pdf_file_name = f"dormant_summary_report_{datetime.now().strftime('%Y%m%d')}.pdf"
                    try:
                        # Output PDF to bytes
                        pdf_output = pdf.output(dest='S').encode('latin-1') # Use latin-1 encoding
                        st.download_button(
                            label="Click to Download PDF",
                            data=pdf_output,
                            file_name=pdf_file_name,
                            mime="application/pdf"
                        )
                        st.success("PDF prepared for download.")
                    except Exception as pdf_e:
                        st.error(f"Error generating PDF: {pdf_e}")


        else: # Individual Dormant Agent Logic
            st.subheader(f"Detailed Analysis: {selected_analysis}")
            data_filtered = pd.DataFrame()
            agent_desc = "No analysis performed."
            agent_executed = False

            # Map selection to function and run
            agent_mapping = {
                "🔐 Safe Deposit Box": check_safe_deposit,
                "💼 Investment Inactivity": check_investment_inactivity,
                "🏦 Fixed Deposit": check_fixed_deposit_inactivity,
                "📉 General Inactivity(Savings/Call/Current)": check_general_inactivity,
                "📵 Unreachable + Marked Dormant": check_unreachable_dormant
            }

            if selected_analysis in agent_mapping:
                 with st.spinner(f"Running {selected_analysis}..."):
                      analysis_func = agent_mapping[selected_analysis]
                      # Call the appropriate function with the threshold if needed
                      if selected_analysis == "📵 Unreachable + Marked Dormant Agent":
                           data_filtered, count, agent_desc = analysis_func(current_df)
                      else:
                           data_filtered, count, agent_desc = analysis_func(current_df, inactivity_threshold_date)
                      agent_executed = True
                      st.metric(f"Accounts Detected by Agent", count, help=agent_desc)
            else:
                st.warning(f"Logic for '{selected_analysis}' seems to be missing or not mapped correctly.")

            if agent_executed:
                if not data_filtered.empty:
                    st.success(f"{len(data_filtered)} accounts detected matching criteria.")
                    # Option to view detected accounts
                    if st.checkbox(f"View detected accounts sample (max 25)", key=f"view_{selected_analysis}_data"):
                        display_filtered_df = data_filtered.head(25).copy()
                        # Apply original names if possible
                        if 'column_mapping' in st.session_state and st.session_state['column_mapping']:
                             try:
                                 display_columns = {std: st.session_state['column_mapping'].get(std, std) for std in display_filtered_df.columns}
                                 display_filtered_df.rename(columns=display_columns, inplace=True)
                             except Exception as display_e:
                                 st.warning(f"Could not map original names for display: {display_e}")
                        st.dataframe(display_filtered_df)

                    # Download button for the filtered list
                    try:
                         csv_data = data_filtered.to_csv(index=False).encode('utf-8')
                         agent_file_name = f"{selected_analysis.replace(' ', '_').lower()}_detected_accounts.csv"
                         st.download_button(label=f"📁 Download Full List ({len(data_filtered)} rows)",
                                            data=csv_data,
                                            file_name=agent_file_name,
                                            mime='text/csv',
                                            key=f"download_{selected_analysis}_csv")
                    except Exception as csv_e:
                         st.error(f"Failed to generate CSV for download: {csv_e}")


                    # AI Insights for the specific agent's results
                    st.subheader("🤖 Detailed AI Insights for Detected Accounts")
                    if st.button(f"Generate Detailed Insights", key=f"gen_insights_{selected_analysis}_button"):
                        if llm and LANGCHAIN_AVAILABLE:
                            with st.spinner("🧠 Running detailed insight generation..."):
                                try:
                                    # Use a sample for large datasets to avoid overwhelming LLM/token limits
                                    sample_size = min(50, len(data_filtered))
                                    sample_data_csv = data_filtered.sample(n=sample_size, random_state=1).to_csv(index=False) # Use random_state for consistency if needed
                                    data_columns = ", ".join(data_filtered.columns)

                                    # Define prompts using f-strings for dynamic content (safer within controlled env)
                                    obs_prompt_text = f"""Analyze this SAMPLE data ({sample_size} rows) for accounts detected by the '{selected_analysis}'. Focus on patterns within this group related to Branch, Account Type, KYC Status, or other available columns. Columns available: {data_columns}.

Sample Data (CSV format):
{{data}}

Key Observations within this detected group:"""
                                    trend_prompt_text = f"""Based on the sample data from '{selected_analysis}', identify potential micro-trends or common characteristics among these specific accounts. Are there clusters or notable sub-groups?

Sample Data (CSV format):
{{data}}

Potential Trends within this group:"""
                                    insight_prompt_text = f"""Synthesize the observations and trends from the '{selected_analysis}' data into actionable insights. What are the implications for compliance or operations regarding this specific group?

Observations:
{{observation}}

Trends:
{{trend}}

Synthesized Insights:"""
                                    action_prompt_text = f"""Based on the insights derived from the '{selected_analysis}' data, recommend specific, actionable steps for the bank to take regarding these detected accounts. Prioritize actions based on risk or regulatory impact.

Insights:
{{insight}}

Recommended Actions for this group:"""

                                    # Create chains
                                    obs_chain = PromptTemplate.from_template(obs_prompt_text) | llm | StrOutputParser()
                                    trend_chain = PromptTemplate.from_template(trend_prompt_text) | llm | StrOutputParser()
                                    insight_chain = PromptTemplate.from_template(insight_prompt_text) | llm | StrOutputParser()
                                    action_chain = PromptTemplate.from_template(action_prompt_text) | llm | StrOutputParser()

                                    # Invoke chains sequentially
                                    obs_output = obs_chain.invoke({"data": sample_data_csv})
                                    trend_output = trend_chain.invoke({"data": sample_data_csv})
                                    insight_output = insight_chain.invoke({"observation": obs_output, "trend": trend_output})
                                    action_output = action_chain.invoke({"insight": insight_output})

                                    # Display results in expanders
                                    with st.expander("🔍 AI Observation (Based on Sample)"): st.markdown(obs_output)
                                    with st.expander("📊 AI Trend Analysis (Based on Sample)"): st.markdown(trend_output)
                                    with st.expander("💡 AI Insight"): st.markdown(insight_output)
                                    with st.expander("🚀 AI Recommended Actions"): st.markdown(action_output)

                                    # Save the generated summary to the log
                                    if save_summary_to_db(f"Observation for {selected_analysis} (Sample)", trend_output, insight_output, action_output, db_name):
                                        st.success("Analysis summary saved to insight log.")
                                    else:
                                        st.error("Failed to save summary to insight log.")

                                except Exception as insight_e:
                                    st.error(f"Detailed insight generation failed: {insight_e}")
                        else:
                            st.warning("AI Assistant not available. Cannot generate detailed insights.")
                elif agent_desc.startswith("(Skipped") or agent_desc.startswith("(Error"):
                     st.warning(f"Agent could not run successfully. Reason: {agent_desc}")
                else: # Agent ran but found no accounts
                    st.info(f"No accounts detected matching the criteria for {selected_analysis}.")

    # === Compliance Multi-Agent MODE ===
    elif app_mode == "🔒 Compliance Multi-Agent":
        st.header("Compliance Agent Actions & Analysis")
        st.caption("These agents identify accounts based on compliance rules. Actions like flagging/freezing are typically performed in core banking systems after review.")

        # --- Define Compliance Detection Helper Functions (USING STANDARDIZED NAMES) ---
        # Return filtered DataFrame, count, and description. Focus on DETECTION.
        def detect_incomplete_contact(df):
            count = 0; desc = "(Prerequisite check failed)"; filtered_df = pd.DataFrame()
            req = ["Email_Contact_Attempt", "SMS_Contact_Attempt", "Phone_Call_Attempt", "Account_ID"]
            missing = [c for c in req if c not in df.columns]
            if not missing:
                 try:
                     # Identify rows where NOT ALL attempts are 'yes' (case-insensitive)
                     df_filtered_logic = df[~((df[req[0]].str.lower().str.strip() == 'yes') & (df[req[1]].str.lower().str.strip() == 'yes') & (df[req[2]].str.lower().str.strip() == 'yes'))]
                     filtered_df = df_filtered_logic.copy()
                     count = len(filtered_df)
                     desc = f"{count} accounts may have incomplete contact attempts (not all Email, SMS, Phone marked 'yes'). Review needed."
                 except Exception as e: desc = f"(Error checking contact attempts: {e})"
            else: desc = f"(Skipped: Missing std columns: {', '.join(missing)})"
            return filtered_df, count, desc

        def detect_flag_candidates(df, threshold_date):
             count = 0; desc = "(Prerequisite check failed)"; filtered_df = pd.DataFrame()
             req = ["Account_Status", "Last_Transaction_Date", "Account_ID"]
             missing = [c for c in req if c not in df.columns]
             if not missing:
                 try:
                     # Condition: Status is 'dormant' OR Last Transaction is before threshold
                     df_filtered_logic = df[
                         (df['Account_Status'].str.lower().str.strip() == 'dormant') |
                         (df['Last_Transaction_Date'].notna() & (df['Last_Transaction_Date'] < threshold_date))
                     ]
                     filtered_df = df_filtered_logic.copy()
                     count = len(filtered_df)
                     desc = f"{count} accounts identified as candidates for dormant flagging (status is 'dormant' or inactive > {inactivity_threshold_days} days)."
                 except Exception as e: desc = f"(Error checking flag candidates: {e})"
             else: desc = f"(Skipped: Missing std columns: {', '.join(missing)})"
             return filtered_df, count, desc

        def detect_ledger_candidates(df):
            count = 0; desc = "(Prerequisite check failed)"; filtered_df = pd.DataFrame()
            req = ['Account_Status', 'Account_ID']
            missing = [c for c in req if c not in df.columns]
            if not missing:
                try:
                    # Simple check for 'dormant' status for ledger review
                    df_filtered_logic = df[df['Account_Status'].str.lower().str.strip() == 'dormant']
                    filtered_df = df_filtered_logic.copy()
                    count = len(filtered_df)
                    desc = f"{count} accounts currently marked 'Dormant' identified for potential ledger reclassification review."
                except Exception as e: desc = f"(Error checking ledger candidates: {e})"
            else: desc = f"(Skipped: Missing std columns: {', '.join(missing)})"
            return filtered_df, count, desc

        def detect_freeze_candidates(df, inactive_threshold_date):
             count = 0; desc = "(Prerequisite check failed)"; filtered_df = pd.DataFrame()
             # Example criteria: Dormant status, inactive N years, expired KYC
             req = ["Account_Status", "Last_Transaction_Date", "KYC_Status", "Account_ID"]
             missing = [c for c in req if c not in df.columns]
             if not missing:
                 try:
                     freeze_threshold_days = (datetime.now() - inactive_threshold_date).days
                     df_filtered_logic = df[
                         (df['Account_Status'].str.lower().str.strip() == 'dormant') &
                         (df['Last_Transaction_Date'].notna() & (df['Last_Transaction_Date'] < inactive_threshold_date)) &
                         (df['KYC_Status'].str.lower().str.strip() == 'expired') # Needs exact 'expired' status
                     ]
                     filtered_df = df_filtered_logic.copy()
                     count = len(filtered_df)
                     desc = f"{count} accounts meet potential freeze criteria (Status: Dormant, Inactive > {freeze_threshold_days} days, KYC: Expired)."
                 except Exception as e: desc = f"(Error checking freeze candidates: {e})"
             else: desc = f"(Skipped: Missing std columns: {', '.join(missing)})"
             return filtered_df, count, desc

        def detect_transfer_candidates(df, cutoff_date):
             count = 0; desc = "(Prerequisite check failed)"; filtered_df = pd.DataFrame()
             req = ['Last_Transaction_Date', 'Account_ID']
             missing = [c for c in req if c not in df.columns]
             if not missing:
                 if cutoff_date and isinstance(cutoff_date, datetime):
                     try:
                         # Find accounts with last transaction ON or BEFORE the cutoff date
                         df_filtered_logic = df[
                             pd.notna(df['Last_Transaction_Date']) & (df['Last_Transaction_Date'] <= cutoff_date)
                         ]
                         filtered_df = df_filtered_logic.copy()
                         count = len(filtered_df)
                         cutoff_str = cutoff_date.strftime('%Y-%m-%d')
                         desc = f"{count} accounts require review for potential CBUAE transfer (last transaction <= {cutoff_str})."
                     except Exception as e: desc = f"(Error checking transfer candidates: {e})"
                 else: desc = "(Skipped: Invalid CBUAE cutoff date provided)"
             else: desc = f"(Skipped: Missing std columns: {', '.join(missing)})"
             return filtered_df, count, desc

        # --- Agent Selection ---
        agent_options = ["📊 Summarized Compliance Detection",
                         "📨 Contact Attempt Verification Agent",
                         "🚩 Flag Dormant Candidate Agent",
                         "📘 Dormant Ledger Review Agent",
                         "❄️ Account Freeze Candidate Agent",
                         "🏦 CBUAE Transfer Candidate Agent"]
        selected_agent = st.selectbox("Select Compliance Task or Summary", agent_options, key="compliance_agent_selector")

        # Define thresholds (make them configurable if needed)
        # Use the same inactivity threshold as dormant analysis for flagging consistency
        inactivity_threshold_days = 3 * 365
        general_inactivity_threshold_date = datetime.now() - timedelta(days=inactivity_threshold_days)
        # Freeze might have a different threshold (e.g., 2 years)
        freeze_inactivity_threshold_days = st.number_input("Freeze Inactivity Threshold (days)", min_value=30, value=2*365, step=30, key="freeze_threshold_days")
        freeze_inactivity_threshold_date = datetime.now() - timedelta(days=freeze_inactivity_threshold_days)
        # CBUAE Cutoff Date (configurable)
        default_cbuae_date = "2020-04-24"
        cbuae_cutoff_str = st.text_input("CBUAE Transfer Cutoff Date (YYYY-MM-DD)", value=default_cbuae_date, key="cbuae_cutoff_date")
        try:
            cbuae_cutoff_date = datetime.strptime(cbuae_cutoff_str, "%Y-%m-%d")
            st.caption(f"Using CBUAE cutoff: {cbuae_cutoff_date.strftime('%Y-%m-%d')}")
        except ValueError:
            cbuae_cutoff_date = None
            st.error("Invalid CBUAE cutoff date format. Please use YYYY-MM-DD.")


        # --- Execute Selected Agent Logic ---
        if selected_agent == "📊 Summarized Compliance Detection":
            st.subheader("📈 Summarized Compliance Detection Results")
            if st.button("📊 Run Summarized Compliance Analysis", key="run_summary_compliance_button"):
                 if 'compliance_summary_results' not in st.session_state or st.session_state.get('compliance_summary_rerun', True):
                    with st.spinner("Running all compliance checks..."):
                        contact_df, contact_count, contact_desc = detect_incomplete_contact(current_df)
                        flag_df, flag_count, flag_desc = detect_flag_candidates(current_df, general_inactivity_threshold_date)
                        ledger_df, ledger_count, ledger_desc = detect_ledger_candidates(current_df)
                        freeze_df, freeze_count, freeze_desc = detect_freeze_candidates(current_df, freeze_inactivity_threshold_date)
                        transfer_df, transfer_count, transfer_desc = detect_transfer_candidates(current_df, cbuae_cutoff_date)

                        st.session_state.compliance_summary_results = {
                            "contact": {"df": contact_df, "count": contact_count, "desc": contact_desc},
                            "flag": {"df": flag_df, "count": flag_count, "desc": flag_desc},
                            "ledger": {"df": ledger_df, "count": ledger_count, "desc": ledger_desc},
                            "freeze": {"df": freeze_df, "count": freeze_count, "desc": freeze_desc},
                            "transfer": {"df": transfer_df, "count": transfer_count, "desc": transfer_desc},
                            "total_accounts": len(current_df)
                        }
                        st.session_state.compliance_summary_rerun = False # Mark as run

                 # Display results
                 results = st.session_state.compliance_summary_results
                 st.subheader("🔢 Numerical Summary")
                 col1, col2, col3 = st.columns(3)
                 with col1:
                    st.metric("Incomplete Contact Checks", results["contact"]["count"], help=results["contact"]["desc"])
                    st.metric("Freeze Candidates", results["freeze"]["count"], help=results["freeze"]["desc"])
                 with col2:
                    st.metric("Flag Candidates", results["flag"]["count"], help=results["flag"]["desc"])
                    st.metric("Transfer Candidates", results["transfer"]["count"], help=results["transfer"]["desc"])
                 with col3:
                    st.metric("Ledger Review Candidates", results["ledger"]["count"], help=results["ledger"]["desc"])

                 # Combine descriptions for AI
                 summary_input = (f"Compliance Analysis Results ({results['total_accounts']} total accounts):\n"
                                  f"- {results['contact']['desc']}\n"
                                  f"- {results['flag']['desc']}\n"
                                  f"- {results['ledger']['desc']}\n"
                                  f"- {results['freeze']['desc']}\n"
                                  f"- {results['transfer']['desc']}")

                 st.subheader("📝 Narrative Summary (AI Generated)")
                 if llm and LANGCHAIN_AVAILABLE:
                     try:
                         with st.spinner("Generating AI Compliance Summary..."):
                             summary_prompt = PromptTemplate.from_template(
                                """Act as a Chief Compliance Officer AI. Review the following automated compliance checks ({total_accounts} accounts analyzed). Provide a high-level summary focusing on potential regulatory risks, areas needing immediate attention, and process improvement suggestions.

Analysis Findings:
{analysis_details}

Compliance Officer Summary:""")
                             summary_chain = summary_prompt | llm | StrOutputParser()
                             narrative_summary = summary_chain.invoke({"total_accounts": results['total_accounts'], "analysis_details": summary_input})
                         st.markdown(narrative_summary)
                     except Exception as llm_e:
                         st.error(f"AI summary failed: {llm_e}")
                         st.text_area("Raw Findings:", summary_input, height=150)
                 else:
                     st.warning("AI Assistant not available.")
                     st.text_area("Raw Findings:", summary_input, height=150)

        else: # Individual Compliance Agent Logic
            st.subheader(f"Agent Task Results: {selected_agent}")
            data_filtered = pd.DataFrame()
            agent_desc = "No analysis performed."
            agent_executed = False

            agent_mapping = {
                "📨 Contact Attempt Verification Agent": detect_incomplete_contact,
                "🚩 Flag Dormant Candidate Agent": detect_flag_candidates,
                "📘 Dormant Ledger Review Agent": detect_ledger_candidates,
                "❄️ Account Freeze Candidate Agent": detect_freeze_candidates,
                "🏦 CBUAE Transfer Candidate Agent": detect_transfer_candidates
            }

            if selected_agent in agent_mapping:
                 with st.spinner(f"Running {selected_agent}..."):
                    analysis_func = agent_mapping[selected_agent]
                    # Pass correct arguments based on agent
                    if selected_agent == "🚩 Flag Dormant Candidate Agent":
                         data_filtered, count, agent_desc = analysis_func(current_df, general_inactivity_threshold_date)
                    elif selected_agent == "❄️ Account Freeze Candidate Agent":
                         data_filtered, count, agent_desc = analysis_func(current_df, freeze_inactivity_threshold_date)
                    elif selected_agent == "🏦 CBUAE Transfer Candidate Agent":
                         data_filtered, count, agent_desc = analysis_func(current_df, cbuae_cutoff_date)
                    else: # Contact, Ledger
                         data_filtered, count, agent_desc = analysis_func(current_df)
                    agent_executed = True
                    st.metric(f"Accounts Identified by Agent", count, help=agent_desc)

            if agent_executed:
                st.markdown(f"**Agent Description:** {agent_desc}")
                if not data_filtered.empty:
                    st.success(f"{len(data_filtered)} accounts identified.")
                    if st.checkbox(f"View identified accounts sample (max 25)", key=f"view_{selected_agent}_data"):
                        display_filtered_df = data_filtered.head(25).copy()
                        if 'column_mapping' in st.session_state and st.session_state['column_mapping']:
                             try:
                                 display_columns = {std: st.session_state['column_mapping'].get(std, std) for std in display_filtered_df.columns}
                                 display_filtered_df.rename(columns=display_columns, inplace=True)
                             except Exception as display_e: st.warning(f"Could not map names: {display_e}")
                        st.dataframe(display_filtered_df)

                    # Download button
                    try:
                         csv_data = data_filtered.to_csv(index=False).encode('utf-8')
                         agent_file_name = f"{selected_agent.replace(' ', '_').lower()}_identified_accounts.csv"
                         st.download_button(label=f"📁 Download Full List ({len(data_filtered)} rows)",
                                            data=csv_data,
                                            file_name=agent_file_name,
                                            mime='text/csv',
                                            key=f"download_{selected_agent}_csv")
                    except Exception as csv_e:
                        st.error(f"Failed to generate CSV for download: {csv_e}")

                    # Placeholder for actions - Emphasize manual review
                    st.subheader("ℹ️ Next Steps (Manual Review Recommended)")
                    st.info(f"""
                    The list above identifies candidates based on defined rules. **No automated actions (e.g., flagging, freezing) have been performed.**
                    Recommended next steps:
                    1. **Download** the list for detailed review.
                    2. **Validate** the findings against account details in the core banking system.
                    3. **Perform** necessary actions (flagging, contact updates, freezing, ledger changes, transfer preparation) through established operational procedures.
                    4. **Document** all actions taken for audit purposes.
                    """)
                    # Example: Button to *simulate* adding to flag table (but don't modify core data)
                    if selected_agent == "🚩 Flag Dormant Candidate Agent":
                        if st.button("Log Flagging Instruction to DB (for Audit)", key="log_flag_instruction"):
                             try:
                                 with sqlite3.connect(db_name) as conn_flag:
                                     cursor_flag = conn_flag.cursor()
                                     flagged_ids = data_filtered['Account_ID'].tolist()
                                     timestamp = datetime.now().isoformat()
                                     instructions = [(acc_id, f"Identified by {selected_agent} for flagging review", timestamp) for acc_id in flagged_ids]
                                     # Use INSERT OR IGNORE to avoid errors if account already logged
                                     cursor_flag.executemany("INSERT OR IGNORE INTO dormant_flags (account_id, flag_instruction, timestamp) VALUES (?, ?, ?)", instructions)
                                     st.success(f"Logged instructions for {len(instructions)} accounts to 'dormant_flags' table for review.")
                             except Exception as db_e:
                                 st.error(f"Failed to log flag instructions: {db_e}")

                elif agent_desc.startswith("(Skipped") or agent_desc.startswith("(Error"):
                     st.warning(f"Agent could not run successfully. Reason: {agent_desc}")
                else: # Agent ran but found no accounts
                    st.info(f"No accounts identified matching the criteria for {selected_agent}.")


    # === SQL BOT MODE ===
    elif app_mode == "🔍 SQL Bot":
        st.header("SQL Database Query Bot")
        st.caption(f"Querying the local SQLite database: `{db_name}`")

        if not llm or not LANGCHAIN_AVAILABLE:
             st.warning("AI Assistant (Groq/Langchain) is needed for Natural Language to SQL conversion. Please configure the API key.")
        else:
            # Get database schema info function
            @st.cache_data(show_spinner="Fetching database schema...") # Cache schema
            def get_db_schema(db_path=DB_NAME):
                schema_info = {}
                try:
                    with sqlite3.connect(db_path) as conn_schema:
                        cursor = conn_schema.cursor()
                        # Get list of tables
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                        tables = cursor.fetchall()

                        for table in tables:
                            table_name = table[0]
                            cursor.execute(f"PRAGMA table_info('{table_name}');") # Use quotes for safety
                            columns = cursor.fetchall()
                            # Store column names and optionally types
                            schema_info[table_name] = [(col[1], col[2]) for col in columns] # (name, type)
                except sqlite3.Error as e:
                    st.error(f"Error fetching schema from {db_path}: {e}")
                    return None
                except Exception as e:
                    st.error(f"Unexpected error fetching schema: {e}")
                    return None
                return schema_info

            # Get DB schema
            schema = get_db_schema()

            if schema:
                # Format schema for prompt with types
                schema_text = "Database Schema (SQLite):\n"
                for table, columns in schema.items():
                    schema_text += f"Table: {table}\n"
                    col_defs = [f"- {name} ({dtype})" for name, dtype in columns]
                    schema_text += f"Columns:\n{chr(10).join(col_defs)}\n\n" # chr(10) is newline

                # Display schema if requested
                with st.expander("Show Database Schema"):
                    st.code(schema_text, language='text')

                # Create natural language to SQL chain
                nl_to_sql_prompt = PromptTemplate.from_template("""
You are an expert SQLite query generator. Based on the provided schema and the user's question, generate a SINGLE, valid SQLite SQL query.
- Ensure column names exactly match the schema (case-sensitive in some contexts, though SQLite is often case-insensitive for names).
- If dates are involved, they are likely stored as TEXT in ISO format (e.g., 'YYYY-MM-DD HH:MM:SS'). Use appropriate date functions if needed (e.g., `DATE()`, `STRFTIME()`).
- Only return the raw SQL query. Do not include explanations, markdown formatting (like ```sql), comments, or any other text.

Database Schema:
{schema}

Natural language question: {question}

SQLite Query:""")

                nl_to_sql_chain = nl_to_sql_prompt | llm | StrOutputParser()

                # Explanation chain for explaining the generated SQL (without results)
                sql_explanation_prompt = PromptTemplate.from_template("""
You are a data analyst explaining an SQL query to a non-technical compliance officer.
Explain what the following SQLite query intends to do, which tables/columns it uses, and what kind of information it aims to retrieve.
Do not execute the query or talk about specific results. Focus on the query's purpose based on the schema.

Schema:
{schema}

Query:
```sql
{sql_query}
```

Explanation:""")

                sql_explanation_chain = sql_explanation_prompt | llm | StrOutputParser()

                # Input for natural language query
                nl_query = st.text_area("Ask a database question in natural language:",
                                        placeholder="e.g., How many dormant accounts are there in the 'Dubai' branch?",
                                        height=100, key="sql_bot_nl_query")

                # Preview mode toggle
                generate_only = st.checkbox("Preview Mode (Generate SQL only, don't execute)", value=True, key="sql_bot_preview_mode")

                if nl_query:
                    # Use session state to store generated query to avoid re-generation on checkbox toggle
                    if 'generated_sql' not in st.session_state or st.session_state.get('last_nl_query') != nl_query:
                        st.session_state.last_nl_query = nl_query
                        try:
                            with st.spinner("🤖 Converting natural language to SQL..."):
                                # Clean the invocation arguments
                                invoke_args = {
                                    "schema": schema_text,
                                    "question": nl_query.strip() # Strip whitespace
                                }
                                sql_query_raw = nl_to_sql_chain.invoke(invoke_args)
                                # Clean potential markdown/extra text from LLM output
                                sql_query = re.sub(r"^```sql\s*|\s*```$", "", sql_query_raw, flags=re.MULTILINE).strip()
                                st.session_state.generated_sql = sql_query
                        except Exception as e:
                             st.error(f"Error during SQL generation: {e}")
                             st.session_state.generated_sql = None
                    else:
                         sql_query = st.session_state.generated_sql


                    if sql_query:
                        st.subheader("Generated SQL Query")
                        st.code(sql_query, language='sql')

                        # Explain what the query will do without executing it
                        if st.button("Analyze Generated Query", key="analyze_sql_button"):
                            if generate_only:
                                with st.spinner("🧠 Analyzing query purpose..."):
                                    try:
                                        explanation = sql_explanation_chain.invoke({
                                            "sql_query": sql_query,
                                            "schema": schema_text
                                        })
                                        st.subheader("Query Analysis")
                                        st.markdown(explanation)
                                    except Exception as e:
                                        st.error(f"Error generating query explanation: {e}")

                            # Add option to save this query for later
                            if st.button("Save Query to History", key="save_sql_history_button"):
                                try:
                                    with sqlite3.connect(db_name) as conn_hist:
                                        cursor = conn_hist.cursor()
                                        cursor.execute(
                                            "INSERT INTO sql_query_history (natural_language_query, sql_query) VALUES (?, ?)",
                                            (nl_query, sql_query))
                                    st.success("Query saved to history!")
                                except sqlite3.Error as e:
                                    st.error(f"Error saving query to history: {e}")
                        else:
                            # Execute the SQL query if not in preview mode
                            if st.button("Execute SQL Query", key="execute_sql_button"):
                                with st.spinner("⏳ Executing query..."):
                                    try:
                                        with sqlite3.connect(db_name) as conn_exec:
                                            results_df = pd.read_sql_query(sql_query, conn_exec)

                                            # Save query to history after successful execution
                                            cursor = conn_exec.cursor()
                                            cursor.execute(
                                                "INSERT INTO sql_query_history (natural_language_query, sql_query) VALUES (?, ?)",
                                                (nl_query, sql_query))
                                            # Commit happens automatically with context manager

                                        # Display results
                                        st.subheader("Query Results")
                                        if not results_df.empty:
                                            st.dataframe(results_df)
                                            st.info(f"Query returned {len(results_df)} rows.")

                                            # Optional: Generate explanation of results (can be slow/costly)
                                            # with st.spinner("Analyzing results..."):
                                            #     explanation = sql_explanation_chain.invoke({"sql_query": sql_query, "schema": schema_text})
                                            # st.subheader("Analysis")
                                            # st.write(explanation)

                                            # Download options
                                            try:
                                                csv = results_df.to_csv(index=False).encode("utf-8")
                                                st.download_button("📁 Download Results (CSV)",
                                                                   data=csv,
                                                                   file_name="sql_query_results.csv",
                                                                   mime="text/csv",
                                                                   key="download_sql_results_csv")
                                            except Exception as csv_e:
                                                st.error(f"Failed to generate CSV for download: {csv_e}")
                                        else:
                                            st.info("Query executed successfully but returned no results.")
                                    except pd.io.sql.DatabaseError as e:
                                         st.error(f"Database Error executing SQL: {e}. The generated query might be invalid for the schema.")
                                    except sqlite3.Error as e:
                                         st.error(f"SQLite Error executing SQL: {e}. Check the query syntax.")
                                    except Exception as e:
                                        st.error(f"An unexpected error occurred during query execution: {e}")

            else:
                 st.error("Could not retrieve database schema. SQL Bot functionality is limited.")


        # Display query history outside the generation block
        st.subheader("Query History")
        if st.checkbox("Show Recent SQL Queries", key="show_sql_history_checkbox"):
            try:
                with sqlite3.connect(db_name) as conn_hist_disp:
                    history_df = pd.read_sql_query("SELECT timestamp, natural_language_query, sql_query FROM sql_query_history ORDER BY timestamp DESC LIMIT 10", conn_hist_disp)

                if not history_df.empty:
                    for _, row in history_df.iterrows():
                        ts = pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                        with st.expander(f"Query at {ts}: \"{row['natural_language_query'][:50]}...\""):
                            st.code(row['sql_query'], language='sql')
                            # Add button to rerun query? Could be complex.
                else:
                    st.info("No queries recorded in history yet.")
            except sqlite3.Error as e:
                 st.error(f"Error retrieving query history: {e}")
            except Exception as e:
                 st.error(f"Unexpected error retrieving history: {e}")

    # === Chatbot Only MODE ===
    elif app_mode == "💬 Chatbot Only":
        st.header("Compliance Assistant Chatbot")
        st.caption("Ask questions about the loaded data or request plots.")
        # Chat interface is displayed below for all modes where data is loaded

    # --- Unified Chat Interface (Displayed if data is loaded, regardless of mode unless mode is SQL Bot only?) ---
    # Decide if chat should always be shown, or only in certain modes.
    # Showing it generally allows quick questions even when focusing on specific agent tasks.
    if app_mode != "🔍 SQL Bot": # Hide chat in SQL bot mode to avoid confusion? Or keep it? Let's keep it.
        st.divider()
        st.header("💬 Compliance Assistant Chatbot")
        st.caption(f"Interacting with {len(current_df)} rows. Use standardized column names like `Account_Type`, `Branch` etc.")

        chat_container = st.container(height=400) # Set height for scrollable container
        with chat_container:
            # Display chat messages from history
            for i, message in enumerate(st.session_state.chat_messages):
                with st.chat_message(message["role"]):
                    # Render markdown content safely
                    if isinstance(message.get("content"), str):
                        st.markdown(message["content"], unsafe_allow_html=False)
                    else:
                         # Handle potential non-string content gracefully
                         st.write(message.get("content", "[No content]"))

                    # Display chart if present in the message data
                    if message.get("chart"):
                         try:
                             st.plotly_chart(message["chart"], use_container_width=True)
                         except Exception as chart_e:
                             st.error(f"Could not display chart: {chart_e}")

        # --- Clear Chat Button ---
        if st.button("Clear Chat History", key="clear_chat_button"):
            # Reset chat history to initial welcome message
            std_cols_example = ', '.join([f"`{c}`" for c in current_df.columns[:min(3, len(current_df.columns))]])
            initial_message = (f"Chat history cleared. Data ({len(current_df)} rows) is loaded. "
                               f"Available standardized columns include: {std_cols_example}...\n"
                               f"Ask questions or request plots.")
            st.session_state.chat_messages = [{"role": "assistant", "content": initial_message}]
            st.rerun()

        # Chat Input - Placed below the container
        if prompt := st.chat_input("Ask about data or request a plot..."):
            # Add user message to state and display immediately (handled by rerun)
            st.session_state.chat_messages.append({"role": "user", "content": prompt})

            # Get response from the dynamic chatbot function
            # Ensure llm and current_df are available
            if llm and current_df is not None:
                assistant_response_text, chart_object = get_response_and_chart(
                    prompt, current_df, llm
                )
            elif not llm:
                 assistant_response_text = "AI Assistant is not available. Cannot process request."
                 chart_object = None
            else: # Should not happen if data_processed is True, but as safeguard
                 assistant_response_text = "Data seems unavailable for chat. Please re-process."
                 chart_object = None

            # Prepare assistant message data (handle potential None chart)
            assistant_message_data = {
                "role": "assistant",
                "content": assistant_response_text,
                "chart": chart_object # Store None if no chart
            }
            st.session_state.chat_messages.append(assistant_message_data)

            # Rerun to update the chat display
            st.rerun()

# Fallback message if data is not processed yet
elif not st.session_state.logged_in:
    st.info("Please log in using the sidebar to access the application.")
    # Login UI is handled at the top
else:
    # Data not processed, show welcome/instructions
    st.header("Welcome to the Unified Banking Compliance Solution")
    st.info("👋 Please upload or fetch data using the sidebar options and click 'Process' to begin analysis.")
    # Check if LLM failed to load due to missing key and add specific instruction
    if not llm and LANGCHAIN_AVAILABLE and hasattr(st, 'secrets') and not st.secrets.get("GROQ_API_KEY"):
         st.warning("AI features require a GROQ API key configured in your Streamlit secrets (`.streamlit/secrets.toml`).")
    elif not LANGCHAIN_AVAILABLE:
         st.warning("AI features are disabled because Langchain/Groq libraries are not installed (`pip install langchain langchain-groq`).")


# === Sidebar Bottom Section ===
st.sidebar.divider()

# === Database Status ===
st.sidebar.subheader("📊 Database Status")
# Provide more context about the counts
db_counts = {'Accounts': 0, 'Flags Logged': 0, 'Ledger Logged': 0, 'Insights Logged': 0, 'SQL History': 0}
error_messages = []
db_file_exists = os.path.exists(db_name)

if db_file_exists:
    try:
        # Use a single connection for status check
        with sqlite3.connect(db_name) as conn_status:
            cursor_status = conn_status.cursor()

            def get_table_count(cursor, table_name):
                count = 0
                exists = False
                try:
                    # Check if table exists first
                    cursor.execute(f"SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                    if cursor.fetchone()[0] == 1 :
                        exists = True
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        result = cursor.fetchone()
                        count = result[0] if result else 0
                    else:
                         error_messages.append(f"Table '{table_name}' not found.")
                except sqlite3.Error as e:
                    error_messages.append(f"Err counting {table_name}: {e}")
                return count, exists

            # Map display names to table names
            table_map = {
                'Accounts': 'accounts_data',
                'Flags Logged': 'dormant_flags',
                'Ledger Logged': 'dormant_ledger',
                'Insights Logged': 'insight_log',
                'SQL History': 'sql_query_history'
            }

            for display_name, table_name in table_map.items():
                 count, _ = get_table_count(cursor_status, table_name)
                 db_counts[display_name] = count

        # Format the status message
        status_lines = [f"- {name}: {count}" for name, count in db_counts.items()]
        st.sidebar.info("**DB Record Counts:**\n" + "\n".join(status_lines))
        if error_messages:
            for msg in error_messages: st.sidebar.warning(msg)

    except sqlite3.Error as e:
        st.sidebar.error(f"DB Status Connection Error: {e}")
    except Exception as e:
         st.sidebar.error(f"Unexpected error checking DB status: {e}")
else:
    st.sidebar.warning(f"Database file '{db_name}' not found. Counts are zero.")


st.sidebar.caption(f"DB File: {db_name}")