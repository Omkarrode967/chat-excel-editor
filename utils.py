import re
import subprocess
import sys
import pandas as pd
import streamlit as st
from io import BytesIO
import traceback
import os
from groq import Groq, RateLimitError
import matplotlib.pyplot as plt
import io
from PIL import Image  
import re
from dotenv import load_dotenv

load_dotenv()

my_api_key = os.getenv("MY_API_KEY")

# Initialize Groq Client
def initialize_groq_client(api_key):
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Groq client: {e}")
        return None

# Socratic assistant response generation using Groq API
def socratic_assistant_response(client, input_text):
    system_prompt = """
    You are an intelligent assistant designed to help users modify Excel files using Python and pandas. Based on the user's input, generate Python code that reads the Excel file 'data.xlsx', applies the requested modifications, and returns the modified DataFrame.

    You are an intelligent assistant designed to help users modify Excel files using Python and pandas. Based on the user's input and the column names provided, generate Python code that reads the Excel file `data.xlsx`, applies the requested modifications, and returns the modified DataFrame.

### Requirements for Generated Python Code:
1. **File Handling**:
   - Import pandas and read the Excel file `data.xlsx`.
   - Ensure the code references **only the columns provided** in the dataset.

2. **Code Structure**:
   - Apply the requested modifications to the DataFrame.
   - Include proper error handling for cases such as:
     - Incorrect or missing column names.
     - Invalid data formats or unsupported operations.
   - Return the modified DataFrame without additional comments or explanations.

3. **Tasks**:
   - Adding/removing rows or columns.
   - Creating or modifying columns based on conditions.
   - Filtering data using specific conditions or ranges.
   - Aggregating or summarizing data (e.g., sums, averages, counts).
   - Formatting or cleaning data (e.g., date conversions, string formatting).
   - Merging, joining, or grouping data.
   - Sorting rows or columns.

4. **Validation**:
   - Verify the presence of required columns before performing operations.
   - Handle invalid operations gracefully with try-except blocks.

5. **Output**:
   - The output must be plain Python code, enclosed in triple backticks (` ``` `), ready to execute without modification.

### Context to Consider:
- **Column Names Provided**: Ensure all operations are based on the following column names from the dataset: `<column_names_here>` (replace this placeholder with the actual column names).
- Ensure data transformations strictly follow the provided dataset structure and column names.
- If the user's request references non-existent columns, return an error-handling mechanism that raises a clear and concise exception.

### Additional Notes:
- Ensure compatibility with different Excel sheet formats, especially for dates and times.
- Do not include unnecessary comments or explanations in the output—just the Python code.

### Examples of Tasks:
1. Convert a column to datetime format.
2. Filter rows based on a value range in a column.
3. Create a summary report of aggregated values.
4. Add a new column based on a condition.
5. Merge two sheets from the same Excel file.
6. Update values in a column based on a condition.

"""

    conversation = f"User: {input_text}\nAssistant:"

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": conversation}],
            model="llama3-70b-8192",
            temperature=0.5
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"

# Extract Python code from the assistant's response
def extract_python_code(llm_output):
    code_block_pattern = r"```(?:python)?(.*?)```"
    match = re.search(code_block_pattern, llm_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def install_module(module_name):
    print(module_name)
    """Function to install a Python module if it's not found."""
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "install", module_name],
                                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Successfully installed {module_name}.\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error installing {module_name}.\n{e.stderr}")
        raise


def execute_code_on_excel(code, input_df, client, user_query):
    if not code:
        return "No valid Python code block found.", None

    local_env = {'pd': pd, 'df': input_df, 'plt': plt}

    try:
        # Initialize output buffer before execution
        output_buffer = io.BytesIO() if "b'" in code else io.StringIO()
        local_env['output_buffer'] = output_buffer  # Pass output_buffer explicitly to exec's local_env

        # Execute the Python code
        exec(f"import sys\nsys.stdout = output_buffer\n{code}\nsys.stdout = sys.__stdout__", {}, local_env)

        # Get the output from the buffer
        output_buffer.seek(0)  
        executed_output = (
            output_buffer.getvalue().decode('utf-8').strip() if isinstance(output_buffer, io.BytesIO)
            else output_buffer.getvalue().strip()
        )

        # Collect all plot buffers
        img_buffers = []
        if plt.get_fignums():
            for i in plt.get_fignums():
                fig = plt.figure(i)
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format='png')  # Save the current figure to the buffer
                img_buffer.seek(0)  # Rewind the buffer
                img_buffers.append(img_buffer)
            plt.close()

        # Check for variables in local_env after execution
        output_object = local_env.get('output', None)  # Expect user to store output in 'output'
        if output_object is not None:
            return output_object, executed_output , img_buffers
        elif 'df' in local_env and isinstance(local_env['df'], pd.DataFrame):
            return local_env['df'], executed_output , img_buffers
        else:
            return "No output object was created or modified.", executed_output , img_buffers
    except ModuleNotFoundError as e:
        # Catch the missing module error and install the module
        missing_module = str(e).split('No module named ')[1].strip().strip("'")
        
        # Install the missing module
        st.info(f"Module '{missing_module}' not found. Installing it...")
        install_module(missing_module)
        
        # Retry the execution with the installed module
        return execute_code_on_excel(code, input_df, client, user_query)  # Recursive call to re-execute
    except Exception as e:
        # Catch other errors
        error_message = f"Error executing code: {e}"
        detailed_traceback = traceback.format_exc()

        # Log the error
        st.error(f"Execution failed: {error_message}")
        st.text("Traceback:")
        st.code(detailed_traceback, language="text")

        # Regenerate the code using the Groq API with error context
        st.info("Regenerating code based on the error...")

        # Provide the Groq client with the error context and user query
        error_context = f"Error occurred while executing this code: {detailed_traceback}\n\nOriginal user query: {user_query}"
        regenerated_code = socratic_assistant_response(client, error_context)

        # Extract the new code from the regenerated response
        new_code = extract_python_code(regenerated_code)
        
        if not new_code:
            return "Unable to regenerate valid Python code.", None

        # Execute the regenerated code
        st.info("Executing regenerated code...")
        return execute_code_on_excel(new_code, input_df, client, user_query)



# Main Application
def main():
    st.title("Excel Modification Assistant")

    # Initialize session state for data and chat history
    if 'modified_df' not in st.session_state:
        st.session_state.modified_df = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload File")
        uploaded_file = st.file_uploader("Choose a file (Excel or CSV)", type=["xlsx", "csv"])
        
        if uploaded_file:
            file_extension = uploaded_file.name.split(".")[-1].lower()
            
            try:
                if file_extension == "xlsx":
                    st.session_state.modified_df = pd.read_excel(uploaded_file)
                elif file_extension == "csv":
                    df = pd.read_csv(uploaded_file)  # Load CSV
                    # Save it as an Excel file in memory
                    excel_buffer = BytesIO()
                    df.to_excel(excel_buffer, index=False, engine='openpyxl')
                    excel_buffer.seek(0)  # Reset buffer position
                    st.session_state.modified_df = excel_buffer  # Store Excel version in session state

                
                st.success(f"Loaded file: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

    # Ensure the modified DataFrame is valid
    if isinstance(st.session_state.modified_df, pd.DataFrame):
        st.subheader("Original Excel Data")
        st.dataframe(st.session_state.modified_df)
    
    # User query for modifications
    user_query = st.chat_input("Describe the modification you'd like to make:")
    if user_query:
        st.session_state.chat_history.append(("user", user_query))

        # Initialize Groq Client with environment variable API key
        client = initialize_groq_client(my_api_key)
        if client:
            # Generate response
            sample_data = st.session_state.modified_df.head(5).to_string()
            complete_query = f"Sample Data:\n{sample_data}\n\nModification: {user_query}"
            response = socratic_assistant_response(client, complete_query)
            print(response)
            # Extract code and execute
            pure_code = extract_python_code(response)
            if not pure_code:
                st.error("No valid Python code was generated. Please refine your query.")
            else:
                st.session_state.chat_history.append(("assistant", pure_code))

                # Display chat history
                for sender, message in st.session_state.chat_history:
                    if sender == "user":
                        st.chat_message("User").write(message)
                    else:
                        st.chat_message("Assistant").code(message, language="python")

                # Execute Python code
                result = execute_code_on_excel(client=client, user_query=user_query, code=pure_code, input_df=st.session_state.modified_df)

                # Check the result type and handle accordingly
                if isinstance(result, tuple):
                    executed_output , output_object, img_buffers  = result
                    print(type(executed_output))

                    if isinstance(output_object, pd.DataFrame) and not output_object.empty:
                        # Update modified DataFrame
                        st.session_state.modified_df = output_object

                        # Display executed output
                        if executed_output:
                            st.subheader("Executed Output")
                            st.code(executed_output, language="text")

                        # Display modified DataFrame
                        st.subheader("Modified Excel Data")
                        st.dataframe(st.session_state.modified_df)

                        # Download button
                        output = io.BytesIO()
                        output_object.to_excel(output, index=False)
                        output.seek(0)
                        st.download_button(
                            label="Download modified Excel",
                            data=st.session_state.modified_df,
                            file_name="modified_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                     # Handle Image Buffer (for graphs)
                    if img_buffers:
                        for idx, img_buffer in enumerate(img_buffers):
                            st.subheader(f"Generated Plot {idx+1}")
                            img_buffer.seek(0)
                            st.image(img_buffer, caption=f"Generated Plot {idx+1}", use_column_width=True)

                            # Provide a download button for the plot image
                            st.download_button(
                                label=f"Download Plot {idx+1} as PNG",
                                data=img_buffer,
                                file_name=f"generated_plot_{idx+1}.png",
                                mime="image/png"
                            )

                            st.dataframe(st.session_state.modified_df)

                        # Display executed output
                        if output_object:
                            st.subheader("Executed Output")
                            st.code(executed_output, language="text")
                            st.dataframe(st.session_state.modified_df)

                    else:
                        # Handle other output types
                        st.subheader("Output Object")
                        if isinstance(output_object, (dict, list)):
                            st.json(output_object)
                        elif isinstance(output_object, io.StringIO):
                            st.text(output_object.getvalue())
                        else:
                            st.text(str(output_object))

                        # Display executed output
                        st.subheader("Executed Output")
                        st.code(executed_output, language="text")
                        st.dataframe(st.session_state.modified_df)
                else:
                    error_message, detailed_traceback = result
                    st.error(error_message)
                    if detailed_traceback:
                        st.text("Traceback:")
                        st.code(detailed_traceback, language="text")
                    st.dataframe(st.session_state.modified_df)    

            # Centralized Download Button Logic
        if isinstance(st.session_state.modified_df, pd.DataFrame):
            output = BytesIO()
            st.session_state.modified_df.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)  # Reset stream position to the beginning

            st.download_button(
                label="Download modified Excel",
                data=output.getvalue(),  # Use the content of the buffer
                file_name="modified_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        else:
            st.error("Could not initialize Groq client. Please check your API key.")
    else:
        st.warning("Please upload an Excel file to start.")

if __name__ == "__main__":
    main()
