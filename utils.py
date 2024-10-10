import re
import pandas as pd
import streamlit as st
from io import BytesIO
from groq import Groq, RateLimitError
import traceback

# Initialize Groq Client
def initialize_groq_client(api_key):
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        return e

def socratic_assistant_response(client, input_text):
    system_prompt = """
    You are a smart assistant designed to help users modify Excel files using Python and pandas. Based on the user's input, return Python code that reads the Excel file, applies the requested modifications, and returns the modified DataFrame.

    You must adhere to the following format:
    - Include the import of pandas.
    - Read the Excel file 'data.xlsx'.
    - Perform the requested modification(s).
    - Return the modified DataFrame.

    Do not return anything other than Python code wrapped in triple backticks. No explanations or comments are necessary.

    Examples of tasks include:
    - Adding/removing data
    - Adding columns
    - Updating cell values based on conditions
    - Filtering rows
    - Summarizing or aggregating data (e.g., sums, averages)
    - Formatting data (e.g., dates, strings)
    - Merging/joining Excel sheets
    - Sorting and rearranging rows/columns

    Example Input: "Add a column named 'Full Name' with entry ashish"
    Expected Output:
    ```python
    import pandas as pd
    df = pd.read_excel('data.xlsx')
    df['Full Name'] = 'ashish'
    ```
    """
    conversation = f"User: {input_text}\nAssistant:"

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation}
            ],
            model="llama3-70b-8192",
            temperature=0.5
        )
        response = chat_completion.choices[0].message.content
        return response
    except Exception as e:
        return f"Error generating response: {e}"

# Extract Python code from LLM response
def extract_python_code(llm_output):
    code_block_pattern = r"```(?:python)?(.*?)```"
    match = re.search(code_block_pattern, llm_output, re.DOTALL)
    return match.group(1).strip() if match else "No valid Python code block found."

# Execute generated Python code on the uploaded Excel sheet with debugging
def execute_code_on_excel(code, df):
    local_env = {'pd': pd, 'df': df}
    try:
        exec(code, {}, local_env)
        return local_env['df']
    except Exception as e:
        error_message = f"Error executing code: {e}"
        detailed_traceback = traceback.format_exc()
        return error_message, detailed_traceback

# Save the modified DataFrame to data.xlsx
def save_modified_excel(df):
    output = BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)
    
    with open('data.xlsx', 'wb') as f:
        f.write(output.read())
    st.success("The modifications have been saved to 'data.xlsx'.")

def main():
    st.title("Excel Modification Assistant")

    # Initialize session state for data
    if 'modified_df' not in st.session_state:
        st.session_state.modified_df = None

    # Upload Excel file
    uploaded_file = st.file_uploader("Upload Excel file", type="xlsx")
    
    if uploaded_file is not None:
        # Load the file only if it's not already in session state
        if st.session_state.modified_df is None:
            df = pd.read_excel(uploaded_file)
            st.session_state.modified_df = df  # Set as initial DataFrame
        st.write("Original Excel Data:")
        st.dataframe(st.session_state.modified_df)


        # Chat window: Simulate a conversation with the assistant
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Chat input box
        user_query = st.chat_input("Describe the modification you'd like to make:")
        
        if user_query:
            # Append the user query to chat history
            st.session_state.chat_history.append(("user", user_query))

            # Display the chat history
            for sender, message in st.session_state.chat_history:
                if sender == "user":
                    st.chat_message("User", avatar="🧑").write(message)
                else:
                    st.chat_message("Assistant", avatar="🤖").code(message)

            # Generate and display response when user submits a query
            client = initialize_groq_client("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")
            
            if isinstance(client, Groq):
                sample_data = st.session_state.modified_df.head(5).to_string()
                complete_query = f"Here is a sample of the Excel sheet:\n\n{sample_data}\n\nModification request: {user_query}"
                
                # Generate response using Groq API
                generated_response = socratic_assistant_response(client, complete_query)
                
                # Extract Python code from the response
                pure_code = extract_python_code(generated_response)
                print(pure_code)
                # Append assistant response to chat history
                st.session_state.chat_history.append(("assistant", pure_code))



                # Toggle code visibility when the button is clicked
                if pure_code:
                    
   
                            st.code(pure_code, language="python")

                # Execute the code and display modified Excel
                try:
                    result = execute_code_on_excel(pure_code, st.session_state.modified_df)
                    
                    if isinstance(result, pd.DataFrame):
                        st.session_state.modified_df = result
                        
                        
                        save_modified_excel(result)

                        output = BytesIO()
                        result.to_excel(output, index=False)
                        output.seek(0)
                        st.download_button(
                            label="Download modified Excel",
                            data=output,
                            file_name="modified_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        error_message, detailed_traceback = result
                        st.error(error_message)
                        st.text("Traceback:")
                        st.code(detailed_traceback, language="text")
                except Exception as e:
                    error_message = f"Error executing code: {e}"
                    detailed_traceback = traceback.format_exc()
                    st.error(error_message)
                    st.text("Traceback:")
                    st.code(detailed_traceback, language="text")
                
                with st.sidebar:
                            st.header("Modified Data")
                            if st.session_state.modified_df is not None:
                                st.dataframe(st.session_state.modified_df)
                            else:
                                st.write("No modifications applied yet.")
            else:
                st.error("Failed to initialize Groq client. Please check your API key.")

if __name__ == "__main__":
    main()
