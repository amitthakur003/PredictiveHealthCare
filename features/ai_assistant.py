import streamlit as st
from utils.ai_helper import get_ai_insights

def render_ai_assistant(df, models):
    """Render the AI Assistant page"""
    st.header("AI-Powered Claims Assistant")
    
    st.markdown("""
    Ask questions about your claims data and get AI-powered insights in plain English.
    The assistant analyzes patterns, explains predictions, and provides actionable recommendations.
    """)
    
    # Data summary for context
    data_summary = f"""
    Total Claims: {len(df):,}
    Total Hospitals: {df['Hospital_ID'].nunique()}
    Total Insurers: {df['Insurer_ID'].nunique()}
    Total Claim Amount: ${df['Claim_Amount'].sum():,.0f}
    Total Paid Amount: ${df['Paid_Amount'].sum():,.0f}
    Average Payment Ratio: {df['Payment_Ratio'].mean():.2%}
    Denial Rate: {df['Denial_Flag'].mean():.1%}
    Disease Categories: {', '.join(df['Disease_Category'].unique())}
    Avg Days to Payment: {df['Days_To_Payment'].mean():.1f}
    """
    
    # Chat interface
    st.subheader("Chat with AI Assistant")
    
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat['role']):
            st.write(chat['content'])
    
    # Suggested questions
    st.subheader("Suggested Questions")
    col1, col2 = st.columns(2)
    
    # Use session state to manage temporary question storage
    if 'pending_question' not in st.session_state:
        st.session_state.pending_question = None

    if col1.button("Which insurer has highest denial rate?"):
        st.session_state.pending_question = "Which insurer has the highest denial rate and what could be the reasons?"
    
    if col2.button("How to reduce claim denials?"):
        st.session_state.pending_question = "What strategies can reduce claim denials based on this data?"
    
    col3, col4 = st.columns(2)
    
    if col3.button("Payment optimization tips?"):
        st.session_state.pending_question = "How can hospitals optimize their claim payments and reduce processing time?"
    
    if col4.button("Disease-specific insights?"):
        st.session_state.pending_question = "Which disease categories have the best payment ratios and why?"
    
    # Process pending question from button clicks
    if st.session_state.pending_question:
        st.session_state.chat_history.append({'role': 'user', 'content': st.session_state.pending_question})
        user_question = st.session_state.pending_question
        st.session_state.pending_question = None
        
        with st.spinner("AI is analyzing..."):
            ai_response = get_ai_insights(user_question, data_summary)
            st.session_state.chat_history.append({'role': 'assistant', 'content': ai_response})
        
        st.rerun()

    # User input (chat input)
    user_question = st.chat_input("Ask a question about your claims data...")
    
    if user_question:
        st.session_state.chat_history.append({'role': 'user', 'content': user_question})
        
        with st.spinner("AI is analyzing..."):
            ai_response = get_ai_insights(user_question, data_summary)
            st.session_state.chat_history.append({'role': 'assistant', 'content': ai_response})
        
        st.rerun()
    
    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
