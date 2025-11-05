"""
Dynamic Systems Module
Interface for Fuzzy ODEs and p-Fuzzy systems
"""

import streamlit as st
from modules import fuzzy_ode_module




def run():

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 0.25rem 0 0.125rem 0; margin-top: 0.5rem;">
            <h2 style="margin: 0.25rem 0 0.125rem 0; color: #667eea;">Fuzzy ODE Solver</h2>
            <p style="color: #6b7280; font-size: 0.9rem; margin: 0;">
                Solve ODEs with fuzzy parameters
            </p>
        </div>
        <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 0.25rem 0 0.5rem 0;">
        """, unsafe_allow_html=True)

        # System type selection

        # st.markdown("<hr style='border: none; border-top: 1px solid #e5e7eb; margin: 0.5rem 0;'>", unsafe_allow_html=True)


    # Main content
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0;">
        <h3 style="color: #6b7280; font-weight: 500; margin: 0; font-size: 1.1rem;">
            Differencial Equations With Fuzzy Uncertainty
        </h3>
    </div>
    """, unsafe_allow_html=True)

    # st.markdown("<div style='border-bottom: 1px solid #e5e7eb; margin: 0.5rem 0 1.5rem 0;'></div>", unsafe_allow_html=True)

    # Render appropriate interface based on system type
    fuzzy_ode_module.run()
