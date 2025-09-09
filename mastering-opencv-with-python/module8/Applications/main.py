try:
    # for streamlit >= 1.12.1
    from streamlit.web import bootstrap
except ImportError:
    from streamlit import bootstrap

real_script = '08_05_Image_Filters_Streamlit_app.py'
bootstrap.run(real_script, False, [], {})
