mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"jimgao0606@gmail.com\"\n\
" >credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" >config.toml
