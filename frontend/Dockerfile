FROM nginx:alpine

COPY frontend/ /usr/share/nginx/html/

# AudioWorklet requires HTTPS or localhost.
# Add headers so SharedArrayBuffer works if needed later.
RUN printf 'server {\n\
    listen 3000;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
    location / {\n\
        add_header Cross-Origin-Opener-Policy same-origin;\n\
        add_header Cross-Origin-Embedder-Policy require-corp;\n\
        try_files $uri $uri/ /index.html;\n\
    }\n\
    # pcm-processor.js must be served with correct MIME type\n\
    location ~* \\.js$ {\n\
        add_header Content-Type application/javascript;\n\
        add_header Cross-Origin-Opener-Policy same-origin;\n\
        add_header Cross-Origin-Embedder-Policy require-corp;\n\
    }\n\
}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 3000