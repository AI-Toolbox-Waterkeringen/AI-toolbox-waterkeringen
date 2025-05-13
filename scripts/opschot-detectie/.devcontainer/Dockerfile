FROM debian:bookworm-slim

# Stel de werkdirectory in
WORKDIR /code

# Kopieer alle files naar de werkdirectory
COPY .vscode/ /code/.vscode/
COPY src/ /code/src/
COPY .gitignore /code/.gitignore
COPY LICENSE /code/LICENSE
COPY pixi.lock /code/pixi.lock
COPY pixi.toml /code/pixi.toml
COPY README.md /code/README.md

RUN apt-get -y update
RUN apt-get -y install git curl
RUN curl -fsSL https://pixi.sh/install.sh | bash
