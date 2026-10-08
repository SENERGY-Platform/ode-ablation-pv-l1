# The operator image. Built and pushed by .github/workflows/build.yml.
#
# uv rather than plain pip, and "uv run" rather than plain "python": Ray worker
# processes inherit the launching interpreter, so a driver started outside the
# locked environment can hand workers an environment that lacks its dependencies.
FROM python:3.10-slim

# git for the Operator Lib dependency, which is a git reference rather than a PyPI
# release; librdkafka for confluent-kafka, which Operator Lib pins.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git librdkafka-dev gcc \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /usr/src/app

# Dependencies first, so a source change does not re-resolve them.
COPY pyproject.toml ./
COPY uv.lock* ./
RUN uv sync --no-dev

COPY . .

# Which commit this image is, read at startup by Operator Lib and reported in the
# operator's own log. It is also what makes an experiment's MLflow tag and a
# running operator comparable (§5.11 item 7).
ARG GIT_COMMIT=unknown
RUN printf 'commit=%s\n' "${GIT_COMMIT}" > git_commit

CMD ["uv", "run", "--no-dev", "main.py"]
