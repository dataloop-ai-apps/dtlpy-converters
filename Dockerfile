FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.11_opencv

USER root

# Install any system dependencies if needed
# RUN apt-get update && apt-get install -y <packages>

# Create directory and set ownership
RUN mkdir -p /tmp/app && chown 1000:1000 /tmp/app

USER 1000

# Install Python dependencies
RUN pip install \
    nest_asyncio \
    pycocotools

# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/dtlpy-converters:3.1.0 -f ./Dockerfile  .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/dtlpy-converters:3.1.0