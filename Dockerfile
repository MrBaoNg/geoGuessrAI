# Stage 1: Build the Python Conda env
FROM continuumio/miniconda3:latest AS conda_build

WORKDIR /opt/conda_env
COPY environment.yml .
RUN conda env create -f environment.yml

# Stage 2: Build the final image with Node + Conda env
FROM node:18-bullseye

# Copy in the Conda environment we just built
COPY --from=conda_build /opt/conda_env /opt/conda_env

# Ensure the new env is on PATH
ENV PATH=/opt/conda_env/envs/geoenv/bin:$PATH

WORKDIR /app

# Install Node dependencies
COPY package.json yarn.lock ./
RUN yarn install --frozen-lockfile

# Copy application code
COPY . .

RUN mkdir -p /app/config /app/checkpoints

# Download config + weights
RUN curl -L "$DINO_CONFIG_URL"  -o /app/config/GroundingDINO_SwinB_OGC.py && \
    curl -L "$DINO_WEIGHTS_URL" -o /app/checkpoints/groundingdino_swinb_ogc.pth


# Expose port 5000 (Express default)
EXPOSE 5000

# Start your server
CMD ["node", "server.js"]
