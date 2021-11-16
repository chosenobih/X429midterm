FROM continuumio/miniconda3


WORKDIR /analysis

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

# Create the environment:
COPY environment.yml .
COPY README.md .
RUN conda env create -f environment.yml

# Initialize conda in bash config fiiles:
RUN conda init bash

# Activate the environment, and make sure it's activated:
RUN echo "conda activate info529midterm" > ~/.bashrc
RUN echo "Make sure joblib is installed:"
RUN python -c "import joblib; print('joblib imported')"

RUN echo "Beginning to add relevant directories to container"

# The code to run when container is started:
ADD scripts/ scripts/
ADD results/ results/
 
WORKDIR /analysis/scripts
RUN pwd
RUN ls -l
RUN chmod +x ensemble_train_bash_dev.sh


RUN echo "Beginning Ensemble Below"
ENTRYPOINT ["/analysis/scripts/ensemble_train_bash_dev.sh"]
